"""PrivateAttend: High-Performance FastAPI REST Service.

Exposes RESTful endpoints for student management, consent tracking, multi-sample
biometric enrollment, presentation attack screening, attendance verification, and audit trail retrieval.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
import hmac
import os
import time
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, EmailStr, Field
from enrollment import build_template
from attendance import AttendanceEngine, AttendanceVerificationResult
from config import config
from database import Database
from modules.detection import FaceDetector
from modules.preprocessing import assess_image_quality, load_image
from modules.recognition import FaceRecognizer
from privacy import KeyManager, TemplateProtector


# ---------------- API Models ---------------- #

class AdminLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int


class StudentCreate(BaseModel):
    student_id: str = Field(..., min_length=2, max_length=32)
    first_name: str = Field(..., min_length=1, max_length=64)
    last_name: str = Field(..., min_length=1, max_length=64)
    email: str = Field(..., min_length=3, max_length=128)
    department: str = Field(..., min_length=2, max_length=64)


class StudentResponse(BaseModel):
    id: int
    student_id: str
    first_name: str
    last_name: str
    email: str
    department: str
    is_active: bool
    created_at: str


class ConsentCreate(BaseModel):
    student_id: str
    status: str = Field(..., pattern="^(GRANTED|WITHDRAWN|REVOKED)$")
    terms_version: str = "v1.0"
    notes: Optional[str] = None


class SessionCreate(BaseModel):
    course_code: str
    session_name: str


class ManualAttendanceCreate(BaseModel):
    session_id: int
    student_id: str
    reason: str


class SystemStatusResponse(BaseModel):
    service_status: str
    detector_backend: str
    yunet_weights_loaded: bool
    arcface_weights_loaded: bool
    liveness_weights_loaded: bool
    total_enrolled_students: int
    active_sessions: int
    dataset_status: str


# ---------------- Application Initialization ---------------- #

app = FastAPI(
    title="PrivateAttend Biometric API",
    description="Privacy-Preserving On-Device Biometric Attendance API with Open-Set Verification and Liveness Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Singletons
db = Database()
detector = FaceDetector()
recognizer = FaceRecognizer()
protector = TemplateProtector()
attendance_engine = AttendanceEngine(db=db, detector=detector, recognizer=recognizer, protector=protector)


# ---------------- Authentication Dependency ---------------- #

def create_access_token(username: str) -> str:
    """Creates a time-limited HMAC-signed bearer token."""
    timestamp = int(time.time())
    payload = f"{username}:{timestamp}"
    signature = hmac.new(config.JWT_SECRET.encode(), payload.encode(), "sha256").hexdigest()
    token_str = f"{payload}:{signature}"
    return base64.urlsafe_b64encode(token_str.encode()).decode()


def verify_admin_token(authorization: Optional[str] = Header(None)) -> str:
    """Validates authorization header containing Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication token.",
        )
    token = authorization.split(" ", 1)[1]
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")
        if len(parts) != 3:
            raise ValueError
        username, ts_str, signature = parts
        payload = f"{username}:{ts_str}"
        expected_sig = hmac.new(config.JWT_SECRET.encode(), payload.encode(), "sha256").hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            raise ValueError
        if time.time() - int(ts_str) > config.JWT_EXPIRY_MINUTES * 60:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
        return username
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature.")


# ---------------- System & Health Endpoints ---------------- #

@app.get("/api/v1/health", tags=["System"])
def health_check():
    """Liveness probe verifying service availability."""
    return {"status": "healthy", "service": "PrivateAttend", "timestamp": time.time()}


@app.get("/api/v1/system/status", response_model=SystemStatusResponse, tags=["System"])
def system_status():
    """Reports model weight availability, database metrics, and dataset status."""
    students = db.list_students()
    active_sess = db.get_active_session()
    return SystemStatusResponse(
        service_status="operational",
        detector_backend=detector.backend,
        yunet_weights_loaded=(detector.backend == "yunet"),
        arcface_weights_loaded=recognizer.weights_loaded,
        liveness_weights_loaded=attendance_engine.liveness_detector.weights_loaded,
        total_enrolled_students=len(students),
        active_sessions=1 if active_sess else 0,
        dataset_status="Dataset-independent code ready. UMDFace research subset pending integration.",
    )


# ---------------- Authentication Endpoints ---------------- #

@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login(creds: AdminLoginRequest):
    """Authenticates admin and returns a signed bearer token."""
    if db.verify_admin(creds.username, creds.password):
        token = create_access_token(creds.username)
        return TokenResponse(access_token=token, expires_in=config.JWT_EXPIRY_MINUTES * 60)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")


# ---------------- Student Management Endpoints ---------------- #

@app.post("/api/v1/students", status_code=status.HTTP_201_CREATED, tags=["Students"])
def register_student(payload: StudentCreate, user: str = Depends(verify_admin_token)):
    """Registers a new student profile."""
    if db.get_student(payload.student_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Student ID {payload.student_id} is already registered.",
        )
    try:
        db.add_student(
            student_id=payload.student_id,
            first_name=payload.first_name,
            last_name=payload.last_name,
            email=payload.email,
            department=payload.department,
        )
        return {"message": "Student successfully registered.", "student_id": payload.student_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/students", tags=["Students"])
def list_students(active_only: bool = True):
    """Lists registered students."""
    return db.list_students(active_only=active_only)


@app.get("/api/v1/students/{student_id}", tags=["Students"])
def get_student(student_id: str):
    """Retrieves a single student's profile and consent status."""
    student = db.get_student(student_id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")
    has_consent = db.has_active_consent(student_id)
    return {**student, "has_active_consent": has_consent}


@app.delete("/api/v1/students/{student_id}", tags=["Students"])
def delete_student(student_id: str, user: str = Depends(verify_admin_token)):
    """Deletes student profile, automatically purging all associated biometric templates."""
    if not db.get_student(student_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")
    success = db.delete_student(student_id)
    return {"message": "Student and biometric data successfully purged.", "student_id": student_id}


# ---------------- Consent Management Endpoints ---------------- #

@app.post("/api/v1/consent", tags=["Consent"])
def record_consent(payload: ConsentCreate):
    """Records informed consent or revokes it. Revoking purges biometric templates immediately."""
    student = db.get_student(payload.student_id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

    cid = db.record_consent(
        student_id=payload.student_id,
        status=payload.status,
        terms_version=payload.terms_version,
        notes=payload.notes,
    )
    return {
        "consent_id": cid,
        "student_id": payload.student_id,
        "status": payload.status,
        "template_purged": payload.status in ("WITHDRAWN", "REVOKED"),
    }


# ---------------- Biometric Enrollment Endpoints ---------------- #

@app.post("/api/v1/biometrics/enroll", tags=["Biometrics"])
async def enroll_biometrics(
    student_id: str = Form(...),
    files: List[UploadFile] = File(...),
    allow_synthetic: bool = Form(False),
):
    """Enrolls multi-sample facial photographs for a consenting student."""
    student = db.get_student(student_id)
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

    if not db.has_active_consent(student_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student has not granted active biometric consent.",
        )

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image files uploaded.")

    images = [await f.read() for f in files]
    res = build_template(images, detector, recognizer)
    if res.template is None:
        raise HTTPException(status_code=400, detail={"message": "No usable images.", "reasons": res.rejections})

    db.store_template(
        student_id=student_id,
        encrypted_template=protector.encrypt_template(res.template),
        sample_count=res.accepted,
        quality_score=res.mean_quality,
    )
    return {
        "message": f"Enrolled {res.accepted} samples for {student_id}.",
        "student_id": student_id,
        "samples_accepted": res.accepted,
        "mean_quality_score": res.mean_quality,
        "rejected": res.rejections,
    }



# ---------------- Attendance Verification Endpoints ---------------- #

@app.post("/api/v1/attendance/sessions", status_code=status.HTTP_201_CREATED, tags=["Attendance"])
def create_session(payload: SessionCreate, user: str = Depends(verify_admin_token)):
    """Creates a new course attendance session."""
    sid = db.create_session(
        course_code=payload.course_code,
        session_name=payload.session_name,
        created_by=user,
    )
    return {"session_id": sid, "course_code": payload.course_code, "session_name": payload.session_name}


@app.get("/api/v1/attendance/sessions", tags=["Attendance"])
def list_sessions():
    """Lists all attendance sessions."""
    return db.list_sessions()


@app.post("/api/v1/attendance/sessions/{session_id}/close", tags=["Attendance"])
def close_session(session_id: int, user: str = Depends(verify_admin_token)):
    """Concludes an attendance session."""
    success = db.close_session(session_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return {"message": "Session closed.", "session_id": session_id}


@app.post("/api/v1/attendance/verify", tags=["Attendance"])
async def verify_attendance(
    session_id: int = Form(...),
    file: UploadFile = File(...),
    allow_synthetic: bool = Form(False),
):
    """Processes probe image through PAD, recognition, open-set screening, and records attendance."""
    contents = await file.read()
    result: AttendanceVerificationResult = attendance_engine.process_attendance_frame(
        image_input=contents, session_id=session_id, allow_synthetic=allow_synthetic
    )

    status_http = status.HTTP_200_OK if result.success else status.HTTP_400_BAD_REQUEST
    if result.status_code == "ALREADY_MARKED":
        status_http = status.HTTP_200_OK  # Not a fatal error

    return {
        "success": result.success,
        "status_code": result.status_code,
        "student_id": result.student_id,
        "student_name": result.student_name,
        "confidence_score": result.confidence_score,
        "liveness_score": result.liveness_score,
        "message": result.message,
    }


@app.post("/api/v1/attendance/manual", tags=["Attendance"])
def manual_attendance(payload: ManualAttendanceCreate, user: str = Depends(verify_admin_token)):
    """Allows an administrator to manually record attendance."""
    success = attendance_engine.manual_mark_attendance(
        session_id=payload.session_id,
        student_id=payload.student_id,
        reason=payload.reason,
        actor=user,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to record manual attendance (already marked or student not found).",
        )
    return {"message": "Manual attendance successfully recorded.", "student_id": payload.student_id}


@app.get("/api/v1/attendance/records", tags=["Attendance"])
def get_attendance_records(session_id: Optional[int] = Query(None)):
    """Retrieves attendance records for a session."""
    if session_id:
        return db.get_session_attendance(session_id)
    active = db.get_active_session()
    if active:
        return db.get_session_attendance(active["id"])
    return []


@app.get("/api/v1/attendance/export", response_class=PlainTextResponse, tags=["Attendance"])
def export_attendance(session_id: int):
    """Exports session attendance as standard CSV stream."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    csv_data = attendance_engine.export_attendance_csv(session_id)
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_session_{session_id}.csv"},
    )


# ---------------- Audit Endpoints ---------------- #

@app.get("/api/v1/audit/logs", tags=["Audit"])
def get_audit_logs(limit: int = 100, user: str = Depends(verify_admin_token)):
    """Retrieves immutable system audit event stream."""
    return db.get_audit_logs(limit=limit)
