"""PrivateAttend: End-to-End Biometric Attendance Engine.

Orchestrates the full pipeline:
Detection -> Quality Screening -> Presentation Attack Detection (Liveness) ->
ArcFace Feature Extraction -> Decrypt Gallery Templates -> Open-Set Decision ->
Consent Verification -> Duplicate Prevention -> Atomic Database Transaction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import csv
import io
import numpy as np

from config import config
from database import Database
from face_engine import DecisionState, FaceEngine
from liveness import LivenessDetector
from modules.detection import FaceDetectionResult, FaceDetector
from modules.preprocessing import assess_image_quality, load_image
from modules.recognition import FaceRecognizer
from privacy import CryptographicError, KeyManager, TemplateProtector


@dataclass
class AttendanceVerificationResult:
    """Comprehensive outcome of a biometric attendance scan."""
    success: bool
    status_code: str
    student_id: Optional[str]
    student_name: Optional[str]
    confidence_score: float
    liveness_score: float
    message: str
    detection_bbox: Optional[Tuple[int, int, int, int]] = None


class AttendanceEngine:
    """High-level verification engine coordinating biometrics, PAD, privacy, and persistence."""

    def __init__(
        self,
        db: Optional[Database] = None,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        liveness_detector: Optional[LivenessDetector] = None,
        protector: Optional[TemplateProtector] = None,
    ):
        self.db = db or Database()
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.liveness_detector = liveness_detector or LivenessDetector()
        self.protector = protector or TemplateProtector()
        self.face_engine = FaceEngine(detector=self.detector, recognizer=self.recognizer)

    def process_attendance_frame(
        self,
        image_input: Union[str, bytes, np.ndarray],
        session_id: int,
        allow_synthetic: bool = False,
    ) -> AttendanceVerificationResult:
        """Executes the full biometric verification pipeline for an incoming frame/photo.

        Pipeline Stages:
            1. Verify attendance session is open and active.
            2. Localize face and extract landmarks.
            3. Evaluate photometric and frequency image quality.
            4. Verify physical liveness / presentation attack detection.
            5. Extract normalized 512-d feature vector.
            6. Ephemerally decrypt active enrolled gallery templates.
            7. Evaluate open-set identification boundary (KNOWN, UNKNOWN, UNCERTAIN).
            8. Verify active student consent status.
            9. Prevent duplicate attendance entries.
            10. Commit attendance record and audit event.
        """
        # Stage 1: Verify Session
        session = self.db.get_session(session_id)
        if not session:
            return AttendanceVerificationResult(
                success=False,
                status_code="SESSION_NOT_FOUND",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=0.0,
                message=f"Attendance session ID {session_id} does not exist.",
            )
        if not session["is_active"]:
            return AttendanceVerificationResult(
                success=False,
                status_code="SESSION_CLOSED",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=0.0,
                message=f"Attendance session '{session['session_name']}' is closed.",
            )

        # Stage 2: Face Detection
        try:
            detections = self.detector.detect(image_input, max_faces=1)
        except Exception as e:
            return AttendanceVerificationResult(
                success=False,
                status_code="DETECTION_ERROR",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=0.0,
                message=f"Face detection error: {e}",
            )

        if not detections:
            return AttendanceVerificationResult(
                success=False,
                status_code="NO_FACE_DETECTED",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=0.0,
                message="No face localized in the provided image.",
            )

        best_face = detections[0]

        # Stage 3: Image Quality Validation
        if not best_face.quality.is_acceptable:
            reasons = "; ".join(best_face.quality.failure_reasons)
            return AttendanceVerificationResult(
                success=False,
                status_code="LOW_QUALITY_IMAGE",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=0.0,
                message=f"Image quality rejected: {reasons}",
                detection_bbox=best_face.bbox,
            )

        # Stage 4: Presentation Attack Detection (Liveness)
        liveness_score = self.liveness_detector.predict_frame(
            best_face.aligned_face, allow_synthetic=allow_synthetic
        )
        liveness_eval = self.liveness_detector.update_temporal_window(liveness_score)

        if liveness_eval.attack_detected:
            self.db.log_audit(
                "SPOOF_ATTACK_BLOCKED",
                "SECURITY_ENGINE",
                None,
                f"Liveness failure in session {session_id} (Score: {liveness_score:.3f}).",
            )
            return AttendanceVerificationResult(
                success=False,
                status_code="SPOOF_REJECTED",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=liveness_score,
                message=f"Presentation attack rejected: {liveness_eval.details}",
                detection_bbox=best_face.bbox,
            )

        # Stage 5: Feature Extraction
        try:
            probe_embedding = self.recognizer.extract_embedding(
                best_face.aligned_face, allow_synthetic=allow_synthetic
            )
        except Exception as e:
            return AttendanceVerificationResult(
                success=False,
                status_code="FEATURE_EXTRACTION_ERROR",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=liveness_score,
                message=f"Failed to generate facial embedding: {e}",
                detection_bbox=best_face.bbox,
            )

        # Stage 6: Ephemeral Decryption of Gallery Templates
        encrypted_gallery = self.db.get_all_active_templates()
        if not encrypted_gallery:
            return AttendanceVerificationResult(
                success=False,
                status_code="EMPTY_GALLERY",
                student_id=None,
                student_name=None,
                confidence_score=0.0,
                liveness_score=liveness_score,
                message="No enrolled student templates found with active consent.",
                detection_bbox=best_face.bbox,
            )

        decrypted_gallery: Dict[str, np.ndarray] = {}
        try:
            for sid, enc_bytes in encrypted_gallery.items():
                decrypted_gallery[sid] = self.protector.decrypt_template(enc_bytes)

            # Stage 7: Open-Set Recognition Matching
            match_res = self.face_engine.match_probe(probe_embedding, decrypted_gallery)
        finally:
            # Stage 7b: Cryptographic memory scrub of plaintext gallery templates
            for vec in decrypted_gallery.values():
                TemplateProtector.zero_memory(vec)

        if match_res.decision == DecisionState.UNKNOWN:
            self.db.log_audit(
                "UNKNOWN_IDENTITY_REJECTED",
                "SECURITY_ENGINE",
                None,
                f"Probe rejected in session {session_id} (Top score: {match_res.matched_score:.3f}).",
            )
            return AttendanceVerificationResult(
                success=False,
                status_code="UNKNOWN_REJECTED",
                student_id=None,
                student_name=None,
                confidence_score=match_res.matched_score,
                liveness_score=liveness_score,
                message=match_res.explanation,
                detection_bbox=best_face.bbox,
            )

        if match_res.decision == DecisionState.UNCERTAIN:
            return AttendanceVerificationResult(
                success=False,
                status_code="UNCERTAIN_PREDICTION",
                student_id=match_res.student_id,
                student_name=None,
                confidence_score=match_res.matched_score,
                liveness_score=liveness_score,
                message=match_res.explanation,
                detection_bbox=best_face.bbox,
            )

        # At this point: match_res.decision == KNOWN
        matched_student_id = match_res.student_id

        # Stage 8: Verify Active Consent
        if not self.db.has_active_consent(matched_student_id):
            return AttendanceVerificationResult(
                success=False,
                status_code="CONSENT_REVOKED_BLOCKED",
                student_id=matched_student_id,
                student_name=None,
                confidence_score=match_res.matched_score,
                liveness_score=liveness_score,
                message="Biometric consent for this student is revoked or not on file.",
                detection_bbox=best_face.bbox,
            )

        student_info = self.db.get_student(matched_student_id)
        student_name = (
            f"{student_info['first_name']} {student_info['last_name']}" if student_info else matched_student_id
        )

        # Stage 9: Duplicate Attendance Prevention
        if self.db.is_attendance_marked(session_id, matched_student_id):
            return AttendanceVerificationResult(
                success=False,
                status_code="ALREADY_MARKED",
                student_id=matched_student_id,
                student_name=student_name,
                confidence_score=match_res.matched_score,
                liveness_score=liveness_score,
                message=f"Attendance already recorded for {student_name} in this session.",
                detection_bbox=best_face.bbox,
            )

        # Stage 10: Commit Attendance Transaction
        recorded = self.db.record_attendance(
            session_id=session_id,
            student_id=matched_student_id,
            status="PRESENT",
            confidence_score=match_res.matched_score,
            liveness_score=liveness_score,
            verification_method="BIOMETRIC",
            notes=f"Automated match ({match_res.matched_score:.2f})",
        )

        if not recorded:
            return AttendanceVerificationResult(
                success=False,
                status_code="TRANSACTION_ERROR",
                student_id=matched_student_id,
                student_name=student_name,
                confidence_score=match_res.matched_score,
                liveness_score=liveness_score,
                message="Failed to persist attendance record (duplicate conflict).",
                detection_bbox=best_face.bbox,
            )

        return AttendanceVerificationResult(
            success=True,
            status_code="ATTENDANCE_CONFIRMED",
            student_id=matched_student_id,
            student_name=student_name,
            confidence_score=match_res.matched_score,
            liveness_score=liveness_score,
            message=f"Attendance verified and confirmed for {student_name}.",
            detection_bbox=best_face.bbox,
        )

    def manual_mark_attendance(
        self,
        session_id: int,
        student_id: str,
        reason: str,
        actor: str = "admin",
    ) -> bool:
        """Allows an authorized supervisor to manually record attendance."""
        student = self.db.get_student(student_id)
        if not student:
            return False

        if self.db.is_attendance_marked(session_id, student_id):
            return False

        return self.db.record_attendance(
            session_id=session_id,
            student_id=student_id,
            status="MANUAL_VERIFIED",
            confidence_score=1.0,
            liveness_score=1.0,
            verification_method="MANUAL",
            notes=f"Manual override by {actor}: {reason}",
        )

    def export_attendance_csv(self, session_id: int) -> str:
        """Exports session attendance records to a formatted CSV string."""
        records = self.db.get_session_attendance(session_id)
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Session ID",
            "Student ID",
            "First Name",
            "Last Name",
            "Email",
            "Department",
            "Timestamp",
            "Status",
            "Confidence Score",
            "Liveness Score",
            "Verification Method",
            "Notes",
        ])

        for r in records:
            writer.writerow([
                r["session_id"],
                r["student_id"],
                r["first_name"],
                r["last_name"],
                r["email"],
                r["department"],
                r["timestamp"],
                r["status"],
                f"{r['confidence_score']:.3f}",
                f"{r['liveness_score']:.3f}",
                r["verification_method"],
                r["notes"] or "",
            ])

        return output.getvalue()
