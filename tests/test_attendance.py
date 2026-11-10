"""Unit tests for attendance.py (End-to-End Attendance Pipeline)."""

import numpy as np
import pytest

from attendance import AttendanceEngine, AttendanceVerificationResult
from database import Database
from modules.detection import FaceDetector
from modules.recognition import FaceRecognizer
from privacy import KeyManager, TemplateProtector


def test_attendance_pipeline_success(test_db: Database, template_protector: TemplateProtector, synthetic_face_image):
    """Test full attendance pipeline: detection -> quality -> PAD -> recognition -> DB transaction."""
    # 1. Setup Student & Consent
    test_db.add_student("STU_ATT_01", "Katherine", "Johnson", "kjohnson@nasa.gov", "Aerospace")
    test_db.record_consent("STU_ATT_01", status="GRANTED")

    # 2. Extract and store encrypted template for synthetic face
    recognizer = FaceRecognizer(allow_mock=True)
    detector = FaceDetector(backend="haar")
    emb = recognizer.extract_embedding(synthetic_face_image, allow_synthetic=True)
    enc_bytes = template_protector.encrypt_template(emb)
    test_db.store_template("STU_ATT_01", enc_bytes, sample_count=1, quality_score=90.0)

    # 3. Create Session
    session_id = test_db.create_session("ASTRO-101", "Orbital Mechanics")

    engine = AttendanceEngine(
        db=test_db,
        detector=detector,
        recognizer=recognizer,
        protector=template_protector,
    )

    # 4. Verify probe (same face)
    res: AttendanceVerificationResult = engine.process_attendance_frame(
        synthetic_face_image, session_id=session_id, allow_synthetic=True
    )

    assert res.success is True
    assert res.status_code == "ATTENDANCE_CONFIRMED"
    assert res.student_id == "STU_ATT_01"
    assert res.confidence_score >= 0.55
    assert test_db.is_attendance_marked(session_id, "STU_ATT_01")


def test_attendance_duplicate_prevention(test_db: Database, template_protector: TemplateProtector, synthetic_face_image):
    """Test duplicate scan in the same session is politely ignored."""
    test_db.add_student("STU_ATT_DUP", "John", "von Neumann", "jvn@ias.edu", "Math")
    test_db.record_consent("STU_ATT_DUP", status="GRANTED")

    recognizer = FaceRecognizer(allow_mock=True)
    detector = FaceDetector(backend="haar")
    emb = recognizer.extract_embedding(synthetic_face_image, allow_synthetic=True)
    enc_bytes = template_protector.encrypt_template(emb)
    test_db.store_template("STU_ATT_DUP", enc_bytes, sample_count=1, quality_score=85.0)

    session_id = test_db.create_session("COMP-501", "Automata Theory")
    engine = AttendanceEngine(db=test_db, detector=detector, recognizer=recognizer, protector=template_protector)

    # First attempt: success
    res1 = engine.process_attendance_frame(synthetic_face_image, session_id=session_id, allow_synthetic=True)
    assert res1.success is True

    # Second attempt: ALREADY_MARKED
    res2 = engine.process_attendance_frame(synthetic_face_image, session_id=session_id, allow_synthetic=True)
    assert res2.success is False
    assert res2.status_code == "ALREADY_MARKED"


def test_attendance_blocked_when_consent_withdrawn(test_db: Database, template_protector: TemplateProtector, synthetic_face_image):
    """Test that student with revoked consent cannot mark biometric attendance."""
    test_db.add_student("STU_NO_CONSENT", "No", "Consent", "noconsent@univ.edu", "Law")
    # Student withdraws consent
    test_db.record_consent("STU_NO_CONSENT", status="WITHDRAWN")

    session_id = test_db.create_session("LAW-101", "Privacy Jurisprudence")
    engine = AttendanceEngine(db=test_db, protector=template_protector)

    res = engine.process_attendance_frame(synthetic_face_image, session_id=session_id, allow_synthetic=True)
    # Since consent is withdrawn, template is absent from active gallery
    assert res.success is False
    assert res.status_code in ("EMPTY_GALLERY", "UNKNOWN_REJECTED", "CONSENT_REVOKED_BLOCKED")


def test_manual_attendance_marking(test_db: Database):
    """Test administrative manual attendance override."""
    test_db.add_student("STU_MANUAL", "Ada", "Lovelace", "ada@oxford.edu", "Math")
    session_id = test_db.create_session("MATH-200", "Analytical Engines")

    engine = AttendanceEngine(db=test_db)
    success = engine.manual_mark_attendance(session_id, "STU_MANUAL", reason="Medical exemption", actor="admin")
    assert success is True
    assert test_db.is_attendance_marked(session_id, "STU_MANUAL")

    csv_out = engine.export_attendance_csv(session_id)
    assert "STU_MANUAL" in csv_out
    assert "MANUAL_VERIFIED" in csv_out
