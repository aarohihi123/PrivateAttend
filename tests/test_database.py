"""Unit tests for database.py (SQLite Schema, Constraints, and Consent Purge)."""

import pytest
from database import Database


def test_student_registration_and_retrieval(test_db: Database):
    """Test registering student and fetching record."""
    sid = test_db.add_student(
        student_id="STU_TEST_01",
        first_name="Alan",
        last_name="Turing",
        email="alan.turing@univ.edu",
        department="Computer Science",
    )
    assert sid > 0

    student = test_db.get_student("STU_TEST_01")
    assert student is not None
    assert student["first_name"] == "Alan"
    assert student["department"] == "Computer Science"


def test_consent_and_automatic_template_purge(test_db: Database):
    """Test that withdrawing consent automatically purges encrypted biometric template."""
    test_db.add_student("STU_PURGE", "Grace", "Hopper", "grace@univ.edu", "Engineering")

    # 1. Grant consent
    test_db.record_consent("STU_PURGE", status="GRANTED")
    assert test_db.has_active_consent("STU_PURGE")

    # 2. Store dummy template
    test_db.store_template("STU_PURGE", encrypted_template=b"dummy_encrypted_bytes", sample_count=3)
    assert test_db.get_template("STU_PURGE") == b"dummy_encrypted_bytes"

    # 3. Withdraw consent
    test_db.record_consent("STU_PURGE", status="WITHDRAWN", notes="User revocation")
    assert not test_db.has_active_consent("STU_PURGE")

    # 4. Verify template has been purged from database
    assert test_db.get_template("STU_PURGE") is None


def test_duplicate_attendance_prevention(test_db: Database):
    """Test that duplicate attendance in the same session is prevented by database constraint."""
    test_db.add_student("STU_DUP", "Claude", "Shannon", "claude@univ.edu", "Math")
    test_db.record_consent("STU_DUP", status="GRANTED")

    sess_id = test_db.create_session("CS-101", "Information Theory")

    # First record should succeed
    first = test_db.record_attendance(
        session_id=sess_id,
        student_id="STU_DUP",
        status="PRESENT",
        confidence_score=0.88,
        liveness_score=0.95,
        verification_method="BIOMETRIC",
    )
    assert first is True

    # Duplicate in same session must fail cleanly
    duplicate = test_db.record_attendance(
        session_id=sess_id,
        student_id="STU_DUP",
        status="PRESENT",
        confidence_score=0.89,
        liveness_score=0.94,
        verification_method="BIOMETRIC",
    )
    assert duplicate is False


def test_audit_logging(test_db: Database):
    """Test that security audit events are persisted and retrievable."""
    test_db.log_audit("TEST_EVENT", "ADMIN", "ENTITY_42", "Verification of audit logging system.")
    logs = test_db.get_audit_logs(limit=10)
    assert any(log["event_type"] == "TEST_EVENT" for log in logs)
