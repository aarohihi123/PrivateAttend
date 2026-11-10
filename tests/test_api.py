"""Unit tests for api.py (FastAPI REST Backend)."""

import pytest
from fastapi.testclient import TestClient

from api import app, create_access_token
from config import config
from database import Database


@pytest.fixture
def client():
    """Provides a TestClient connected to the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Generates valid Bearer authentication headers for test requests."""
    token = create_access_token(config.ADMIN_USERNAME)
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint(client: TestClient):
    """Test /api/v1/health returns HTTP 200 and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "PrivateAttend"


def test_system_status_endpoint(client: TestClient):
    """Test /api/v1/system/status diagnostics output."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["service_status"] == "operational"
    assert "dataset_status" in data
    assert "detector_backend" in data


def test_admin_login(client: TestClient):
    """Test admin authentication credentials verification."""
    # Valid login
    res_valid = client.post(
        "/api/v1/auth/login",
        json={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD},
    )
    assert res_valid.status_code == 200
    token_data = res_valid.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "Bearer"

    # Invalid login
    res_invalid = client.post(
        "/api/v1/auth/login",
        json={"username": "wrong_user", "password": "bad_password"},
    )
    assert res_invalid.status_code == 401


def test_student_and_consent_api_flow(client: TestClient, auth_headers: dict):
    """Test registering student, querying, granting consent, and deleting."""
    student_payload = {
        "student_id": "API_STU_001",
        "first_name": "Rosalind",
        "last_name": "Franklin",
        "email": "rfranklin@kings.edu",
        "department": "Biophysics",
    }

    # 1. Register student
    res_create = client.post("/api/v1/students", json=student_payload, headers=auth_headers)
    assert res_create.status_code in (201, 400)  # 201 created or 400 if already exists

    # 2. Get student
    res_get = client.get("/api/v1/students/API_STU_001")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["student_id"] == "API_STU_001"
    assert data["first_name"] == "Rosalind"

    # 3. Record consent
    consent_payload = {
        "student_id": "API_STU_001",
        "status": "GRANTED",
        "terms_version": "v1.0",
    }
    res_consent = client.post("/api/v1/consent", json=consent_payload)
    assert res_consent.status_code == 200
    assert res_consent.json()["status"] == "GRANTED"

    # 4. Delete student
    res_del = client.delete("/api/v1/students/API_STU_001", headers=auth_headers)
    assert res_del.status_code == 200


def test_session_and_manual_attendance_api(client: TestClient, auth_headers: dict):
    """Test creating session, manual attendance override, and CSV export."""
    # Register test student
    client.post(
        "/api/v1/students",
        json={
            "student_id": "API_MANUAL_STU",
            "first_name": "Dorothy",
            "last_name": "Hodgkin",
            "email": "dhodgkin@oxford.ac.uk",
            "department": "Chemistry",
        },
        headers=auth_headers,
    )
    client.post(
        "/api/v1/consent",
        json={"student_id": "API_MANUAL_STU", "status": "GRANTED"},
    )

    # Create session
    sess_res = client.post(
        "/api/v1/attendance/sessions",
        json={"course_code": "CHEM-301", "session_name": "Crystallography"},
        headers=auth_headers,
    )
    assert sess_res.status_code == 201
    sid = sess_res.json()["session_id"]

    # Manual mark
    man_res = client.post(
        "/api/v1/attendance/manual",
        json={"session_id": sid, "student_id": "API_MANUAL_STU", "reason": "Authorized manual entry"},
        headers=auth_headers,
    )
    assert man_res.status_code == 200

    # Export CSV
    export_res = client.get(f"/api/v1/attendance/export?session_id={sid}")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers.get("content-type", "")
    assert "API_MANUAL_STU" in export_res.text
