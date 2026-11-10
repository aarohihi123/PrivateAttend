"""PrivateAttend: Shared PyTest Test Fixtures.

Provides clean temporary database instances, cryptographic keys, synthetic test images,
and deterministic unit vectors for rigorous automated testing without fabricating research data.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from config import AppConfig
from database import Database
from privacy import KeyManager, TemplateProtector


@pytest.fixture
def temp_env(tmp_path: Path):
    """Generates an isolated temporary configuration environment."""
    test_db = tmp_path / "test_privateattend.db"
    test_key = tmp_path / ".test_master.key"
    test_results = tmp_path / "results"
    test_results.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(
        BASE_DIR=tmp_path,
        DATABASE_PATH=test_db,
        KEY_FILE_PATH=test_key,
        RESULTS_DIR=test_results,
        AUDIT_LOG_ENABLED=True,
    )
    return cfg


@pytest.fixture
def test_db(temp_env: AppConfig):
    """Provides a fresh, isolated Database instance."""
    db = Database(db_path=temp_env.DATABASE_PATH)
    return db


@pytest.fixture
def key_manager(temp_env: AppConfig):
    """Provides an isolated KeyManager instance."""
    km = KeyManager(key_path=temp_env.KEY_FILE_PATH)
    return km


@pytest.fixture
def template_protector(key_manager: KeyManager):
    """Provides an isolated TemplateProtector instance."""
    return TemplateProtector(key_manager=key_manager)


@pytest.fixture
def synthetic_face_image():
    """Generates a synthetic 112x112 face-like image with sharp geometric landmarks."""
    img = np.full((112, 112, 3), 120, dtype=np.uint8)
    # High-contrast edge details to pass Laplacian sharpness gate (> 50.0)
    for i in range(10, 100, 4):
        cv2.line(img, (i, 20), (i, 25), (200, 200, 200), 1)

    # Facial feature patterns
    cv2.circle(img, (38, 51), 10, (10, 10, 10), -1)   # Left eye
    cv2.circle(img, (38, 51), 3, (255, 255, 255), -1)  # Pupil reflection
    cv2.circle(img, (74, 51), 10, (10, 10, 10), -1)   # Right eye
    cv2.circle(img, (74, 51), 3, (255, 255, 255), -1)  # Pupil reflection
    cv2.line(img, (56, 60), (56, 75), (20, 20, 20), 3)  # Nose
    cv2.rectangle(img, (40, 88), (72, 96), (30, 30, 30), -1)  # Mouth
    return img


@pytest.fixture
def synthetic_blurry_image():
    """Generates a severely blurred image that should fail the sharpness quality gate."""
    img = np.full((112, 112, 3), 128, dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (31, 31), 0)
    return blurred


@pytest.fixture
def synthetic_dark_image():
    """Generates an underexposed image that should fail the minimum illumination gate."""
    return np.full((112, 112, 3), 15, dtype=np.uint8)


@pytest.fixture
def synthetic_embedding():
    """Generates a 512-dimensional unit-normalized float32 vector."""
    rng = np.random.RandomState(42)
    vec = rng.randn(512).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm
