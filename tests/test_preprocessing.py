"""Unit tests for modules/preprocessing.py."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from modules.preprocessing import (
    ImageQualityResult,
    InvalidImageError,
    LowQualityImageError,
    assess_image_quality,
    load_image,
    normalize_facial_crop,
)


def test_load_image_numpy(synthetic_face_image):
    """Test loading directly from pre-existing numpy array."""
    loaded = load_image(synthetic_face_image)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (112, 112, 3)
    assert loaded.dtype == np.uint8


def test_load_image_bytes(synthetic_face_image):
    """Test decoding image from raw byte stream."""
    success, encoded = cv2.imencode(".jpg", synthetic_face_image)
    assert success
    raw_bytes = encoded.tobytes()

    loaded = load_image(raw_bytes)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape[:2] == (112, 112)


def test_load_image_invalid():
    """Test rejection of empty or corrupt inputs."""
    with pytest.raises(InvalidImageError):
        load_image(None)

    with pytest.raises(InvalidImageError):
        load_image(b"this is not a valid image stream")

    with pytest.raises(InvalidImageError):
        load_image(np.array([]))


def test_assess_image_quality_sharp(synthetic_face_image):
    """Test quality metrics on sharp synthetic image."""
    quality: ImageQualityResult = assess_image_quality(synthetic_face_image)
    assert quality.is_acceptable
    assert quality.sharpness > 50.0
    assert quality.brightness >= 40.0
    assert len(quality.failure_reasons) == 0


def test_assess_image_quality_blurry(synthetic_blurry_image):
    """Test rejection of blurred image."""
    quality = assess_image_quality(synthetic_blurry_image)
    assert not quality.is_acceptable
    assert any("blurry" in r.lower() for r in quality.failure_reasons)


def test_assess_image_quality_dark(synthetic_dark_image):
    """Test rejection of underexposed image."""
    quality = assess_image_quality(synthetic_dark_image)
    assert not quality.is_acceptable
    assert any("dark" in r.lower() or "underexposed" in r.lower() for r in quality.failure_reasons)


def test_normalize_facial_crop(synthetic_face_image):
    """Test canonical 112x112 normalization and float32 [-1, 1] range."""
    tensor = normalize_facial_crop(synthetic_face_image, target_size=(112, 112))
    assert tensor.shape == (3, 112, 112)
    assert tensor.dtype == np.float32
    assert -1.05 <= np.min(tensor) <= np.max(tensor) <= 1.05
