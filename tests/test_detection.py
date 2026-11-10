"""Unit tests for modules/detection.py."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from modules.detection import FaceDetectionResult, FaceDetector, align_face_5pts


def test_align_face_5pts(synthetic_face_image):
    """Test 5-point affine transformation to 112x112 reference landmarks."""
    # Source landmarks corresponding to the drawn circles in synthetic_face_image
    landmarks = np.array([
        [38.0, 51.0],  # Left eye
        [74.0, 51.0],  # Right eye
        [56.0, 72.0],  # Nose
        [41.0, 92.0],  # Left mouth
        [71.0, 92.0],  # Right mouth
    ], dtype=np.float32)

    aligned = align_face_5pts(synthetic_face_image, landmarks, target_size=(112, 112))
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8


def test_face_detector_fallback_init():
    """Test detector initialization with Haar cascade fallback."""
    detector = FaceDetector(backend="haar")
    assert detector.backend == "haar"
    assert detector._haar is not None


def test_face_detector_prealigned_bypass(synthetic_face_image):
    """Test detector recognizing an already 112x112 aligned image."""
    detector = FaceDetector(backend="haar")
    results = detector.detect(synthetic_face_image)
    assert len(results) == 1
    det = results[0]
    assert det.bbox == (0, 0, 112, 112)
    assert det.aligned_face.shape == (112, 112, 3)
    assert det.confidence == 1.0
