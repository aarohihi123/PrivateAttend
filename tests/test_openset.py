"""Unit tests for face_engine.py (Open-Set Recognition & Threshold Calibration)."""

import numpy as np
import pytest

from face_engine import DecisionState, FaceEngine, RecognitionMatchResult
from modules.recognition import FaceRecognizer


def test_openset_known_decision(synthetic_embedding):
    """Test KNOWN decision state when score >= theta_known and margin is sufficient."""
    engine = FaceEngine(theta_known=0.55, theta_unknown=0.35, top2_margin=0.08)

    gallery = {
        "STU1001": synthetic_embedding,
        "STU1002": FaceRecognizer.normalize(np.random.RandomState(1).randn(512).astype(np.float32)),
    }

    # Probe identical to STU1001
    res: RecognitionMatchResult = engine.match_probe(synthetic_embedding, gallery)
    assert res.decision == DecisionState.KNOWN
    assert res.student_id == "STU1001"
    assert res.matched_score >= 0.99
    assert res.margin >= 0.08


def test_openset_unknown_decision(synthetic_embedding):
    """Test UNKNOWN decision state when score < theta_unknown."""
    engine = FaceEngine(theta_known=0.55, theta_unknown=0.35, top2_margin=0.08)

    # Gallery with orthogonal vectors
    rng = np.random.RandomState(42)
    rand1 = FaceRecognizer.normalize(rng.randn(512).astype(np.float32))
    gallery = {"STU2001": rand1}

    # Construct orthogonal probe
    probe = rng.randn(512).astype(np.float32)
    probe = probe - np.dot(probe, rand1) * rand1
    probe = FaceRecognizer.normalize(probe)

    res = engine.match_probe(probe, gallery)
    assert res.decision == DecisionState.UNKNOWN
    assert res.student_id is None
    assert res.matched_score < 0.35


def test_openset_uncertain_ambiguous_margin():
    """Test UNCERTAIN decision state when top-1 and top-2 candidates are too close."""
    engine = FaceEngine(theta_known=0.55, theta_unknown=0.35, top2_margin=0.08)

    rng = np.random.RandomState(77)
    probe = FaceRecognizer.normalize(rng.randn(512).astype(np.float32))

    # Gallery where two students have nearly identical match to probe (margin < 0.08)
    stu1 = probe.copy()
    stu2 = FaceRecognizer.normalize(probe + rng.normal(0, 0.005, 512).astype(np.float32))
    gallery = {"STU_A": stu1, "STU_B": stu2}

    res = engine.match_probe(probe, gallery)
    assert res.decision == DecisionState.UNCERTAIN
    assert res.margin < 0.08


def test_calibrate_thresholds():
    """Test empirical validation-based threshold calibration algorithm."""
    rng = np.random.RandomState(999)
    # 200 genuine scores centered at 0.72
    known_val = rng.normal(0.72, 0.08, 200).tolist()
    # 500 unknown impostor scores centered at 0.25
    unknown_val = rng.normal(0.25, 0.08, 500).tolist()

    th_known, th_unknown = FaceEngine.calibrate_thresholds(
        known_val, unknown_val, target_far=0.01
    )

    assert th_known > th_unknown
    assert 0.30 <= th_unknown <= 0.60
    assert 0.40 <= th_known <= 0.85
