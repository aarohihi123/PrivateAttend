"""Unit tests for liveness.py (PAD & Temporal Aggregation)."""

import numpy as np
import pytest

from liveness import LivenessDetector, LivenessResult, compute_pad_metrics


def test_compute_pad_metrics_synthetic():
    """Test ISO/IEC 30107-3 APCER, BPCER, ACER calculation on deterministic labels."""
    # 10 bona fide (label 1), 10 attacks (label 0)
    y_true = np.array([1] * 10 + [0] * 10)
    # 9 bona fide pass (>=0.5), 1 fails (score 0.3) -> 1 false rejection (BPCER = 1/10 = 10%)
    # 8 attacks fail (<0.5), 2 pass (score 0.7) -> 2 false acceptances (APCER = 2/10 = 20%)
    y_scores = np.array(
        [0.9, 0.85, 0.88, 0.92, 0.75, 0.80, 0.65, 0.70, 0.60, 0.30] +
        [0.1, 0.20, 0.15, 0.05, 0.25, 0.35, 0.40, 0.45, 0.70, 0.75]
    )

    metrics = compute_pad_metrics(y_true, y_scores, threshold=0.50)
    assert np.isclose(metrics["apcer"], 0.20, atol=1e-4)
    assert np.isclose(metrics["bpcer"], 0.10, atol=1e-4)
    assert np.isclose(metrics["acer"], 0.15, atol=1e-4)  # (0.20 + 0.10) / 2
    assert metrics["roc_auc"] > 0.80


def test_liveness_temporal_aggregation():
    """Test sliding-window temporal smoothing and variance thresholding."""
    detector = LivenessDetector(window_size=5, threshold=0.50, max_variance=0.06)

    # Sequence of consistent high scores
    for score in [0.82, 0.85, 0.83, 0.84]:
        detector.update_temporal_window(score)

    final_res: LivenessResult = detector.update_temporal_window(0.86)
    assert final_res.is_live
    assert not final_res.attack_detected
    assert final_res.variance < 0.06
    assert final_res.liveness_score >= 0.80


def test_liveness_temporal_variance_rejection():
    """Test rejection when frame scores are erratic / unstable across consecutive frames."""
    detector = LivenessDetector(window_size=4, threshold=0.50, max_variance=0.04)

    # Erratic sequence alternating between 0.10 and 0.90
    detector.update_temporal_window(0.10)
    detector.update_temporal_window(0.90)
    detector.update_temporal_window(0.10)
    res = detector.update_temporal_window(0.90)

    # Variance for [0.1, 0.9, 0.1, 0.9] is 0.16 > 0.04 -> fails stability
    assert not res.confidence_passed
    assert not res.is_live
