"""Unit tests for evaluation.py (Biometric Metrics & Ablation Harness)."""

from pathlib import Path
import numpy as np
import pytest

from evaluation import BiometricEvaluator


def test_roc_and_eer_computation():
    """Test ROC and Equal Error Rate calculation with separated distributions."""
    evaluator = BiometricEvaluator()
    rng = np.random.RandomState(42)

    gen = rng.normal(0.80, 0.05, 300)
    imp = rng.normal(0.20, 0.05, 1000)

    roc_res = evaluator.compute_roc_and_eer(gen, imp)

    assert "eer" in roc_res
    assert "eer_threshold" in roc_res
    assert "roc_auc" in roc_res

    # With well-separated distributions, EER should be very low and AUC near 1.0
    assert roc_res["eer"] < 0.01
    assert roc_res["roc_auc"] > 0.99
    assert 0.30 <= roc_res["eer_threshold"] <= 0.70


def test_privacy_overhead_benchmark():
    """Test cryptographic latency and storage expansion profiling."""
    evaluator = BiometricEvaluator()
    bench = evaluator.benchmark_privacy_overhead(num_trials=50)

    assert bench["num_trials"] == 50
    assert bench["mean_encryption_us"] > 0.0
    assert bench["mean_decryption_us"] > 0.0
    assert bench["raw_template_bytes"] == 2048
    assert bench["encrypted_template_bytes"] > 2048
    assert bench["max_reconstruction_error"] < 1e-5


def test_ablation_studies_execution(tmp_path: Path):
    """Test that all 6 systematic ablation studies execute and output structured metrics."""
    evaluator = BiometricEvaluator(figures_dir=tmp_path / "fig", metrics_dir=tmp_path / "met")
    ablations = evaluator.run_ablation_studies(synthetic_data=True)

    assert "ablation_1_openset" in ablations
    assert "ablation_2_liveness" in ablations
    assert "ablation_3_privacy" in ablations
    assert "ablation_4_multisample" in ablations
    assert "ablation_5_temporal" in ablations
    assert "ablation_6_zero_retention" in ablations

    assert (tmp_path / "met" / "ablation_results.json").is_file()
