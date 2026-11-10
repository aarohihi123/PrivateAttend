"""Unit tests for modules/recognition.py."""

from pathlib import Path
import numpy as np
import pytest

from models import ModelNotLoadedError
from modules.recognition import FaceRecognizer


def test_embedding_normalization(synthetic_embedding):
    """Test L2 unit normalization enforcing norm == 1.0."""
    unnormalized = synthetic_embedding * 15.75
    normed = FaceRecognizer.normalize(unnormalized)

    assert normed.shape == (512,)
    assert normed.dtype == np.float32
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-5)


def test_cosine_similarity(synthetic_embedding):
    """Test cosine similarity mathematical properties."""
    e1 = synthetic_embedding
    e2 = synthetic_embedding.copy()

    # Identical vectors -> similarity == 1.0
    sim_identical = FaceRecognizer.compute_similarity(e1, e2)
    assert np.isclose(sim_identical, 1.0, atol=1e-5)

    # Opposite vector -> similarity == -1.0
    sim_opposite = FaceRecognizer.compute_similarity(e1, -e2)
    assert np.isclose(sim_opposite, -1.0, atol=1e-5)

    # Orthogonal vector -> similarity == 0.0
    rng = np.random.RandomState(99)
    random_vec = rng.randn(512).astype(np.float32)
    orthogonal = random_vec - np.dot(random_vec, e1) * e1
    orthogonal = FaceRecognizer.normalize(orthogonal)
    sim_ortho = FaceRecognizer.compute_similarity(e1, orthogonal)
    assert np.isclose(sim_ortho, 0.0, atol=1e-4)


def test_aggregate_templates():
    """Test multi-sample quality-weighted template aggregation."""
    recognizer = FaceRecognizer(allow_mock=True)
    rng = np.random.RandomState(123)

    base = rng.randn(512).astype(np.float32)
    base /= np.linalg.norm(base)

    # Create 3 noisy samples around base with small noise std 0.01
    samples = [FaceRecognizer.normalize(base + rng.normal(0, 0.01, 512).astype(np.float32)) for _ in range(3)]
    weights = [50.0, 80.0, 100.0]

    template = recognizer.aggregate_templates(samples, weights=weights)
    assert template.shape == (512,)
    assert np.isclose(np.linalg.norm(template), 1.0, atol=1e-5)

    # Aggregated template should have very high similarity with base
    sim = recognizer.compute_similarity(template, base)
    assert sim > 0.95


def test_missing_weights_rejection(synthetic_face_image):
    """Test clean ModelNotLoadedError when operational weights are missing."""
    recognizer = FaceRecognizer(weights_path=Path("non_existent_weights.onnx"), allow_mock=False)
    with pytest.raises(ModelNotLoadedError):
        recognizer.extract_embedding(synthetic_face_image, allow_synthetic=False)
