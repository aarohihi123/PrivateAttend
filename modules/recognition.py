"""PrivateAttend: Face Recognition Feature Extraction and Metric Space.

Extracts 512-dimensional L2-normalized feature vectors using ArcFace-compatible
models, computes cosine similarity, aggregates multi-sample templates, and validates embeddings.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

from config import config
from models import ModelNotLoadedError, ONNXRecognitionInference
from modules.preprocessing import normalize_facial_crop


class FaceRecognizer:
    """Facial feature extractor projecting aligned crops onto the 512-d unit hypersphere."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        backend: Optional[str] = None,
        allow_mock: bool = False,
    ):
        self.weights_path = weights_path or (config.WEIGHTS_DIR / config.ARCFACE_WEIGHTS_FILE)
        self.backend = (backend or config.RECOGNITION_BACKEND).lower()
        self.allow_mock = allow_mock

        self._onnx_inference: Optional[ONNXRecognitionInference] = None
        self._torch_model = None
        self.weights_loaded = False

        self._init_model()

    def _init_model(self) -> None:
        """Loads inference engine if model weights are present."""
        if self.weights_path.is_file():
            try:
                if self.backend == "onnx":
                    self._onnx_inference = ONNXRecognitionInference(self.weights_path)
                    self.weights_loaded = True
                    return
                elif self.backend == "torch":
                    import torch
                    from models import MobileFaceNet

                    net = MobileFaceNet(embedding_dim=config.EMBEDDING_DIM)
                    torch_mod = net.build_torch_module()
                    state_dict = torch.load(str(self.weights_path), map_location="cpu")
                    torch_mod.load_state_dict(state_dict)
                    torch_mod.eval()
                    self._torch_model = torch_mod
                    self.weights_loaded = True
                    return
            except Exception:
                self.weights_loaded = False

        self.weights_loaded = False

    def extract_embedding(
        self,
        aligned_face: np.ndarray,
        allow_synthetic: bool = False,
    ) -> np.ndarray:
        """Generates an L2-normalized 512-dimensional embedding vector.

        Args:
            aligned_face: (112, 112, 3) BGR uint8 aligned facial image.
            allow_synthetic: If True and weights are absent, allows a deterministic synthetic
                             embedding strictly for unit testing / interface verification.

        Returns:
            np.ndarray: float32 1-D vector of shape (512,) where np.linalg.norm(e) == 1.0.

        Raises:
            ModelNotLoadedError: If pretrained model weights are absent and allow_synthetic=False.
        """
        if self._onnx_inference is not None:
            tensor = normalize_facial_crop(aligned_face, target_size=config.FACE_TARGET_SIZE)
            embedding = self._onnx_inference.run(tensor)
            return self.normalize(embedding)

        if self._torch_model is not None:
            import torch

            tensor = normalize_facial_crop(aligned_face, target_size=config.FACE_TARGET_SIZE)
            torch_tensor = torch.from_numpy(tensor).unsqueeze(0)
            with torch.no_grad():
                out = self._torch_model(torch_tensor)
                embedding = out.squeeze(0).cpu().numpy()
            return self.normalize(embedding)

        if allow_synthetic or self.allow_mock:
            # Deterministic pseudo-embedding based on pixel hashes for testing interfaces
            seed = int(np.sum(aligned_face[:20, :20]) * 1000) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            vec = rng.randn(config.EMBEDDING_DIM).astype(np.float32)
            return self.normalize(vec)

        raise ModelNotLoadedError(
            f"ArcFace model weights not found at {self.weights_path}. "
            "Operational inference requires pretrained weights in weights/. "
            "For automated testing, enable allow_synthetic=True."
        )

    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        """Projects a vector onto the unit hypersphere: e / ||e||_2."""
        norm = float(np.linalg.norm(embedding))
        if norm == 0.0 or np.isnan(norm):
            raise ValueError("Zero or NaN norm encountered during embedding normalization.")
        return (embedding / norm).astype(np.float32)

    @staticmethod
    def compute_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        """Calculates cosine similarity between two unit vectors: dot(e1, e2).

        Args:
            e1: First 512-d normalized embedding.
            e2: Second 512-d normalized embedding.

        Returns:
            float: Cosine similarity in range [-1.0, 1.0].
        """
        dot = float(np.dot(e1.flatten(), e2.flatten()))
        # Clamp against floating point inaccuracies
        return float(np.clip(dot, -1.0, 1.0))

    def aggregate_templates(
        self,
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None,
        outlier_threshold: float = 0.30,
    ) -> np.ndarray:
        """Aggregates multiple enrollment sample embeddings into a robust identity template.

        Args:
            embeddings: List of 512-d unit-normalized vectors.
            weights: Optional quality weights for each sample.
            outlier_threshold: Minimum pairwise similarity to prevent corrupt sample poisoning.

        Returns:
            np.ndarray: Unified unit-normalized 512-d identity template vector.

        Raises:
            ValueError: If embeddings list is empty or samples are inconsistent.
        """
        if not embeddings:
            raise ValueError("Cannot generate template from empty embedding list.")

        if len(embeddings) == 1:
            return self.normalize(embeddings[0])

        stacked = np.stack([self.normalize(e) for e in embeddings], axis=0)  # (N, 512)

        # Outlier filtering: compute centroid and remove samples with low similarity
        centroid = np.mean(stacked, axis=0)
        centroid = self.normalize(centroid)
        similarities = np.dot(stacked, centroid)

        valid_mask = similarities >= outlier_threshold
        if not np.any(valid_mask):
            valid_mask = np.ones(len(embeddings), dtype=bool)

        filtered_stacked = stacked[valid_mask]

        if weights is not None:
            w_arr = np.array(weights, dtype=np.float32)[valid_mask]
            w_sum = np.sum(w_arr)
            if w_sum > 0:
                w_arr = w_arr / w_sum
                agg = np.sum(filtered_stacked * w_arr[:, np.newaxis], axis=0)
            else:
                agg = np.mean(filtered_stacked, axis=0)
        else:
            agg = np.mean(filtered_stacked, axis=0)

        return self.normalize(agg)
