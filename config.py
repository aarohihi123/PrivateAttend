"""PrivateAttend: Centralized Configuration Management.

Provides type-safe, environment-aware configuration across all modules,
enforcing deterministic paths, thresholds, and security policies.
"""

from pathlib import Path
import os
from typing import Tuple
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """System-wide configuration parameters."""

    # Project Base Directories
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    RAW_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "processed")
    MANIFESTS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "manifests")
    WEIGHTS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "weights")
    RESULTS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "results")
    REPORTS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "results" / "reports")
    METRICS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "results" / "metrics")
    FIGURES_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "results" / "figures")

    # Persistence & Security Paths
    DATABASE_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "privateattend.db")
    KEY_FILE_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / ".master.key")
    EMBEDDINGS_NPZ_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "embeddings_buffalo_sc.npz")
    RENAME_MAP_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "rename_map.json")

    # Face Detection & Landmark Alignment Settings
    DETECTOR_BACKEND: str = Field(
        default_factory=lambda: os.getenv("PRIVATEATTEND_DETECTOR_BACKEND", "yunet")
    )  # "yunet" or "haar"
    DETECTOR_CONFIDENCE: float = 0.60
    DETECTOR_NMS_THRESHOLD: float = 0.30
    FACE_TARGET_SIZE: Tuple[int, int] = (112, 112)
    MIN_FACE_RESOLUTION: int = 80  # Minimum face width/height in pixels

    # Image Quality Assessment Thresholds
    BLUR_THRESHOLD: float = 25.0  # Minimum acceptable Laplacian variance
    MIN_BRIGHTNESS: float = 40.0  # Minimum mean grayscale value (0-255)
    MAX_BRIGHTNESS: float = 220.0  # Maximum mean grayscale value (0-255)
    MIN_CONTRAST: float = 25.0  # Minimum standard deviation of grayscale intensities

    # Face Recognition & Metric Space
    EMBEDDING_DIM: int = 512
    RECOGNITION_BACKEND: str = Field(
        default_factory=lambda: os.getenv("PRIVATEATTEND_RECOGNITION_BACKEND", "onnx")
    )  # "onnx" or "torch"
    ARCFACE_WEIGHTS_FILE: str = "w600k_mbf.onnx"
    YUNET_WEIGHTS_FILE: str = "face_detection_yunet_2023mar.onnx"

    # Open-Set Recognition Decision Boundaries (Empirically calibrated on UMDFace buffalo_sc: 0.295 operating point)
    THETA_KNOWN: float = 0.35  # Similarity >= THETA_KNOWN -> High-confidence KNOWN identity
    THETA_UNKNOWN: float = 0.295  # Similarity < THETA_UNKNOWN -> UNKNOWN identity (<=0.8% stranger FA)
    TOP2_MARGIN: float = 0.05  # Required delta between top-1 and top-2 match
    ENROLLMENT_SAMPLE_COUNT: int = 3  # Default K=3 enrollment images per person (supports K=1,3,5)

    # Liveness Detection (PAD) Settings
    LIVENESS_THRESHOLD: float = 0.50  # Probability >= 0.50 -> Bona fide live face
    LIVENESS_WINDOW_SIZE: int = 5  # Number of frames for temporal smoothing
    LIVENESS_MAX_VARIANCE: float = 0.06  # Maximum allowable temporal variance
    LIVENESS_WEIGHTS_FILE: str = "liveness_cnn_v1.pth"

    # Privacy & Cryptography
    MASTER_KEY_ENV_VAR: str = "PRIVATEATTEND_MASTER_KEY"
    CIPHER_ALGORITHM: str = "AES-256-GCM / Fernet + Cancelable Orthogonal Projection"
    IMAGE_RETENTION_HOURS: int = 0  # 0 indicates zero retention (wipe immediately)
    AUDIT_LOG_ENABLED: bool = True

    # API & Service Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    ADMIN_USERNAME: str = Field(default_factory=lambda: os.getenv("PRIVATEATTEND_ADMIN_USER", "admin"))
    ADMIN_PASSWORD: str = Field(default_factory=lambda: os.getenv("PRIVATEATTEND_ADMIN_PASS", "PrivateAttend@2026"))
    JWT_SECRET: str = Field(default_factory=lambda: os.getenv("PRIVATEATTEND_JWT_SECRET", "super-secret-dev-jwt-key"))
    JWT_EXPIRY_MINUTES: int = 480

    def ensure_directories(self) -> None:
        """Creates all required project directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MANIFESTS_DIR,
            self.WEIGHTS_DIR,
            self.RESULTS_DIR,
            self.REPORTS_DIR,
            self.METRICS_DIR,
            self.FIGURES_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global singleton configuration instance
config = AppConfig()
config.ensure_directories()
