# PrivateAttend: Module Interfaces & Integration Contracts

This document establishes the strict type contracts, public APIs, input/output formats, and component dependencies for all modules in PrivateAttend.

---

## 1. `config.py` — Central Configuration Management

### Responsibility
Provides immutable, environment-aware configuration via Pydantic settings. Handles file paths, recognition thresholds, detector parameters, liveness thresholds, security keys, and database connections.

### Key Classes & Attributes
```python
class AppConfig(BaseModel):
    # Paths
    BASE_DIR: Path
    DATA_DIR: Path
    RAW_DATA_DIR: Path
    PROCESSED_DATA_DIR: Path
    MANIFESTS_DIR: Path
    WEIGHTS_DIR: Path
    RESULTS_DIR: Path
    DATABASE_PATH: Path
    KEY_FILE_PATH: Path

    # Detection & Quality Settings
    DETECTOR_BACKEND: str = "yunet"  # Options: "yunet", "haar"
    DETECTOR_CONFIDENCE: float = 0.60
    FACE_TARGET_SIZE: Tuple[int, int] = (112, 112)
    MIN_FACE_RESOLUTION: int = 80
    BLUR_THRESHOLD: float = 50.0  # Laplacian variance threshold
    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 220.0

    # Recognition & Open-Set Settings
    EMBEDDING_DIM: int = 512
    RECOGNITION_BACKEND: str = "onnx"  # Options: "onnx", "torch"
    THETA_KNOWN: float = 0.55  # Acceptance threshold for known identities
    THETA_UNKNOWN: float = 0.35  # Rejection threshold for unknown universe
    TOP2_MARGIN: float = 0.08  # Minimum margin between top-1 and top-2
    DEFAULT_SAMPLES_PER_STUDENT: int = 5

    # Liveness (PAD) Settings
    LIVENESS_THRESHOLD: float = 0.50
    LIVENESS_WINDOW_SIZE: int = 5
    LIVENESS_MAX_VARIANCE: float = 0.05

    # Privacy & Security
    MASTER_KEY_ENV_VAR: str = "PRIVATEATTEND_MASTER_KEY"
    AUDIT_LOG_ENABLED: bool = True
    IMAGE_RETENTION_HOURS: int = 0  # 0 = zero retention (instant wipe)
```

---

## 2. `modules/preprocessing.py` — Image Ingestion & Quality Assessment

### Responsibility
Handles safe ingestion of image sources (numpy arrays, byte buffers, base64 strings, file paths), color conversion, validation, blur detection, illumination checks, and canonical crop normalization.

### Public Functions
```python
def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Loads an image into a 3-channel BGR numpy array (uint8).
    
    Args:
        source: File path, byte stream, or pre-existing numpy array.
    Returns:
        np.ndarray: BGR image matrix of shape (H, W, 3).
    Raises:
        InvalidImageError: If input cannot be decoded or is corrupt.
    """

def assess_image_quality(
    image: np.ndarray, 
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> ImageQualityResult:
    """Calculates sharpness, illumination, and dimensional quality metrics.
    
    Args:
        image: BGR image (H, W, 3).
        bbox: Optional (x, y, w, h) bounding box of the facial region.
    Returns:
        ImageQualityResult: Contains sharpness (float), mean_brightness (float),
                            contrast (float), is_acceptable (bool), reason (str).
    """

def normalize_facial_crop(
    crop: np.ndarray, 
    target_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """Resizes and normalizes an aligned facial crop into float32 [-1, 1] format.
    
    Args:
        crop: BGR facial crop (H, W, 3).
        target_size: Desired spatial dimension (W, H).
    Returns:
        np.ndarray: Normalized float32 tensor of shape (3, H, W).
    """
```

---

## 3. `modules/detection.py` — Face Detection & Alignment

### Responsibility
Performs lightweight face localization, 5-point facial landmark regression (eyes, nose, mouth corners), and affine similarity rectification to standard 112x112 coordinates.

### Public Classes & Functions
```python
@dataclass
class FaceDetectionResult:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: np.ndarray  # Shape (5, 2) in image coordinates
    aligned_face: np.ndarray  # (112, 112, 3) BGR uint8 aligned image
    quality: ImageQualityResult

class FaceDetector:
    def __init__(self, backend: str = "yunet", min_confidence: float = 0.6):
        """Initializes detector with YuNet or Haar fallback."""

    def detect(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """Detects all faces in the image, performs alignment and quality checks.
        
        Args:
            image: BGR image (H, W, 3).
        Returns:
            List[FaceDetectionResult]: Detected faces with aligned crops.
        """

    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Applies 5-point similarity transformation matrix to align face.
        
        Args:
            image: Original BGR image.
            landmarks: (5, 2) array of coordinates.
        Returns:
            np.ndarray: 112x112 aligned facial crop.
        """
```

---

## 4. `models.py` & `modules/recognition.py` — Face Recognition & Metric Space

### Responsibility
Extracts 512-dimensional hyperspherical embeddings using an ArcFace-compatible architecture, performs L2 normalization, computes cosine similarities, and aggregates multi-sample enrollment templates.

### Public Classes & Functions
```python
class FaceRecognizer:
    def __init__(self, weights_path: Optional[Path] = None, backend: str = "onnx"):
        """Loads ArcFace model in ONNX Runtime or PyTorch."""

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extracts an L2-normalized 512-d feature embedding vector.
        
        Args:
            aligned_face: (112, 112, 3) BGR uint8 aligned crop.
        Returns:
            np.ndarray: 1-D float32 vector of shape (512,) where ||e||_2 == 1.0.
        Raises:
            ModelNotLoadedError: If pretrained model weights are missing.
        """

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Computes cosine similarity between two unit vectors: dot(e1, e2)."""

    def aggregate_templates(
        self, 
        embeddings: List[np.ndarray], 
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Calculates a quality-weighted average unit template: e_agg / ||e_agg||_2."""
```

---

## 5. `face_engine.py` — Open-Set Decision Engine

### Responsibility
Orchestrates identification against enrolled gallery templates and enforces open-set partitioning: `KNOWN`, `UNKNOWN`, and `UNCERTAIN`.

### Decision Classes & Logic
```python
class DecisionState(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"

@dataclass
class RecognitionMatchResult:
    decision: DecisionState
    student_id: Optional[str]
    matched_score: float
    runner_up_id: Optional[str]
    runner_up_score: float
    margin: float
    explanation: str

class FaceEngine:
    def match_probe(
        self, 
        probe_embedding: np.ndarray, 
        gallery_templates: Dict[str, np.ndarray]
    ) -> RecognitionMatchResult:
        """Compares probe embedding against all active decrypted gallery templates.
        
        Returns:
            RecognitionMatchResult with calibrated open-set decision.
        """
```

---

## 6. `liveness.py` — Presentation Attack Detection (PAD)

### Responsibility
Executes deep texture and frequency analysis to detect presentation attacks (printed photos, screen replays, video spoofs). Provides frame-level scoring and multi-frame temporal window smoothing.

### Public Classes & Functions
```python
@dataclass
class LivenessResult:
    is_live: bool
    liveness_score: float  # [0.0, 1.0] where 1.0 is bona fide
    attack_detected: bool
    variance: float
    confidence_passed: bool

class LivenessDetector:
    def __init__(self, weights_path: Optional[Path] = None):
        """Initializes LivenessNet model."""

    def predict_frame(self, aligned_face: np.ndarray) -> float:
        """Evaluates a single aligned face crop. Returns bona fide probability."""

    def update_temporal_window(self, frame_score: float) -> LivenessResult:
        """Appends frame score to rolling window of size W and computes aggregated result."""

    def reset_temporal_window(self) -> None:
        """Clears rolling temporal buffer."""
```

---

## 7. `privacy.py` — Cryptographic Protection & Consent Lifecycle

### Responsibility
Provides authenticated encryption at rest for biometric templates using AES-256-GCM / Fernet, manages cryptographic key generation and rotation, enforces consent state machine, and triggers immediate template purging.

### Public Classes & Functions
```python
class KeyManager:
    def __init__(self, key_path: Optional[Path] = None):
        """Loads master encryption key from ENV or secure key file with 0600 permissions."""

    def generate_key(self) -> bytes:
        """Generates a high-entropy 256-bit Fernet/AES key."""

class TemplateProtector:
    def __init__(self, key_manager: KeyManager):
        """Initializes cryptographic cipher."""

    def encrypt_template(self, embedding: np.ndarray) -> bytes:
        """Serializes float32 numpy vector and encrypts with authenticated cipher."""

    def decrypt_template(self, ciphertext: bytes) -> np.ndarray:
        """Decrypts ciphertext and reconstructs 512-d float32 numpy vector."""

class ConsentManager:
    def verify_consent(self, student_id: str, db: Database) -> bool:
        """Checks if student has active GRANTED consent."""

    def revoke_consent(self, student_id: str, db: Database) -> bool:
        """Revokes consent, deletes encrypted template immediately, and logs audit event."""
```

---

## 8. `database.py` — Relational SQLite Persistence

### Responsibility
Thread-safe SQLite database operations with foreign key integrity, ACID transactions, and structured audit logging.

### Public Methods
```python
class Database:
    def __init__(self, db_path: Path):
        """Initializes SQLite connection with PRAGMA foreign_keys = ON and WAL mode."""

    def init_schema(self) -> None:
        """Creates tables: students, consent_records, biometric_templates, 
           attendance_sessions, attendance_records, admin_users, audit_events."""

    # Student CRUD
    def add_student(self, student_data: StudentCreate) -> int: ...
    def get_student(self, student_id: str) -> Optional[Dict]: ...
    def list_students(self, active_only: bool = True) -> List[Dict]: ...
    def delete_student(self, student_id: str) -> bool: ...

    # Biometrics & Consent
    def store_template(self, student_id: str, encrypted_bytes: bytes, count: int, quality: float): ...
    def get_active_templates(self) -> Dict[str, bytes]: ...
    def delete_template(self, student_id: str) -> bool: ...
    def record_consent(self, student_id: str, status: str, version: str) -> int: ...

    # Attendance
    def create_session(self, course_code: str, session_name: str, created_by: str) -> int: ...
    def get_active_session(self) -> Optional[Dict]: ...
    def record_attendance(self, session_id: int, student_id: str, status: str, 
                          confidence: float, liveness: float, notes: str) -> bool: ...
    def is_attendance_marked(self, session_id: int, student_id: str) -> bool: ...
    def get_session_attendance(self, session_id: int) -> List[Dict]: ...

    # Audit
    def log_audit(self, event_type: str, actor: str, entity_id: str, details: str): ...
    def get_audit_logs(self, limit: int = 100) -> List[Dict]: ...
```

---

## 9. `attendance.py` — Business Logic & Attendance Engine

### Responsibility
Integrates detection, quality filtering, PAD verification, ArcFace matching, open-set rejection, consent validation, duplicate prevention, and transaction persistence.

### Public Interface
```python
@dataclass
class AttendanceVerificationResult:
    success: bool
    status_code: str  # e.g., "MARKED", "ALREADY_MARKED", "SPOOF_REJECTED", "UNKNOWN_REJECTED"
    student_id: Optional[str]
    student_name: Optional[str]
    confidence_score: float
    liveness_score: float
    message: str
    audit_logged: bool

class AttendanceEngine:
    def process_attendance_frame(
        self, 
        image: np.ndarray, 
        session_id: int
    ) -> AttendanceVerificationResult:
        """Executes full verification pipeline for an incoming image/frame."""

    def manual_mark_attendance(
        self, 
        session_id: int, 
        student_id: str, 
        reason: str, 
        actor: str
    ) -> bool:
        """Allows administrative override for students with revoked consent or uncertain score."""

    def export_attendance_csv(self, session_id: int) -> str:
        """Generates a standardized CSV report string."""
```

---

## 10. `api.py` — FastAPI REST Interface

### Responsibility
Exposes JSON and multipart endpoints for all operations, implements Bearer JWT authentication, and ensures zero raw embedding exposure.

### Primary Endpoints
- `POST /api/v1/auth/login` $\to$ Returns JWT access token.
- `GET /api/v1/health` $\to$ Service health and component diagnostics.
- `GET /api/v1/system/status` $\to$ Weights availability, database stats, detector mode.
- `POST /api/v1/students` $\to$ Create new student record.
- `GET /api/v1/students` $\to$ List enrolled students.
- `POST /api/v1/consent` $\to$ Grant or revoke biometric consent.
- `POST /api/v1/biometrics/enroll` $\to$ Multipart image upload for multi-sample template creation.
- `POST /api/v1/attendance/sessions` $\to$ Initialize course attendance session.
- `POST /api/v1/attendance/verify` $\to$ Multipart image verification and attendance marking.
- `POST /api/v1/attendance/manual` $\to$ Administrative manual override.
- `GET /api/v1/attendance/records` $\to$ Query attendance records with session/student filtering.
- `GET /api/v1/attendance/export` $\to$ CSV stream of session attendance.
- `GET /api/v1/audit/logs` $\to$ Security and operational audit log stream.

---

## 11. `evaluation.py` — Benchmark & Ablation Metrics

### Responsibility
Calculates biometric error rates (FAR, FRR, ROC, EER), open-set metrics (KAR, URR, DIR@FAR), PAD metrics (APCER, BPCER, ACER), privacy overheads, and system latency distributions.
