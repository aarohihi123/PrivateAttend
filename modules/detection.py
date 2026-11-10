"""PrivateAttend: Face Detection and Landmark Alignment Module.

Implements lightweight on-device face localization using OpenCV YuNet
with seamless fallback to OpenCV Haar Cascades, 5-point facial landmark regression,
and affine similarity alignment to canonical 112x112 ArcFace templates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from config import config
from modules.preprocessing import ImageQualityResult, assess_image_quality, load_image


# Standard ArcFace 112x112 reference landmarks (5 points: [left eye, right eye, nose, left mouth, right mouth])
REFERENCE_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


@dataclass
class FaceDetectionResult:
    """Contains localized face coordinates, landmarks, crop, and quality assessment."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: np.ndarray  # Shape (5, 2) in original image coordinates
    aligned_face: np.ndarray  # (112, 112, 3) BGR uint8 aligned facial crop
    quality: ImageQualityResult


def align_face_5pts(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Warps an image via partial affine similarity transform to reference coordinates.

    Args:
        image: Original BGR image (H, W, 3).
        landmarks: (5, 2) array of coordinates: [left eye, right eye, nose, left mouth, right mouth].
        target_size: Target dimension (width, height), standard is (112, 112).

    Returns:
        np.ndarray: (112, 112, 3) aligned BGR facial image.
    """
    pts_src = np.array(landmarks, dtype=np.float32).reshape((5, 2))
    pts_dst = REFERENCE_5PTS_112.copy()

    # If target size is not 112x112, scale reference points accordingly
    if target_size != (112, 112):
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        pts_dst[:, 0] *= scale_x
        pts_dst[:, 1] *= scale_y

    # Compute partial affine similarity matrix (rotation, uniform scale, translation)
    transform_matrix, inliers = cv2.estimateAffinePartial2D(pts_src, pts_dst, method=cv2.LMEDS)

    if transform_matrix is None:
        # Fallback if points are collinear or degenerate: simple crop and resize
        x_min = int(max(0, np.min(pts_src[:, 0])))
        y_min = int(max(0, np.min(pts_src[:, 1])))
        x_max = int(min(image.shape[1], np.max(pts_src[:, 0])))
        y_max = int(min(image.shape[0], np.max(pts_src[:, 1])))
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            crop = image
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    aligned = cv2.warpAffine(
        image,
        transform_matrix,
        target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned


def find_haar_cascade_xml() -> Optional[str]:
    """Finds the frontal face Haar cascade XML across system and local locations."""
    candidates = [
        config.WEIGHTS_DIR / "haarcascade_frontalface_default.xml",
        Path("/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


class FaceDetector:
    """Multi-backend face detector supporting OpenCV YuNet and Haar Cascades."""

    def __init__(
        self,
        backend: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        weights_path: Optional[Path] = None,
    ):
        self.backend = (backend or config.DETECTOR_BACKEND).lower()
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else config.DETECTOR_CONFIDENCE
        )
        self.nms_threshold = (
            nms_threshold if nms_threshold is not None else config.DETECTOR_NMS_THRESHOLD
        )
        self.weights_path = weights_path or (config.WEIGHTS_DIR / config.YUNET_WEIGHTS_FILE)

        self._yunet = None
        self._haar = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes detector model with fallback mechanism."""
        if self.backend == "yunet" and self.weights_path.is_file():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    model=str(self.weights_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU,
                )
                return
            except Exception:
                self._yunet = None

        # Fallback to OpenCV Haar Cascade classifier
        haar_path = find_haar_cascade_xml()
        if haar_path:
            self._haar = cv2.CascadeClassifier(haar_path)
        self.backend = "haar"

    def detect(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        max_faces: int = 10,
    ) -> List[FaceDetectionResult]:
        """Detects faces, aligns facial crops, and assesses quality.

        Args:
            image_input: File path, raw bytes, or BGR numpy array.
            max_faces: Maximum number of faces to return (sorted by confidence/area).

        Returns:
            List[FaceDetectionResult]: Detected faces with bounding boxes, landmarks,
            canonical 112x112 crops, and quality metrics.
        """
        img = load_image(image_input)
        h, w = img.shape[:2]

        # Check if incoming image is already a 112x112 aligned image (e.g. from UMDFace)
        if (w, h) == config.FACE_TARGET_SIZE and self._looks_prealigned(img):
            quality = assess_image_quality(img, bbox=(0, 0, w, h))
            synthetic_landmarks = REFERENCE_5PTS_112.copy()
            return [
                FaceDetectionResult(
                    bbox=(0, 0, w, h),
                    confidence=1.0,
                    landmarks=synthetic_landmarks,
                    aligned_face=img.copy(),
                    quality=quality,
                )
            ]

        if self._yunet is not None:
            return self._detect_yunet(img, max_faces)
        if self._haar is not None:
            return self._detect_haar(img, max_faces)

        # Extreme fallback if no cascade xml exists: treat central crop as detection
        quality = assess_image_quality(img, bbox=(0, 0, w, h))
        return [
            FaceDetectionResult(
                bbox=(0, 0, w, h),
                confidence=0.70,
                landmarks=REFERENCE_5PTS_112.copy(),
                aligned_face=cv2.resize(img, config.FACE_TARGET_SIZE),
                quality=quality,
            )
        ]

    def _looks_prealigned(self, img: np.ndarray) -> bool:
        """Heuristic check to determine if an image is already a normalized facial crop."""
        if img.shape[:2] != (112, 112):
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return bool(np.std(gray) > 5.0)

    def _detect_yunet(self, img: np.ndarray, max_faces: int) -> List[FaceDetectionResult]:
        """Runs inference via OpenCV FaceDetectorYN."""
        h, w = img.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(img)

        if faces is None or len(faces) == 0:
            return []

        results: List[FaceDetectionResult] = []
        for face in faces[:max_faces]:
            bbox = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
            score = float(face[14])

            right_eye = [face[4], face[5]]
            left_eye = [face[6], face[7]]
            nose = [face[8], face[9]]
            right_mouth = [face[10], face[11]]
            left_mouth = [face[12], face[13]]

            landmarks = np.array(
                [left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32
            )

            aligned = align_face_5pts(img, landmarks, target_size=config.FACE_TARGET_SIZE)
            quality = assess_image_quality(img, bbox=bbox)

            results.append(
                FaceDetectionResult(
                    bbox=bbox,
                    confidence=score,
                    landmarks=landmarks,
                    aligned_face=aligned,
                    quality=quality,
                )
            )

        return results

    def _detect_haar(self, img: np.ndarray, max_faces: int) -> List[FaceDetectionResult]:
        """Fallback detector using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(config.MIN_FACE_RESOLUTION, config.MIN_FACE_RESOLUTION),
        )

        if len(faces) == 0:
            return []

        sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        results: List[FaceDetectionResult] = []

        for x, y, w, h in sorted_faces:
            bbox = (int(x), int(y), int(w), int(h))
            left_eye = [x + 0.35 * w, y + 0.38 * h]
            right_eye = [x + 0.65 * w, y + 0.38 * h]
            nose = [x + 0.50 * w, y + 0.55 * h]
            left_mouth = [x + 0.38 * w, y + 0.75 * h]
            right_mouth = [x + 0.62 * w, y + 0.75 * h]

            landmarks = np.array(
                [left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32
            )

            aligned = align_face_5pts(img, landmarks, target_size=config.FACE_TARGET_SIZE)
            quality = assess_image_quality(img, bbox=bbox)

            results.append(
                FaceDetectionResult(
                    bbox=bbox,
                    confidence=0.85,
                    landmarks=landmarks,
                    aligned_face=aligned,
                    quality=quality,
                )
            )

        return results
