"""PrivateAttend: Image Ingestion, Normalization, and Quality Assessment.

Handles robust input loading, OpenCV conversions, quality metrics calculation
(Laplacian blur, illumination limits, resolution checks), and canonical normalization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import base64
import io
import cv2
import numpy as np

from config import config


class InvalidImageError(ValueError):
    """Raised when an image cannot be read, decoded, or parsed."""
    pass


class LowQualityImageError(ValueError):
    """Raised when a facial image fails minimum quality or resolution criteria."""
    pass


@dataclass
class ImageQualityResult:
    """Stores quantitative and categorical image quality metrics."""
    is_acceptable: bool
    sharpness: float
    brightness: float
    contrast: float
    width: int
    height: int
    failure_reasons: List[str] = field(default_factory=list)


def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Decodes an image from a path, raw bytes, base64 data, or numpy array.

    Args:
        source: Image file path, binary byte stream, base64 encoded string, or ndarray.

    Returns:
        np.ndarray: 3-channel BGR image matrix with dtype=np.uint8.

    Raises:
        InvalidImageError: If source cannot be converted into a valid 3-channel image.
    """
    if source is None:
        raise InvalidImageError("Image source cannot be None.")

    # 1. Existing numpy array
    if isinstance(source, np.ndarray):
        img = source
        if img.size == 0:
            raise InvalidImageError("Input numpy image array is empty.")
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 3:
                return img.copy()
        raise InvalidImageError(f"Unsupported numpy array dimensions: {img.shape}")

    # 2. Raw bytes or BytesIO
    if isinstance(source, (bytes, bytearray)):
        data = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError("Failed to decode image from raw bytes.")
        return img

    if isinstance(source, io.BytesIO):
        data = np.frombuffer(source.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError("Failed to decode image from BytesIO stream.")
        return img

    # 3. Path or string
    if isinstance(source, (str, Path)):
        str_src = str(source).strip()
        # Handle base64 encoded string
        if str_src.startswith("data:image") or len(str_src) > 512 and " " not in str_src and "/" in str_src:
            if "," in str_src:
                str_src = str_src.split(",", 1)[1]
            try:
                decoded_bytes = base64.b64decode(str_src)
                data = np.frombuffer(decoded_bytes, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass  # Fall through to path loading

        path = Path(str_src)
        if not path.is_file():
            raise InvalidImageError(f"Image file does not exist: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError(f"OpenCV failed to read image file: {path}")
        return img

    raise InvalidImageError(f"Unsupported image source type: {type(source)}")


def assess_image_quality(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    blur_threshold: Optional[float] = None,
    min_brightness: Optional[float] = None,
    max_brightness: Optional[float] = None,
    min_contrast: Optional[float] = None,
    min_resolution: Optional[int] = None,
) -> ImageQualityResult:
    """Evaluates facial image sharpness, illumination, contrast, and resolution.

    Args:
        image: Full BGR image (H, W, 3).
        bbox: Optional (x, y, w, h) bounding box of the facial region.
        blur_threshold: Minimum Laplacian variance (defaults to config).
        min_brightness: Minimum average grayscale brightness (defaults to config).
        max_brightness: Maximum average grayscale brightness (defaults to config).
        min_contrast: Minimum grayscale standard deviation (defaults to config).
        min_resolution: Minimum spatial resolution in pixels (defaults to config).

    Returns:
        ImageQualityResult with quantitative values and compliance flags.
    """
    blur_th = blur_threshold if blur_threshold is not None else config.BLUR_THRESHOLD
    min_b = min_brightness if min_brightness is not None else config.MIN_BRIGHTNESS
    max_b = max_brightness if max_brightness is not None else config.MAX_BRIGHTNESS
    min_c = min_contrast if min_contrast is not None else config.MIN_CONTRAST
    min_res = min_resolution if min_resolution is not None else config.MIN_FACE_RESOLUTION

    if bbox is not None:
        x, y, w, h = bbox
        # Clip bounding box within image boundaries
        img_h, img_w = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return ImageQualityResult(
                is_acceptable=False,
                sharpness=0.0,
                brightness=0.0,
                contrast=0.0,
                width=0,
                height=0,
                failure_reasons=["Invalid or zero-area face bounding box."],
            )
        eval_region = image[y1:y2, x1:x2]
        reg_w, reg_h = (x2 - x1), (y2 - y1)
    else:
        eval_region = image
        reg_h, reg_w = image.shape[:2]

    # Grayscale conversion for photometric and frequency analysis
    gray = cv2.cvtColor(eval_region, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # 2. Illumination and contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Evaluate against quality gates
    reasons: List[str] = []
    if sharpness < blur_th:
        reasons.append(f"Image too blurry (sharpness {sharpness:.1f} < threshold {blur_th:.1f}).")
    if brightness < min_b:
        reasons.append(f"Image underexposed / too dark (brightness {brightness:.1f} < {min_b:.1f}).")
    elif brightness > max_b:
        reasons.append(f"Image overexposed / too bright (brightness {brightness:.1f} > {max_b:.1f}).")
    if contrast < min_c:
        reasons.append(f"Image lacks sufficient contrast ({contrast:.1f} < {min_c:.1f}).")
    if reg_w < min_res or reg_h < min_res:
        reasons.append(f"Face resolution ({reg_w}x{reg_h}) below minimum ({min_res}x{min_res}).")

    return ImageQualityResult(
        is_acceptable=(len(reasons) == 0),
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        width=reg_w,
        height=reg_h,
        failure_reasons=reasons,
    )


def normalize_facial_crop(
    crop: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Resizes and normalizes an aligned facial crop into float32 [-1.0, 1.0] format.

    Args:
        crop: (H, W, 3) BGR uint8 facial image.
        target_size: Output (width, height) tuple, canonical is (112, 112).

    Returns:
        np.ndarray: Normalized float32 array of shape (3, H, W) with values in [-1.0, 1.0].
    """
    if crop is None or crop.size == 0:
        raise InvalidImageError("Cannot normalize an empty facial crop.")

    if (crop.shape[1], crop.shape[0]) != target_size:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Normalize to [-1.0, 1.0]
    normalized = (rgb.astype(np.float32) - 127.5) / 128.0

    # Transpose to channel-first (C, H, W) format expected by PyTorch/ONNX
    tensor = np.transpose(normalized, (2, 0, 1))
    return tensor
