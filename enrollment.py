import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EnrollmentResult:
    template: Optional[np.ndarray]
    accepted: int
    mean_quality: float
    rejections: List[str]


def build_template(images: List[bytes], detector, recognizer) -> EnrollmentResult:
    embs, quals, rejections = [], [], []
    for i, raw in enumerate(images, 1):
        try:
            dets = detector.detect(raw, max_faces=1)
            if not dets:
                rejections.append(f"image {i}: no face detected")
                continue
            d = dets[0]
            if not d.quality.is_acceptable:
                rejections.append(f"image {i}: " + "; ".join(d.quality.failure_reasons))
                continue
            emb = recognizer.extract_embedding(d.aligned_face)  # no allow_synthetic
            q = float(d.quality.sharpness)
        except Exception as e:
            log.exception("enrollment failed on image %d", i)
            rejections.append(f"image {i}: {type(e).__name__}: {e}")
            continue
        embs.append(emb)   # append both only after both succeeded
        quals.append(q)

    if not embs:
        return EnrollmentResult(None, 0, 0.0, rejections)
    tpl = recognizer.aggregate_templates(embs, weights=quals)
    return EnrollmentResult(tpl, len(embs), float(np.mean(quals)), rejections)