"""PrivateAttend: Face Recognition and Open-Set Decision Engine.

Orchestrates detection, quality gating, embedding extraction, and open-set
decision boundaries (KNOWN, UNKNOWN, UNCERTAIN) with threshold calibration algorithms.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from config import config
from modules.detection import FaceDetectionResult, FaceDetector
from modules.recognition import FaceRecognizer


class DecisionState(str, Enum):
    """Categorical open-set identification decision states."""
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class RecognitionMatchResult:
    """Detailed matching result against enrolled gallery templates."""
    decision: DecisionState
    student_id: Optional[str]
    matched_score: float
    runner_up_id: Optional[str]
    runner_up_score: float
    margin: float
    explanation: str


@dataclass
class ProbeProcessingResult:
    """Full face processing output for a probe image."""
    success: bool
    detection: Optional[FaceDetectionResult]
    match: Optional[RecognitionMatchResult]
    error_message: Optional[str] = None


class FaceEngine:
    """Orchestrates face localization, quality validation, and open-set identification."""

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        theta_known: Optional[float] = None,
        theta_unknown: Optional[float] = None,
        top2_margin: Optional[float] = None,
    ):
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.theta_known = theta_known if theta_known is not None else config.THETA_KNOWN
        self.theta_unknown = theta_unknown if theta_unknown is not None else config.THETA_UNKNOWN
        self.top2_margin = top2_margin if top2_margin is not None else config.TOP2_MARGIN

    def match_probe(
        self,
        probe_embedding: np.ndarray,
        gallery_templates: Dict[str, np.ndarray],
    ) -> RecognitionMatchResult:
        """Classifies probe embedding against gallery using open-set dual-threshold boundaries.

        Args:
            probe_embedding: 512-d unit-normalized probe vector.
            gallery_templates: Mapping of student_id -> 512-d unit-normalized template.

        Returns:
            RecognitionMatchResult with KNOWN, UNKNOWN, or UNCERTAIN decision.
        """
        if not gallery_templates:
            return RecognitionMatchResult(
                decision=DecisionState.UNKNOWN,
                student_id=None,
                matched_score=0.0,
                runner_up_id=None,
                runner_up_score=0.0,
                margin=0.0,
                explanation="No enrolled student templates exist in the gallery.",
            )

        # Compute cosine similarity against all gallery templates
        similarities: List[Tuple[str, float]] = []
        for sid, tmpl in gallery_templates.items():
            sim = self.recognizer.compute_similarity(probe_embedding, tmpl)
            similarities.append((sid, sim))

        # Sort descending by similarity score
        similarities.sort(key=lambda item: item[1], reverse=True)

        top1_id, top1_score = similarities[0]
        if len(similarities) > 1:
            top2_id, top2_score = similarities[1]
            margin = top1_score - top2_score
        else:
            top2_id, top2_score = None, 0.0
            margin = top1_score

        # Open-Set Decision Logic
        if top1_score >= self.theta_known:
            if margin >= self.top2_margin:
                decision = DecisionState.KNOWN
                explanation = (
                    f"Match confirmed for {top1_id} (score {top1_score:.3f} >= {self.theta_known:.3f}, "
                    f"margin {margin:.3f} >= {self.top2_margin:.3f})."
                )
            else:
                decision = DecisionState.UNCERTAIN
                explanation = (
                    f"Ambiguous top candidates ({top1_id} vs {top2_id}): "
                    f"margin {margin:.3f} < required {self.top2_margin:.3f}."
                )
        elif top1_score < self.theta_unknown:
            decision = DecisionState.UNKNOWN
            explanation = (
                f"No enrolled identity matched (top score {top1_score:.3f} < {self.theta_unknown:.3f}). "
                "Subject is classified as unknown/unregistered."
            )
        else:
            decision = DecisionState.UNCERTAIN
            explanation = (
                f"Score {top1_score:.3f} lies in uncertainty band "
                f"[{self.theta_unknown:.3f}, {self.theta_known:.3f}]. Manual verification required."
            )

        return RecognitionMatchResult(
            decision=decision,
            student_id=top1_id if decision == DecisionState.KNOWN else None,
            matched_score=top1_score,
            runner_up_id=top2_id,
            runner_up_score=top2_score,
            margin=margin,
            explanation=explanation,
        )

    def process_probe_image(
        self,
        image_input: Union[str, np.ndarray],
        gallery_templates: Dict[str, np.ndarray],
        allow_synthetic: bool = False,
    ) -> ProbeProcessingResult:
        """Executes full detection, quality screening, and open-set matching.

        Args:
            image_input: Image file path, raw bytes, or BGR array.
            gallery_templates: Enrolled identity templates.
            allow_synthetic: For software unit tests when weights are absent.

        Returns:
            ProbeProcessingResult containing detection, match, and diagnostics.
        """
        try:
            detections = self.detector.detect(image_input, max_faces=1)
        except Exception as e:
            return ProbeProcessingResult(
                success=False, detection=None, match=None, error_message=f"Detection failed: {e}"
            )

        if not detections:
            return ProbeProcessingResult(
                success=False,
                detection=None,
                match=None,
                error_message="No face detected in the input image.",
            )

        best_detection = detections[0]
        if not best_detection.quality.is_acceptable:
            reasons = "; ".join(best_detection.quality.failure_reasons)
            return ProbeProcessingResult(
                success=False,
                detection=best_detection,
                match=None,
                error_message=f"Image quality rejected: {reasons}",
            )

        try:
            probe_emb = self.recognizer.extract_embedding(
                best_detection.aligned_face, allow_synthetic=allow_synthetic
            )
        except Exception as e:
            return ProbeProcessingResult(
                success=False,
                detection=best_detection,
                match=None,
                error_message=f"Embedding extraction failed: {e}",
            )

        match_res = self.match_probe(probe_emb, gallery_templates)
        return ProbeProcessingResult(
            success=True,
            detection=best_detection,
            match=match_res,
            error_message=None,
        )

    @staticmethod
    def calibrate_thresholds(
        val_known_scores: List[float],
        val_unknown_scores: List[float],
        target_far: float = 0.01,
    ) -> Tuple[float, float]:
        """Calibrates theta_known and theta_unknown strictly on validation partition.

        Args:
            val_known_scores: Genuine matching scores from validation identities.
            val_unknown_scores: Impostor / unknown matching scores from validation identities.
            target_far: Allowable False Acceptance Rate (default 1%).

        Returns:
            Tuple[float, float]: Calibrated (theta_known, theta_unknown).

        Raises:
            ValueError: If validation score lists are empty or insufficient.
        """
        if not val_unknown_scores or not val_known_scores:
            raise ValueError("Validation scores required for empirical threshold calibration.")

        unknown_sorted = np.sort(val_unknown_scores)
        n_unknown = len(unknown_sorted)

        # Find threshold where fraction of unknowns >= theta is <= target_far
        # FAR = sum(val_unknown_scores >= theta) / n_unknown
        # Cutoff index: int((1 - target_far) * n_unknown)
        cutoff_idx = int(np.floor((1.0 - target_far) * n_unknown))
        cutoff_idx = min(max(0, cutoff_idx), n_unknown - 1)
        theta_known = float(unknown_sorted[cutoff_idx])

        # theta_unknown is calibrated at 95% unknown rejection or known score 1st percentile
        unknown_rejection_idx = int(np.floor(0.95 * n_unknown))
        unknown_rejection_idx = min(max(0, unknown_rejection_idx), n_unknown - 1)
        theta_unknown = float(unknown_sorted[unknown_rejection_idx] * 0.90)

        # Enforce consistency: theta_unknown must be strictly less than theta_known
        if theta_unknown >= theta_known:
            theta_unknown = max(0.10, theta_known - 0.15)

        return theta_known, theta_unknown
