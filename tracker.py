from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from detector import DetectionResult


@dataclass
class TrackState:
    """State for a single tracked object used for stabilization and counting."""

    track_id: int
    last_center: Tuple[int, int]
    class_history: List[str]
    conf_history: List[float]
    stable_class: Optional[str] = None
    stable_confidence: float = 0.0


class ClassificationStabilizer:
    """
    Temporal smoothing of class predictions for each track ID using
    majority voting and confidence averaging.
    """

    def __init__(self, history_size: int = 10) -> None:
        self.history_size = history_size
        self.tracks: Dict[int, TrackState] = {}

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Update stabilization state and return detections with stabilized labels
        (cls_name) and confidence values.
        """
        for det in detections:
            if det.track_id is None:
                # No tracking ID -> cannot stabilize across frames
                continue

            track_id = det.track_id
            cx, cy = det.center

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    last_center=(cx, cy),
                    class_history=[det.cls_name],
                    conf_history=[det.confidence],
                )
            else:
                state = self.tracks[track_id]
                state.last_center = (cx, cy)
                state.class_history.append(det.cls_name)
                state.conf_history.append(det.confidence)

                if len(state.class_history) > self.history_size:
                    state.class_history.pop(0)
                if len(state.conf_history) > self.history_size:
                    state.conf_history.pop(0)

            # Apply stabilization: update stable_class every frame using the
            # current rolling-window majority class. This prevents early-freeze
            # where the very first detection permanently locks the class label.
            state = self.tracks[track_id]
            majority_cls = self._majority_class(state.class_history)
            avg_conf = self._average_confidence(state.conf_history)

            # Always update to reflect the current best majority class
            state.stable_class = majority_cls
            state.stable_confidence = avg_conf

            # Overwrite detection label with the temporally stabilized class
            det.cls_name = state.stable_class
            det.confidence = avg_conf if avg_conf > 0 else det.confidence

        return detections

    @staticmethod
    def _majority_class(history: List[str]) -> str:
        freq: Dict[str, int] = {}
        for cls in history:
            freq[cls] = freq.get(cls, 0) + 1
        return max(freq.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _average_confidence(history: List[float]) -> float:
        if not history:
            return 0.0
        return float(sum(history) / len(history))

