from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from detector import DetectionResult


@dataclass
class PotatoStats:
    total: int = 0
    good: int = 0
    defected: int = 0
    damaged: int = 0
    diseased_fungal: int = 0
    sprouted: int = 0
    na: int = 0

    @property
    def defect_rate(self) -> float:
        defective = (
            self.defected + self.damaged + self.diseased_fungal + self.sprouted
        )
        return (defective / self.total) * 100.0 if self.total > 0 else 0.0


class StatisticsManager:
    """
    Maintains real-time potato quality statistics.
    """

    def __init__(self) -> None:
        self.stats = PotatoStats()

    def reset(self) -> None:
        self.stats = PotatoStats()

    # Canonical class name mapping for consistent statistics (handles model variants)
    _CLASS_NORMALIZE: Dict[str, str] = {
        "potato": "Potato",
        "good potato": "Potato",
        "defected potato": "Defected potato",
        "defective potato": "Defected potato",
        "damaged potato": "Damaged potato",
        "diseased-fungal potato": "Diseased-fungal potato",
        "sprouted potato": "Sprouted potato",
    }

    def _normalize_class(self, cls_name: str) -> str:
        if not cls_name:
            return ""
        key = cls_name.strip().lower()
        return self._CLASS_NORMALIZE.get(key, cls_name)

    def update_from_detection(self, det: DetectionResult) -> None:
        """
        Update aggregates based on a single (already-counted) potato.
        Class names are normalized for consistent mapping.
        """
        self.stats.total += 1

        cls = self._normalize_class(det.cls_name)
        if cls == "Potato":
            self.stats.good += 1
        elif cls == "Defected potato":
            self.stats.defected += 1
        elif cls == "Damaged potato":
            self.stats.damaged += 1
        elif cls == "Diseased-fungal potato":
            self.stats.diseased_fungal += 1
        elif cls == "Sprouted potato":
            self.stats.sprouted += 1

    def add_quality_only(self, cls_name: str) -> None:
        """
        Increment quality counters for an object that was already counted 
        in the total sum (e.g. from counting line crossing).
        """
        cls = self._normalize_class(cls_name)
        if cls == "Potato":
            self.stats.good += 1
        elif cls == "Defected potato":
            self.stats.defected += 1
        elif cls == "Damaged potato":
            self.stats.damaged += 1
        elif cls == "Diseased-fungal potato":
            self.stats.diseased_fungal += 1
        elif cls == "Sprouted potato":
            self.stats.sprouted += 1

    def add_na_only(self) -> None:
        """Increment N/A counter only."""
        self.stats.na += 1

    def move_na_to_quality(self, cls_name: str) -> None:
        """Move one count from N/A to a specific quality class."""
        if self.stats.na > 0:
            self.stats.na -= 1
            self.add_quality_only(cls_name)
        else:
            # Fallback if somehow not recorded as NA yet
            self.add_quality_only(cls_name)
