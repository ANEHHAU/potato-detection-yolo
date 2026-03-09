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

    def update_from_detection(self, det: DetectionResult) -> None:
        """
        Update aggregates based on a single (already-counted) potato.
        """
        self.stats.total += 1

        cls = det.cls_name
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

