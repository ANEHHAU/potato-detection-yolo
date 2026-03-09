from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple

from detector import DetectionResult

_counter_logger = logging.getLogger("potato_counter")


class CountingMode(str, Enum):
    LINE_CROSSING = "line"
    ZONE = "zone"
    DIRECTION = "direction"


@dataclass
class LineCountingConfig:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ZoneCountingConfig:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DirectionCountingConfig:
    x_ref: int  # reference vertical line for direction


class ObjectCounter:
    """
    Flexible counting logic for:
    - Line crossing
    - Zone entry/exit
    - Direction-based counting
    """

    def __init__(
        self,
        mode: CountingMode = CountingMode.LINE_CROSSING,
        line_config: LineCountingConfig | None = None,
        zone_config: ZoneCountingConfig | None = None,
        direction_config: DirectionCountingConfig | None = None,
    ) -> None:
        self.mode = mode
        self.line_config = line_config
        self.zone_config = zone_config
        self.direction_config = direction_config

        # track_id -> last center position
        self.last_positions: Dict[int, Tuple[int, int]] = {}
        # track_id -> full history of centers (prev, curr, ...)
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        # track_id -> has been counted
        self.counted_ids: Dict[int, bool] = {}

        self.total_count: int = 0
        self.newly_counted_ids: list[int] = []

    def reset(self) -> None:
        self.last_positions.clear()
        self.track_history.clear()
        self.counted_ids.clear()
        self.total_count = 0
        self.newly_counted_ids = []

    def update(
        self,
        detections: List[DetectionResult],
        countable_ids: Set[int] | None = None,
    ) -> int:
        """
        Update counting using current frame detections.
        countable_ids: if provided, only count objects with these track_ids (e.g. those
        with confirmed final class from Define Zone).
        Returns cumulative total count.
        """
        self.newly_counted_ids = []

        for det in detections:
            if det.track_id is None:
                continue

            track_id = det.track_id
            cx, cy = det.center
            # Maintain explicit center history per ID
            history = self.track_history.setdefault(track_id, [])
            history.append((cx, cy))

            # Need at least two positions to check a crossing
            if len(history) < 2:
                self.last_positions[track_id] = (cx, cy)
                continue

            if self.counted_ids.get(track_id, False):
                self.last_positions[track_id] = (cx, cy)
                continue

            # Only count if track_id is in countable_ids (has confirmed class)
            if countable_ids is not None and track_id not in countable_ids:
                px, py = history[-2]
                cx2, cy2 = history[-1]
                crossed = (
                    self.mode == CountingMode.LINE_CROSSING
                    and self.line_config
                    and self._check_line_cross(px, py, cx2, cy2, self.line_config)
                )
                if crossed:
                    _counter_logger.debug(
                        "Object id=%d crosses COUNT line but has no confirmed class (ignore)",
                        track_id,
                    )
                self.last_positions[track_id] = (cx, cy)
                continue

            # Previous vs current center positions
            prev_x, prev_y = history[-2]
            curr_x, curr_y = history[-1]

            if self.mode == CountingMode.LINE_CROSSING and self.line_config:
                if self._check_line_cross(prev_x, prev_y, curr_x, curr_y, self.line_config):
                    self._mark_counted(track_id)

            elif self.mode == CountingMode.ZONE and self.zone_config:
                if self._check_zone_transition(prev_x, prev_y, curr_x, curr_y, self.zone_config):
                    self._mark_counted(track_id)

            elif self.mode == CountingMode.DIRECTION and self.direction_config:
                if self._check_direction(prev_x, curr_x, self.direction_config.x_ref):
                    self._mark_counted(track_id)

            self.last_positions[track_id] = (curr_x, curr_y)

        return self.total_count

    def _mark_counted(self, track_id: int) -> None:
        self.counted_ids[track_id] = True
        self.total_count += 1
        self.newly_counted_ids.append(track_id)

    @staticmethod
    def _orientation(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
        val = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    @staticmethod
    def _on_segment(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> bool:
        return min(ax, cx) <= bx <= max(ax, cx) and min(ay, cy) <= by <= max(ay, cy)

    def _check_line_cross(
        self,
        prev_x: int,
        prev_y: int,
        curr_x: int,
        curr_y: int,
        cfg: LineCountingConfig,
    ) -> bool:
        """
        Check if movement segment (prev -> curr) crosses the counting line
        using center Y-coordinates. Supports both directions.
        """
        y_line = (cfg.y1 + cfg.y2) / 2.0
        # Crossing from above to below (downward)
        if prev_y < y_line <= curr_y:
            return True
        # Crossing from below to above (upward)
        if curr_y < y_line <= prev_y:
            return True
        return False

    @staticmethod
    def _inside_zone(x: int, y: int, cfg: ZoneCountingConfig) -> bool:
        return cfg.x1 <= x <= cfg.x2 and cfg.y1 <= y <= cfg.y2

    def _check_zone_transition(
        self,
        prev_x: int,
        prev_y: int,
        curr_x: int,
        curr_y: int,
        cfg: ZoneCountingConfig,
    ) -> bool:
        """Count when object enters + exits a zone once."""
        prev_inside = self._inside_zone(prev_x, prev_y, cfg)
        curr_inside = self._inside_zone(curr_x, curr_y, cfg)

        # Enter then exit -> count
        return prev_inside and not curr_inside

    @staticmethod
    def _check_direction(prev_x: int, curr_x: int, x_ref: int) -> bool:
        """
        Increase count when object passes reference line moving
        left->right or right->left.
        """
        return (prev_x < x_ref <= curr_x) or (prev_x > x_ref >= curr_x)

