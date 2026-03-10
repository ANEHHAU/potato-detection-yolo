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
        # track_id -> history of centers (capped at TRACK_HISTORY_MAX to prevent memory growth)
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_history_max: int = 5
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
            # Maintain center history per ID; keep only last N positions
            history = self.track_history.setdefault(track_id, [])
            history.append((cx, cy))
            if len(history) > self.track_history_max:
                history[:] = history[-self.track_history_max :]

            # Need at least two positions to check a crossing
            if len(history) < 2:
                self.last_positions[track_id] = (cx, cy)
                continue

            if self.counted_ids.get(track_id, False):
                # Already counted — skip silently
                self.last_positions[track_id] = (cx, cy)
                continue

            # Only count if track_id is in countable_ids (has confirmed final class)
            if countable_ids is not None and track_id not in countable_ids:
                # Object has not yet exited the DEFINE ZONE — check if it crossed
                # the line anyway so we can log the miss for debugging.
                prev_x2, prev_y2 = history[-2]
                curr_x2, curr_y2 = history[-1]
                if (
                    self.mode == CountingMode.LINE_CROSSING
                    and self.line_config
                    and self._check_line_cross(prev_x2, prev_y2, curr_x2, curr_y2, self.line_config)
                ):
                    _counter_logger.debug(
                        "[COUNT LINE] id=%d crossed line but has NO confirmed class yet \u2014 not counted",
                        track_id,
                    )
                self.last_positions[track_id] = (cx, cy)
                continue

            # ---- Object is countable AND not yet counted ----
            prev_x, prev_y = history[-2]
            curr_x, curr_y = history[-1]

            if self.mode == CountingMode.LINE_CROSSING and self.line_config:
                if self._check_line_cross(prev_x, prev_y, curr_x, curr_y, self.line_config):
                    _counter_logger.debug(
                        "[COUNT LINE CROSSED] id=%d crossed COUNT LINE \u2014 incrementing counter (total will be %d)",
                        track_id, self.total_count + 1,
                    )
                    self._mark_counted(track_id)
                else:
                    _counter_logger.debug(
                        "[COUNT LINE] id=%d is countable, prev=(%d,%d) curr=(%d,%d) \u2014 no crossing yet",
                        track_id, prev_x, prev_y, curr_x, curr_y,
                    )

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
        True if the movement segment (prev→curr) crosses the count line segment
        (cfg.x1,cfg.y1)→(cfg.x2,cfg.y2).

        Uses the standard 2-D segment-intersection test based on cross-product
        orientation so it works for horizontal, vertical, and diagonal count lines.
        """
        # Movement segment: A(prev) → B(curr)
        ax, ay = prev_x, prev_y
        bx, by = curr_x, curr_y
        # Count line segment: C → D
        cx, cy = cfg.x1, cfg.y1
        dx, dy = cfg.x2, cfg.y2

        def _cross(ox, oy, px, py, qx, qy) -> float:
            """Signed cross product of vectors (OP × OQ)."""
            return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

        d1 = _cross(cx, cy, dx, dy, ax, ay)
        d2 = _cross(cx, cy, dx, dy, bx, by)
        d3 = _cross(ax, ay, bx, by, cx, cy)
        d4 = _cross(ax, ay, bx, by, dx, dy)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # Collinear edge cases: check if endpoint lies on the other segment
        def _on_seg(ox, oy, px, py, qx, qy) -> bool:
            return (
                min(ox, px) <= qx <= max(ox, px)
                and min(oy, py) <= qy <= max(oy, py)
            )

        if d1 == 0 and _on_seg(cx, cy, dx, dy, ax, ay):
            return True
        if d2 == 0 and _on_seg(cx, cy, dx, dy, bx, by):
            return True
        if d3 == 0 and _on_seg(ax, ay, bx, by, cx, cy):
            return True
        if d4 == 0 and _on_seg(ax, ay, bx, by, dx, dy):
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

