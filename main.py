from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Dict, List

import cv2

# Pipeline debug logging (set DEBUG_PIPELINE=1 to enable)
_DEBUG = os.environ.get("DEBUG_PIPELINE", "0") == "1"
logger = logging.getLogger("potato_pipeline")
import yaml
import numpy as np
from PyQt5 import QtCore, QtWidgets

from counter import (
    CountingMode,
    DirectionCountingConfig,
    LineCountingConfig,
    ObjectCounter,
    ZoneCountingConfig,
)
from database import DatabaseConfig, EventLogger
from detector import PotatoDetector
from preprocessing import PreprocessConfig, Preprocessor
from statistics import StatisticsManager
from tracker import ClassificationStabilizer
from ui import MainWindow, UIConfig, create_app_window
from video_capture import CaptureConfig, VideoCapture


class FrameProcessor(QtCore.QObject):
    """
    Ties together the full industrial pipeline:

    Camera -> Capture -> Preprocessing -> Detection (YOLO) ->
    Tracking (ByteTrack via YOLO) -> Class Stabilization ->
    Counting -> Statistics -> Database -> UI
    """

    # original_frame, processed_frame, stats
    frame_processed = QtCore.pyqtSignal(object, object, object)
    position_updated = QtCore.pyqtSignal(int, int)  # current_frame, total_frames
    state_changed = QtCore.pyqtSignal(str)  # "stopped", "playing", "paused", "ended"

    def __init__(
        self,
        capture_config: CaptureConfig,
        preprocess_config: PreprocessConfig,
        db_config: DatabaseConfig,
        model_path: str,
        model_device: str,
        det_conf: float,
        det_iou: float,
        counting_mode: CountingMode,
        line_config: LineCountingConfig | None,
        zone_config: ZoneCountingConfig | None,
        direction_config: DirectionCountingConfig | None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)

        self._base_capture_config = capture_config
        self._model_path = model_path
        self._model_device = model_device
        self._det_conf = det_conf
        self._det_iou = det_iou

        self.video = VideoCapture(capture_config)
        self.preprocessor = Preprocessor(preprocess_config)
        self.detector = PotatoDetector(
            model_path=model_path,
            confidence_threshold=det_conf,
            iou_threshold=det_iou,
            device=model_device,
        )
        self.stabilizer = ClassificationStabilizer()
        self.counter = ObjectCounter(
            mode=counting_mode,
            line_config=line_config,
            zone_config=zone_config,
            direction_config=direction_config,
        )
        self.stats_manager = StatisticsManager()
        self.db = EventLogger(db_config)

        self._running = False
        self._paused: bool = False
        self._at_end: bool = False
        self._frame_index: int = 0
        self.roi_polygon: list[tuple[int, int]] | None = None
        self.define_zone_polygon: list[tuple[int, int]] | None = None
        self.remove_background: bool = False

        # Background subtraction for foreground-only visualization / detection
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )

        # Define Zone classification aggregation: track_id -> list of (class_name, confidence)
        self._id_class_history: Dict[int, List[tuple[str, float]]] = {}
        # Final, stable class per track once decided in Define Zone
        self._final_classes: Dict[int, str] = {}
        # Minimum number of observations inside Define Zone before fixing class
        self._min_define_zone_observations: int = 1
        # Maximum predictions kept per track_id in zone history (prevents memory growth)
        self._max_zone_history: int = 20

        # Tracking lifetime for filtering out very short-lived false positives
        self._track_total_frames: Dict[int, int] = {}
        self._define_zone_inside: Dict[int, bool] = {}
        self._min_frames_for_class: int = 1

        # Track which IDs have been logged as entering ROI (for debug once-per-entry logging)
        self._roi_logged_ids: set = set()

    def start(self) -> None:
        if self._running:
            return
        self.video.open()
        self._running = True
        self._paused = False
        self._at_end = False
        self.state_changed.emit("playing")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.video.release()
        self._paused = False
        self._at_end = False
        self.state_changed.emit("stopped")

    def pause(self) -> None:
        """
        Pause processing without releasing the underlying video source.
        Used for UI Stop/Resume semantics on uploaded videos and live camera.
        """
        if not self._running or not self.video.is_opened():
            return
        self._running = False
        self._paused = True
        self.state_changed.emit("paused")

    def resume(self) -> None:
        """
        Resume processing from the current frame position after a pause.
        Does not reset counters or reopen the video source.
        """
        if self._running or not self.video.is_opened():
            return
        self._running = True
        self._paused = False
        self._at_end = False
        self.state_changed.emit("playing")

    def set_video_source(self, source) -> None:
        """Switch between camera, test video, or uploaded video."""
        was_running = self._running
        if self.video.is_opened():
            self.video.release()
        self.video = VideoCapture(
            CaptureConfig(
                source=source,
                width=self._base_capture_config.width,
                height=self._base_capture_config.height,
                fps=self._base_capture_config.fps,
            )
        )
        if was_running:
            self.start()

    def set_roi_polygon(self, pts: list[tuple[int, int]]) -> None:
        self.roi_polygon = pts

    def set_define_zone_polygon(self, pts: list[tuple[int, int]]) -> None:
        self.define_zone_polygon = pts

    def reset_session(self) -> None:
        """
        Reset counters, statistics, tracking, and (optionally) detector state
        for a fresh run (used when uploading a new video or switching modes).
        """
        self.counter.reset()
        self.stats_manager.reset()
        self.stabilizer = ClassificationStabilizer()
        # Clear ROI polygon but keep user-defined ROI if desired; here we preserve it.
        # self.roi_polygon = None
        # Recreate detector to reset internal tracking state (ByteTrack)
        self.detector = PotatoDetector(
            model_path=self._model_path,
            confidence_threshold=self._det_conf,
            iou_threshold=self._det_iou,
            device=self._model_device,
        )
        # Reset Define Zone and tracking lifetime state
        self._id_class_history.clear()
        self._final_classes.clear()
        self._track_total_frames.clear()
        self._define_zone_inside.clear()
        self._roi_logged_ids.clear()
        self._frame_index = 0
        self._paused = False
        self._at_end = False

    def preview_first_frame(self) -> None:
        """
        Load and emit a single preview frame for the current video source
        without starting the continuous processing loop.
        Used after uploading a video so the user can see the first frame.
        """
        # Open source temporarily if needed
        if not self.video.is_opened():
            try:
                self.video.open()
            except Exception:
                return

        ok, frame = self.video.read()
        if not ok or frame is None:
            self.video.release()
            return

        # Always restart from the beginning on real Start()
        if self.video.is_file_source():
            # Position is now at frame 1 after read; we want slider to show that
            current = self.video.get_frame_index()
            total = self.video.get_frame_count()
            self.position_updated.emit(current, total)

        base_frame = self.preprocessor.apply(frame)

        processed = base_frame
        if self.remove_background:
            fg_mask = self._bg_subtractor.apply(base_frame)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.medianBlur(fg_mask, 5)
            processed = cv2.bitwise_and(base_frame, base_frame, mask=fg_mask)

        # Draw static overlays (ROI / Define Zone / counting line) on processed frame
        self._draw_overlays(processed, [])

        # Emit preview frames with current statistics (likely zeroed)
        self.frame_processed.emit(base_frame, processed, self.stats_manager.stats)

        # Release after preview so Start() re-opens from the beginning
        self.video.release()

    def process_next_frame(self) -> None:
        if not self._running or not self.video.is_opened():
            return

        ok, frame = self.video.read()
        if not ok or frame is None:
            # End-of-stream handling:
            # - For uploaded videos: mark as ended but keep source open so UI can restart.
            # - For live camera: fully stop and release.
            if self.video.is_file_source():
                self._running = False
                self._at_end = True
                self.state_changed.emit("ended")
                return
            else:
                self.stop()
                return
        self._frame_index += 1
        # Preprocessing (resize, denoise, contrast)
        base_frame = self.preprocessor.apply(frame)

        # ── STAGE 1: DETECT ─────────────────────────────────────────────────────
        # Pixel-mask the frame to the ROI area when one is defined.  This improves
        # detection quality inside the conveyor zone but does NOT restrict which
        # detections survive — that is handled by the ROI filter in Stage 8.
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            roi_mask = self._build_roi_mask(base_frame.shape[:2])
            detect_frame = cv2.bitwise_and(base_frame, base_frame, mask=roi_mask)
        else:
            detect_frame = base_frame

        # ── STAGE 2: TRACK (ByteTrack, persist=True) ────────────────────────────
        # Always runs — tracking must not be gated behind any user-drawn zone.
        raw_results = self.detector.track(detect_frame, persist=True)
        if _DEBUG:
            logger.debug(
                "[DETECT OK][TRACKING OK] frame=%d raw=%d",
                self._frame_index, len(raw_results),
            )

        # ── STAGE 3: VALIDITY FILTER ─────────────────────────────────────────────
        # Drop detections that have no track_id, a degenerate bbox, or a center
        # that cannot be computed.  All survivors go into all_detections.
        all_detections = []
        for d in raw_results:
            if d.track_id is None:
                if _DEBUG:
                    logger.debug("[SKIP] no track_id frame=%d", self._frame_index)
                continue
            if d.x2 <= d.x1 or d.y2 <= d.y1:
                if _DEBUG:
                    logger.debug("[SKIP] bad bbox id=%s frame=%d", d.track_id, self._frame_index)
                continue
            try:
                _ = d.center
            except Exception:
                if _DEBUG:
                    logger.debug("[SKIP] center error id=%s frame=%d", d.track_id, self._frame_index)
                continue
            all_detections.append(d)

        # ── STAGE 4: OVERLAP SUPPRESSION (NMS across all classes) ───────────────
        all_detections = self._suppress_overlaps(all_detections, iou_threshold=0.5)

        # ── STAGE 5: CAPTURE RAW CLASS MAP (before stabilizer mutates labels) ───
        # Snapshot the unmodified YOLO class+confidence per track_id so the
        # DEFINE ZONE history stores per-frame model output, not smoothed labels.
        raw_class_map: Dict[int, tuple[str, float]] = {
            d.track_id: (d.cls_name, float(d.confidence))
            for d in all_detections
            if d.track_id is not None
        }

        # ── STAGE 6: CLASSIFICATION STABILIZER (temporal majority-vote) ─────────
        # Overwrites det.cls_name with the rolling-window majority class per ID.
        all_detections = self.stabilizer.update(all_detections)

        # ── STAGE 7: BACKGROUND REMOVAL ──────────────────────────────────────────
        # Builds a mask from every tracked bbox in all_detections so potato pixels
        # are preserved and the background becomes white.  Runs unconditionally —
        # ROI membership is irrelevant for background removal.
        processed = base_frame.copy()
        if self.remove_background and all_detections:
            h, w = base_frame.shape[:2]
            potato_mask = np.zeros((h, w), dtype=np.uint8)
            for det in all_detections:
                x1 = max(0, det.x1)
                y1 = max(0, det.y1)
                x2 = min(w, det.x2)
                y2 = min(h, det.y2)
                if x2 > x1 and y2 > y1:
                    potato_mask[y1:y2, x1:x2] = 255
            processed[potato_mask == 0] = [255, 255, 255]
            if _DEBUG:
                logger.debug(
                    "[BACKGROUND REMOVAL OK] frame=%d objects=%d",
                    self._frame_index, len(all_detections),
                )

        # ── STAGE 8: ROI FILTER → roi_detections ────────────────────────────────
        # Only objects whose centre lies inside the ROI polygon go forward to the
        # DEFINE ZONE and COUNT LINE stages.
        #
        # When ROI is not yet defined:
        #   roi_detections = []  →  zone and counter are idle.
        #   all_detections still drives visualization, stabilizer, and BG removal
        #   so tracking IDs and bounding boxes are always visible on screen.
        #
        # This is the ONLY place where ROI membership is enforced.
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            roi_detections = self._filter_by_polygon(all_detections, self.roi_polygon)
            if _DEBUG:
                for d in roi_detections:
                    if d.track_id not in self._roi_logged_ids:
                        self._roi_logged_ids.add(d.track_id)
                        logger.debug(
                            "[ROI ENTRY] id=%d first entry frame=%d center=%s",
                            d.track_id, self._frame_index, d.center,
                        )
                logger.debug(
                    "[ROI FILTER OK] frame=%d all=%d roi=%d",
                    self._frame_index, len(all_detections), len(roi_detections),
                )
        else:
            roi_detections = []
            if _DEBUG and all_detections:
                logger.debug(
                    "[ROI FILTER] no ROI defined — %d object(s) tracked/visible "
                    "but NOT eligible for DEFINE ZONE or counting (frame=%d)",
                    len(all_detections), self._frame_index,
                )

        # ── STAGE 9: DEFINE ZONE ─────────────────────────────────────────────────
        # Collect (class_name, confidence) while an ROI object is inside the zone.
        # Finalise the class when the object exits (or disappears while inside).
        # Operates exclusively on roi_detections.
        self._update_define_zone_classes(roi_detections, raw_class_map)

        # countable_ids = track_ids that have a confirmed final class from DEFINE ZONE
        countable_ids = set(self._final_classes.keys())

        # ── STAGE 10: COUNT LINE ─────────────────────────────────────────────────
        # Increments only for objects in countable_ids that cross the count line
        # and have not already been counted.  Operates on roi_detections.
        prev_total = self.counter.total_count
        new_total = self.counter.update(roi_detections, countable_ids=countable_ids)

        if _DEBUG and self.counter.newly_counted_ids:
            for tid in self.counter.newly_counted_ids:
                cls_val = self._final_classes.get(tid, "?")
                logger.debug(
                    "[COUNT LINE CROSSED][COUNTER INCREMENTED] id=%d class=%s total=%d frame=%d",
                    tid, cls_val, new_total, self._frame_index,
                )

        # ── STAGE 11: STATISTICS & DATABASE (newly counted objects only) ─────────
        if new_total > prev_total and self.counter.newly_counted_ids:
            for track_id in self.counter.newly_counted_ids:
                if track_id not in self._final_classes:
                    continue
                latest_det = next(
                    (d for d in reversed(roi_detections) if d.track_id == track_id),
                    None,
                )
                if latest_det is None:
                    continue
                latest_det.cls_name = self._final_classes[track_id]
                self.stats_manager.update_from_detection(latest_det)
                self.db.log_event(latest_det, base_frame, save_snapshot=True)

        # ── STAGE 12: VISUALIZE ──────────────────────────────────────────────────
        # Draw on processed using ALL detections so every tracked object gets a
        # bounding box and track ID, regardless of ROI membership.
        # Confirmed class labels come from _final_classes and appear only after
        # the object has exited the DEFINE ZONE.
        self._draw_overlays(processed, all_detections)

        # Emit frames and stats to the UI
        self.frame_processed.emit(base_frame, processed, self.stats_manager.stats)

        # Timeline slider update for uploaded videos
        if self.video.is_file_source() and self.video.is_opened():
            current = self.video.get_frame_index()
            total = self.video.get_frame_count()
            self.position_updated.emit(current, total)


    def _point_in_polygon(self, poly: list[tuple[int, int]] | None, x: int, y: int) -> bool:
        if not poly or len(poly) < 3:
            return True  # no polygon defined -> accept all
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1
            ):
                inside = not inside
        return inside

    def _filter_by_polygon(self, detections, polygon):
        if not polygon or len(polygon) < 3:
            return list(detections)
        filtered = []
        for det in detections:
            cx, cy = det.center
            if self._point_in_polygon(polygon, cx, cy):
                filtered.append(det)
        return filtered

    def _build_roi_mask(self, shape_hw):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.roi_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _draw_overlays(self, frame, detections) -> None:
        # Draw ROI polygon if defined (green)
        if self.roi_polygon and len(self.roi_polygon) >= 2:
            for i in range(len(self.roi_polygon)):
                x1, y1 = self.roi_polygon[i]
                x2, y2 = self.roi_polygon[(i + 1) % len(self.roi_polygon)]
                # Green ROI polygon (BGR: 0,255,0)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Define Zone polygon if defined (dark blue)
        if self.define_zone_polygon and len(self.define_zone_polygon) >= 2:
            for i in range(len(self.define_zone_polygon)):
                x1, y1 = self.define_zone_polygon[i]
                x2, y2 = self.define_zone_polygon[(i + 1) % len(self.define_zone_polygon)]
                # Dark blue Define Zone polygon (BGR: 139,0,0)
                cv2.line(frame, (x1, y1), (x2, y2), (139, 0, 0), 2)

        # Draw counting line
        if self.counter.line_config:
            cfg = self.counter.line_config
            # Red counting line (BGR: 0,0,255)
            cv2.line(frame, (cfg.x1, cfg.y1), (cfg.x2, cfg.y2), (0, 0, 255), 2)

        # Draw detections
        for det in detections:
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            if det.track_id is not None:
                # Always show tracking ID
                id_text = f"ID: {det.track_id}"
                cv2.putText(
                    frame,
                    id_text,
                    (det.x1, max(0, det.y1 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Class label display rule (strict pipeline):
                # ONLY show class label when a FINAL CLASS has been confirmed.
                # Final class is only assigned AFTER the object exits the DEFINE ZONE.
                # While inside DEFINE ZONE or before entering it: show track ID only.
                # When no DEFINE ZONE is configured: _final_classes is always empty,
                # so nothing is shown (users must configure DEFINE ZONE first).
                if det.track_id in self._final_classes:
                    class_text = self._final_classes[det.track_id]
                    cv2.putText(
                        frame,
                        class_text,
                        (det.x1, max(0, det.y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),  # Bright yellow — confirmed final class
                        2,
                        cv2.LINE_AA,
                    )

    @staticmethod
    def _compute_iou(a: DetectionResult, b: DetectionResult) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _suppress_overlaps(
        self, detections: list["DetectionResult"], iou_threshold: float = 0.5
    ) -> list["DetectionResult"]:
        """
        Ensure that a single physical object is not classified as multiple classes
        by applying an additional NMS-like suppression across all classes.
        """
        if not detections:
            return []

        # Sort by confidence descending
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[DetectionResult] = []

        for det in dets:
            should_keep = True
            for k in kept:
                if self._compute_iou(det, k) > iou_threshold:
                    # Overlaps a stronger detection -> discard
                    should_keep = False
                    break
            if should_keep:
                kept.append(det)

        return kept

    def _update_define_zone_classes(
        self,
        detections,
        raw_class_map: Dict[int, tuple[str, float]] | None = None,
    ) -> None:
        """
        Aggregate class predictions for each track while it passes through the
        Define Zone polygon, and assign a stable final class per track based on
        average confidence. Once assigned, the class remains fixed.

        raw_class_map: optional dict of track_id -> (raw_cls_name, raw_confidence)
            extracted directly from YOLO output BEFORE the stabilizer overwrites
            cls_name. When provided, we prefer this over det.cls_name so the zone
            history reflects the actual per-frame model output.
        """
        if not self.define_zone_polygon or len(self.define_zone_polygon) < 3:
            # No Define Zone configured: do nothing (objects count without zone)
            return

        # Update lifetime for all tracks seen in this frame
        for det in detections:
            if det.track_id is None:
                continue
            track_id = det.track_id
            self._track_total_frames[track_id] = (
                self._track_total_frames.get(track_id, 0) + 1
            )

        current_ids = set()
        for det in detections:
            if det.track_id is None:
                continue

            track_id = det.track_id
            current_ids.add(track_id)

            cx, cy = det.center
            inside = self._point_in_polygon(self.define_zone_polygon, cx, cy)
            was_inside = self._define_zone_inside.get(track_id, False)

            # Skip further accumulation if final class already decided for this track
            if track_id in self._final_classes:
                # Update inside flag and continue — no further work needed
                self._define_zone_inside[track_id] = inside
                continue

            # ----- COLLECT predictions while object is inside DEFINE ZONE -----
            # DO NOT display or assign any class here.
            # Only accumulate (class_name, confidence) into history.
            if inside:
                # Use raw YOLO output (before stabilizer) so the history reflects
                # the actual per-frame model prediction, not the smoothed class.
                if raw_class_map and track_id in raw_class_map:
                    cls_name, conf = raw_class_map[track_id]
                else:
                    cls_name = det.cls_name
                    conf = float(det.confidence)

                history = self._id_class_history.setdefault(track_id, [])
                history.append((cls_name, conf))

                # Cap history size to prevent memory growth (keep most recent 20 frames)
                if len(history) > self._max_zone_history:
                    self._id_class_history[track_id] = history[-self._max_zone_history:]
                    history = self._id_class_history[track_id]

                if _DEBUG:
                    if len(history) == 1:
                        logger.debug(
                            "DEFINE zone: id=%d ENTERED zone at frame %d — "
                            "collecting predictions (first: %s %.2f)",
                            track_id, self._frame_index, cls_name, conf,
                        )
                    else:
                        logger.debug(
                            "DEFINE zone: collecting id=%d frame=%d class=%s conf=%.2f n=%d",
                            track_id, self._frame_index, cls_name, conf, len(history),
                        )

            # ----- FINALIZE when object transitions from inside → outside zone -----
            # This is the ONLY place final class is assigned.
            if was_inside and not inside:
                logger.debug(
                    "DEFINE zone: id=%d LEFT zone at frame %d — finalizing class",
                    track_id, self._frame_index,
                ) if _DEBUG else None
                self._finalize_class_from_history(track_id, reason="exited")

            # Update inside-flag for next frame
            self._define_zone_inside[track_id] = inside

        # ----- Handle tracks that disappeared while inside Define Zone -----
        # (e.g., tracking lost, object left frame without exiting zone cleanly)
        for track_id, was_inside in list(self._define_zone_inside.items()):
            if not was_inside or track_id in current_ids:
                continue
            if track_id in self._final_classes:
                continue
            # Track disappeared while inside zone — finalize its class from collected history
            if _DEBUG:
                logger.debug(
                    "DEFINE zone: id=%d DISAPPEARED inside zone (frame %d) — finalizing class",
                    track_id, self._frame_index,
                )
            self._finalize_class_from_history(track_id, reason="disappeared")

    def _finalize_class_from_history(self, track_id: int, reason: str = "exited") -> None:
        """
        Compute the final class for a track_id from its accumulated zone history
        using average confidence per class. Stores result in self._final_classes.
        """
        history = self._id_class_history.get(track_id, [])
        total_obs = len(history)

        if total_obs < self._min_define_zone_observations:
            if _DEBUG:
                logger.warning(
                    "DEFINE zone: id=%d %s but not enough observations (%d < %d), skip",
                    track_id, reason, total_obs, self._min_define_zone_observations,
                )
            return

        # Compute average confidence per class
        class_sums: Dict[str, float] = {}
        class_counts: Dict[str, int] = {}
        for name, conf in history:
            class_sums[name] = class_sums.get(name, 0.0) + conf
            class_counts[name] = class_counts.get(name, 0) + 1

        best_cls: str | None = None
        best_avg_conf: float = -1.0
        for name, total_conf in class_sums.items():
            avg_conf = float(total_conf / max(1, class_counts[name]))
            if _DEBUG:
                logger.debug(
                    "DEFINE zone: id=%d class=%s avg_conf=%.3f (n=%d)",
                    track_id, name, avg_conf, class_counts[name],
                )
            if avg_conf > best_avg_conf:
                best_avg_conf = avg_conf
                best_cls = name

        if best_cls is not None:
            self._final_classes[track_id] = best_cls
            if _DEBUG:
                logger.debug(
                    "DEFINE zone: id=%d %s → final class='%s' (avg_conf=%.3f, %d obs); "
                    "track_id ADDED to countable_ids",
                    track_id, reason, best_cls, best_avg_conf, total_obs,
                )
        elif _DEBUG:
            logger.warning(
                "DEFINE zone: id=%d %s but no class could be selected (history len=%d)",
                track_id, reason, total_obs,
            )


def load_yaml_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_and_run() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    if _DEBUG:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    cfg = load_yaml_config()

    # UI configuration
    ui_cfg_raw = cfg.get("ui", {})
    ui_cfg = UIConfig(
        defect_alarm_threshold=float(ui_cfg_raw.get("defect_alarm_threshold", 10.0))
    )
    window: MainWindow = create_app_window()
    window.ui_config = ui_cfg

    # Video / camera configuration
    video_cfg = cfg.get("video", {})
    capture_cfg = CaptureConfig(
        source=video_cfg.get("source", 0),
        width=video_cfg.get("width"),
        height=video_cfg.get("height"),
        fps=video_cfg.get("fps"),
    )

    # Preprocessing configuration
    pre_cfg_raw = cfg.get("preprocessing", {})
    preprocess_cfg = PreprocessConfig(
        resize_width=int(pre_cfg_raw.get("resize_width", 640)),
        resize_height=int(pre_cfg_raw.get("resize_height", 640)),
        use_gaussian_blur=bool(pre_cfg_raw.get("use_gaussian_blur", True)),
        use_median_blur=bool(pre_cfg_raw.get("use_median_blur", False)),
        blur_kernel_size=int(pre_cfg_raw.get("blur_kernel_size", 5)),
        use_clahe=bool(pre_cfg_raw.get("use_clahe", True)),
        clahe_clip_limit=float(pre_cfg_raw.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=tuple(pre_cfg_raw.get("clahe_tile_grid_size", [8, 8])),
        brightness=float(pre_cfg_raw.get("brightness", 1.0)),
        contrast=float(pre_cfg_raw.get("contrast", 0.0)),
        roi=None,
    )

    # Detection / model configuration
    model_cfg = cfg.get("model", {})
    det_cfg = cfg.get("detection", {})
    model_path = model_cfg.get("path", "best2.pt")
    model_device = model_cfg.get("device", "")
    det_conf = float(det_cfg.get("confidence_threshold", 0.4))
    det_iou = float(det_cfg.get("iou_threshold", 0.5))

    # Counting configuration
    counting_cfg = cfg.get("counting", {})
    mode_str = counting_cfg.get("mode", "line").lower()
    if mode_str == "zone":
        counting_mode = CountingMode.ZONE
    elif mode_str == "direction":
        counting_mode = CountingMode.DIRECTION
    else:
        counting_mode = CountingMode.LINE_CROSSING

    # Counting line will be defined interactively by the user,
    # so do not create a default line here.
    line_config = None

    zone_raw = counting_cfg.get("zone", {})
    zone_config = ZoneCountingConfig(
        x1=int(zone_raw.get("x1", 0)),
        y1=int(zone_raw.get("y1", 0)),
        x2=int(zone_raw.get("x2", preprocess_cfg.resize_width)),
        y2=int(zone_raw.get("y2", preprocess_cfg.resize_height)),
    )

    dir_x_ref = int(
        counting_cfg.get("direction_x_ref", preprocess_cfg.resize_width // 2)
    )
    direction_config = DirectionCountingConfig(x_ref=dir_x_ref)

    # Database configuration
    db_cfg_raw = cfg.get("database", {})
    db_cfg = DatabaseConfig(
        backend=db_cfg_raw.get("backend", "sqlite"),
        sqlite_path=db_cfg_raw.get("sqlite_path", "potato_qc.db"),
        snapshot_dir=db_cfg_raw.get("snapshot_dir", "snapshots"),
    )

    processor = FrameProcessor(
        capture_cfg,
        preprocess_cfg,
        db_cfg,
        model_path=model_path,
        model_device=model_device,
        det_conf=det_conf,
        det_iou=det_iou,
        counting_mode=counting_mode,
        line_config=line_config,
        zone_config=zone_config,
        direction_config=direction_config,
    )

    # Connect UI controls
    def on_start() -> None:
        # Start a fresh detection session each time Start is pressed
        processor.reset_session()
        processor.start()

    window.start_requested.connect(on_start)

    def on_stop() -> None:
        # UI "Stop" is treated as a pause to allow Resume from current frame.
        processor.pause()

    window.stop_requested.connect(on_stop)
    window.auto_calibrate_requested.connect(
        lambda: processor.preprocessor.auto_calibrate(
            processor.video.read()[1] if processor.video.is_opened() else None
        )
    )

    def on_brightness(alpha: float) -> None:
        processor.preprocessor.config.brightness = alpha

    def on_contrast(beta: float) -> None:
        processor.preprocessor.config.contrast = beta

    def on_detection_conf(conf: float) -> None:
        processor.detector.confidence_threshold = conf

    window.brightness_changed.connect(on_brightness)
    window.contrast_changed.connect(on_contrast)
    window.detection_conf_changed.connect(on_detection_conf)

    # ROI and counting line selection
    def on_roi_polygon(points) -> None:
        # points: list of (x, y) in image coordinates
        processor.set_roi_polygon(points)

    def on_line_segment(seg) -> None:
        # seg: ((x1, y1), (x2, y2))
        (x1, y1), (x2, y2) = seg
        processor.counter.line_config = LineCountingConfig(x1=x1, y1=y1, x2=x2, y2=y2)

    window.roi_selected.connect(on_roi_polygon)
    window.count_line_selected.connect(on_line_segment)

    # Define Zone polygon selection
    def on_define_zone_polygon(points) -> None:
        # points: list of (x, y) in image coordinates
        processor.set_define_zone_polygon(points)

    window.define_zone_selected.connect(on_define_zone_polygon)

    # Upload video handling
    def on_video_file(path: str) -> None:
        processor.reset_session()
        processor.set_video_source(path)
        # Immediately show a preview of the first frame for better UX
        processor.preview_first_frame()

    window.video_file_selected.connect(on_video_file)

    # Mode switching: LiveCam <-> Upload
    camera_source = capture_cfg.source

    def on_mode_changed(mode: str) -> None:
        if mode == "live":
            processor.reset_session()
            processor.set_video_source(camera_source)
        elif mode == "upload":
            # In upload mode we wait for the user to pick a file.
            processor.stop()

    window.mode_changed.connect(on_mode_changed)

    # Background removal toggling
    def on_background(enabled: bool) -> None:
        processor.remove_background = enabled

    window.background_toggled.connect(on_background)

    # Connect processed frames and stats to UI
    processor.frame_processed.connect(
        lambda orig, proc, stats: window.update_frames(orig, proc)
    )
    processor.frame_processed.connect(
        lambda orig, proc, stats: window.update_stats(stats)
    )

    # Timeline updates for uploaded videos
    processor.position_updated.connect(window.set_timeline_position)

    # Playback state updates for changing Stop/Resume/Restart label
    processor.state_changed.connect(window.on_processor_state_changed)

    # Seeking in uploaded video from timeline slider
    def on_seek(frame_index: int) -> None:
        processor.video.seek_to_frame(frame_index)

    window.seek_requested.connect(on_seek)

    # Resume from paused state
    window.resume_requested.connect(processor.resume)

    # Timer for real-time processing
    timer = QtCore.QTimer()
    timer.setInterval(1)  # as fast as possible, ~30+ FPS if hardware allows
    timer.timeout.connect(processor.process_next_frame)
    timer.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    build_and_run()

