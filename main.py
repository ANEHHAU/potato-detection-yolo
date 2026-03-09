from __future__ import annotations

import os
import sys
from typing import Optional, Dict, List

import cv2
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

        # Tracking lifetime for filtering out very short-lived false positives
        self._track_total_frames: Dict[int, int] = {}
        self._define_zone_inside: Dict[int, bool] = {}
        self._min_frames_for_class: int = 1

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
        # Preprocessing (resize, denoise, contrast, ROI)
        base_frame = self.preprocessor.apply(frame)

        # If ROI is defined, zero out everything outside ROI before detection so
        # detections and tracking IDs are created only inside ROI.
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            roi_mask = self._build_roi_mask(base_frame.shape[:2])
            detect_frame = cv2.bitwise_and(base_frame, base_frame, mask=roi_mask)
        else:
            detect_frame = base_frame

        # Detection + tracking (ByteTrack via Ultralytics) on ROI-restricted frame
        detections = self.detector.track(detect_frame)

        # Restrict detections to ROI region (safety filter)
        detections = self._filter_by_polygon(detections, self.roi_polygon)

        # Suppress overlapping multi-class detections (keep highest confidence only)
        detections = self._suppress_overlaps(detections, iou_threshold=0.5)

        # Classification stabilization (temporal smoothing)
        detections = self.stabilizer.update(detections)

        # Update Define Zone-based final class assignments
        self._update_define_zone_classes(detections)

        # Counting logic: only count objects that have confirmed class from Define Zone
        countable_ids = set(self._final_classes.keys())
        prev_total = self.counter.total_count
        new_total = self.counter.update(detections, countable_ids=countable_ids)

        # Update statistics and DB only for newly counted track IDs,
        # and only once a final class from Define Zone is available.
        if new_total > prev_total and self.counter.newly_counted_ids:
            for track_id in self.counter.newly_counted_ids:
                if track_id not in self._final_classes:
                    # Skip objects that never passed through Define Zone
                    continue
                # Find the latest detection for this track in current frame
                latest_det = None
                for det in reversed(detections):
                    if det.track_id == track_id:
                        latest_det = det
                        break
                if latest_det is None:
                    continue
                # Use the confirmed final class for statistics and logging
                latest_det.cls_name = self._final_classes[track_id]
                self.stats_manager.update_from_detection(latest_det)
                self.db.log_event(latest_det, base_frame, save_snapshot=True)

        # Build visualization frame with background removed (when enabled)
        processed = base_frame.copy()
        if self.remove_background and detections:
            # Keep original potato pixels unchanged; set only background to white.
            h, w = base_frame.shape[:2]
            potato_mask = np.zeros((h, w), dtype=np.uint8)
            for det in detections:
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                potato_mask[y1:y2, x1:x2] = 255
            processed[potato_mask == 0] = [255, 255, 255]

        # Draw overlays for visualization (ROI, Define Zone, line, boxes, etc.)
        self._draw_overlays(processed, detections)

        # Emit frames (original + processed) and stats for UI
        self.frame_processed.emit(base_frame, processed, self.stats_manager.stats)

        # Emit playback position for uploaded videos (timeline support)
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

                # Show class label only after Define Zone has confirmed it
                if det.track_id in self._final_classes:
                    class_text = f"Class: {self._final_classes[det.track_id]}"
                    cv2.putText(
                        frame,
                        class_text,
                        (det.x1, max(0, det.y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
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

    def _update_define_zone_classes(self, detections) -> None:
        """
        Aggregate class predictions for each track while it passes through the
        Define Zone polygon, and assign a stable final class per track based on
        average confidence. Once assigned, the class remains fixed.
        """
        if not self.define_zone_polygon or len(self.define_zone_polygon) < 3:
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
            if track_id in self._final_classes:
                # Class already fixed – do not change it
                continue

            cx, cy = det.center
            inside = self._point_in_polygon(self.define_zone_polygon, cx, cy)
            was_inside = self._define_zone_inside.get(track_id, False)

            # Accumulate history while inside Define Zone
            if inside:
                cls_name = det.cls_name
                conf = float(det.confidence)
                history = self._id_class_history.setdefault(track_id, [])
                history.append((cls_name, conf))

            # Detect transition: leaving Define Zone
            if was_inside and not inside:
                history = self._id_class_history.get(track_id, [])
                total_obs = len(history)

                if total_obs >= self._min_define_zone_observations:
                    # Group by class and compute average confidence
                    class_sums: Dict[str, float] = {}
                    class_counts: Dict[str, int] = {}
                    for name, conf in history:
                        class_sums[name] = class_sums.get(name, 0.0) + conf
                        class_counts[name] = class_counts.get(name, 0) + 1

                    best_cls = None
                    best_conf = -1.0
                    for name, total_conf in class_sums.items():
                        avg_conf = float(total_conf / max(1, class_counts[name]))
                        if avg_conf > best_conf:
                            best_conf = avg_conf
                            best_cls = name

                    if best_cls is not None:
                        self._final_classes[track_id] = best_cls

            # Update inside-flag for next frame
            self._define_zone_inside[track_id] = inside

        # Handle tracks that were inside the Define Zone but disappeared
        # from the current frame (e.g., left the frame or became undetected)
        for track_id, was_inside in list(self._define_zone_inside.items()):
            if not was_inside or track_id in current_ids:
                continue
            if track_id in self._final_classes:
                continue

            history = self._id_class_history.get(track_id, [])
            total_obs = len(history)

            if total_obs >= self._min_define_zone_observations:
                class_sums: Dict[str, float] = {}
                class_counts: Dict[str, int] = {}
                for name, conf in history:
                    class_sums[name] = class_sums.get(name, 0.0) + conf
                    class_counts[name] = class_counts.get(name, 0) + 1

                best_cls = None
                best_conf = -1.0
                for name, total_conf in class_sums.items():
                    avg_conf = float(total_conf / max(1, class_counts[name]))
                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_cls = name

                if best_cls is not None:
                    self._final_classes[track_id] = best_cls


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

