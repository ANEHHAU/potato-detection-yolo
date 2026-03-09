from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from counter import CountingMode
from detector import DetectionResult
from preprocessing import PreprocessConfig
from statistics import PotatoStats


@dataclass
class UIConfig:
    defect_alarm_threshold: float = 10.0  # percent


class VideoWidget(QtWidgets.QLabel):
    """Widget to display OpenCV frames and support ROI / line selection."""

    roi_selected = QtCore.pyqtSignal(object)  # list of (x, y) image points
    line_selected = QtCore.pyqtSignal(object)  # ((x1, y1), (x2, y2))
    define_zone_selected = QtCore.pyqtSignal(object)  # list of (x, y) image points

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QtGui.QPixmap | None = None
        self._mode: str = "none"  # "none", "roi", "line", "zone"
        self._roi_points: list[QtCore.QPoint] = []
        self._line_points: list[QtCore.QPoint] = []
        self._zone_points: list[QtCore.QPoint] = []
        self._placeholder_text: str = (
            "No video source detected.\nPlease upload a video or start the camera."
        )

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder_text = text
        self.update()

    def clear_frame(self) -> None:
        """Clear current frame so placeholder text becomes visible again."""
        self._pixmap = None
        self.update()

    def start_roi_selection(self) -> None:
        self._mode = "roi"
        self._roi_points.clear()
        self.update()

    def finish_roi_selection(self) -> None:
        if self._mode != "roi" or self._pixmap is None or len(self._roi_points) < 3:
            self._mode = "none"
            self.update()
            return

        pix_size = self._pixmap.size()
        widget_size = self.size()
        scale_x = pix_size.width() / max(1, widget_size.width())
        scale_y = pix_size.height() / max(1, widget_size.height())

        pts_img: list[tuple[int, int]] = []
        for p in self._roi_points:
            x = int(p.x() * scale_x)
            y = int(p.y() * scale_y)
            pts_img.append((x, y))

        self.roi_selected.emit(pts_img)
        self._mode = "none"
        self.update()

    def start_zone_selection(self) -> None:
        """Begin Define Zone polygon selection."""
        self._mode = "zone"
        self._zone_points.clear()
        self.update()

    def finish_zone_selection(self) -> None:
        """Finalize Define Zone polygon and emit in image coordinates."""
        if self._mode != "zone" or self._pixmap is None or len(self._zone_points) < 3:
            self._mode = "none"
            self.update()
            return

        pix_size = self._pixmap.size()
        widget_size = self.size()
        scale_x = pix_size.width() / max(1, widget_size.width())
        scale_y = pix_size.height() / max(1, widget_size.height())

        pts_img: list[tuple[int, int]] = []
        for p in self._zone_points:
            x = int(p.x() * scale_x)
            y = int(p.y() * scale_y)
            pts_img.append((x, y))

        self.define_zone_selected.emit(pts_img)
        self._mode = "none"
        self.update()

    def start_line_selection(self) -> None:
        self._mode = "line"
        self._line_points.clear()
        self.update()

    def show_frame(self, frame) -> None:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        self._pixmap = QtGui.QPixmap.fromImage(q_image)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)

        if self._pixmap is not None:
            painter.drawPixmap(self.rect(), self._pixmap)
        else:
            # No video source yet -> show centered placeholder text
            painter.setPen(QtGui.QColor(200, 200, 200))
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            rect = self.rect()
            painter.drawText(
                rect,
                QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                self._placeholder_text,
            )

        # Draw ROI polygon (yellow)
        if self._roi_points:
            roi_pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2)  # yellow
            painter.setPen(roi_pen)
            for i in range(len(self._roi_points) - 1):
                painter.drawLine(self._roi_points[i], self._roi_points[i + 1])
            # Close polygon visually when at least 3 points
            if len(self._roi_points) >= 3:
                painter.drawLine(self._roi_points[-1], self._roi_points[0])

        # Draw Define Zone polygon (dark blue)
        if self._zone_points:
            zone_pen = QtGui.QPen(QtGui.QColor(0, 0, 139), 2)  # dark blue
            painter.setPen(zone_pen)
            for i in range(len(self._zone_points) - 1):
                painter.drawLine(self._zone_points[i], self._zone_points[i + 1])
            if len(self._zone_points) >= 3:
                painter.drawLine(self._zone_points[-1], self._zone_points[0])

        # Draw counting line while selecting (red)
        if self._line_points:
            line_pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 2)  # red
            painter.setPen(line_pen)
            if len(self._line_points) == 1:
                # Single point: draw a small marker
                p = self._line_points[0]
                painter.drawEllipse(p, 3, 3)
            elif len(self._line_points) == 2:
                painter.drawLine(self._line_points[0], self._line_points[1])

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            if self._mode == "roi":
                self._roi_points.append(event.pos())
                self.update()
            elif self._mode == "zone":
                self._zone_points.append(event.pos())
                self.update()
            elif self._mode == "line":
                self._line_points.append(event.pos())
                if len(self._line_points) == 2 and self._pixmap is not None:
                    # Finalize line selection
                    pix_size = self._pixmap.size()
                    widget_size = self.size()
                    scale_x = pix_size.width() / max(1, widget_size.width())
                    scale_y = pix_size.height() / max(1, widget_size.height())

                    p1, p2 = self._line_points
                    x1, y1 = int(p1.x() * scale_x), int(p1.y() * scale_y)
                    x2, y2 = int(p2.x() * scale_x), int(p2.y() * scale_y)
                    self.line_selected.emit(((x1, y1), (x2, y2)))
                    self._mode = "none"
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # No dragging-based behavior; rely on clicks
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # No special behavior on release for ROI; finalized via End Set ROI
        super().mouseReleaseEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    resume_requested = QtCore.pyqtSignal()
    auto_calibrate_requested = QtCore.pyqtSignal()

    brightness_changed = QtCore.pyqtSignal(float)
    contrast_changed = QtCore.pyqtSignal(float)
    detection_conf_changed = QtCore.pyqtSignal(float)
    video_file_selected = QtCore.pyqtSignal(str)
    mode_changed = QtCore.pyqtSignal(str)  # "live" or "upload"
    background_toggled = QtCore.pyqtSignal(bool)

    roi_selected = QtCore.pyqtSignal(object)  # list of (x, y)
    count_line_selected = QtCore.pyqtSignal(object)  # ((x1, y1), (x2, y2))
    define_zone_selected = QtCore.pyqtSignal(object)  # list of (x, y)
    seek_requested = QtCore.pyqtSignal(int)  # frame index for uploaded video

    def __init__(self, ui_config: Optional[UIConfig] = None) -> None:
        super().__init__()

        self.ui_config = ui_config or UIConfig()
        self.setWindowTitle("Potato Quality Control Dashboard")

        self._upload_mode: bool = False
        self._has_video_file: bool = False
        self._video_state: str = "stopped"  # "stopped", "playing", "paused", "ended"

        self._build_layout()

        self.alarm_active = False

    # ----- UI Construction -----

    def _build_layout(self) -> None:
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)

        # Video feed (top: original, bottom: processed / interactive)
        # Both widgets use identical dimensions (800x450)
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(800, 450)
        self.video_widget.setStyleSheet("background-color: black;")

        self.original_video_widget = VideoWidget()
        self.original_video_widget.setMinimumSize(800, 450)
        self.original_video_widget.setStyleSheet("background-color: black;")
        self.original_video_widget.set_placeholder_text("Original video")
        self.original_video_widget.hide()

        main_layout.addWidget(self.original_video_widget)
        main_layout.addWidget(self.video_widget)

        # Ensure both video widgets keep the same size when visible
        main_layout.setStretchFactor(self.original_video_widget, 1)
        main_layout.setStretchFactor(self.video_widget, 1)

        # Playback controls: timeline for uploaded videos, LIVE indicator for camera
        playback_layout = QtWidgets.QHBoxLayout()
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.sliderReleased.connect(self._on_timeline_released)

        self.live_label = QtWidgets.QLabel("● LIVE")
        live_font = self.live_label.font()
        live_font.setBold(True)
        self.live_label.setFont(live_font)
        self.live_label.setStyleSheet("color: red;")

        playback_layout.addWidget(self.timeline_slider)
        playback_layout.addWidget(self.live_label)
        main_layout.addLayout(playback_layout)

        # Statistics display
        stats_layout = QtWidgets.QHBoxLayout()
        self.total_label = QtWidgets.QLabel("Total: 0")
        self.good_label = QtWidgets.QLabel("Good: 0")
        self.defected_label = QtWidgets.QLabel("Defected: 0")
        self.damaged_label = QtWidgets.QLabel("Damaged: 0")
        self.diseased_label = QtWidgets.QLabel("Diseased-fungal: 0")
        self.sprouted_label = QtWidgets.QLabel("Sprouted: 0")
        self.defect_rate_label = QtWidgets.QLabel("Defect rate: 0.0%")

        for lbl in [
            self.total_label,
            self.good_label,
            self.defected_label,
            self.damaged_label,
            self.diseased_label,
            self.sprouted_label,
            self.defect_rate_label,
        ]:
            stats_layout.addWidget(lbl)

        main_layout.addLayout(stats_layout)

        # Control buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.auto_btn = QtWidgets.QPushButton("Auto Calibration")
        self.mode_btn = QtWidgets.QPushButton("Switch to Upload")
        self.bg_btn = QtWidgets.QPushButton("Remove Background")
        self.bg_btn.setCheckable(True)
        self.upload_btn = QtWidgets.QPushButton("Upload Video")
        self.roi_btn = QtWidgets.QPushButton("Set ROI")
        self.zone_btn = QtWidgets.QPushButton("Set Define Zone")
        self.line_btn = QtWidgets.QPushButton("Set Counting Line")
        self.line_btn.setCheckable(True)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.auto_btn)
        btn_layout.addWidget(self.mode_btn)
        btn_layout.addWidget(self.bg_btn)
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.roi_btn)
        btn_layout.addWidget(self.zone_btn)
        btn_layout.addWidget(self.line_btn)

        main_layout.addLayout(btn_layout)

        # Calibration and detection sliders
        sliders_layout = QtWidgets.QFormLayout()

        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setMinimum(50)   # 0.5
        self.brightness_slider.setMaximum(150)  # 1.5
        self.brightness_slider.setValue(100)

        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setMinimum(-50)  # -50
        self.contrast_slider.setMaximum(50)   # +50
        self.contrast_slider.setValue(0)

        self.detection_conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.detection_conf_slider.setMinimum(10)  # 0.1
        self.detection_conf_slider.setMaximum(90)  # 0.9
        self.detection_conf_slider.setValue(40)

        sliders_layout.addRow("Brightness", self.brightness_slider)
        sliders_layout.addRow("Contrast", self.contrast_slider)
        sliders_layout.addRow("Detection confidence", self.detection_conf_slider)

        main_layout.addLayout(sliders_layout)

        scroll_area.setWidget(central)
        self.setCentralWidget(scroll_area)

        # Connect signals
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.auto_btn.clicked.connect(self.auto_calibrate_requested.emit)
        self.mode_btn.clicked.connect(self._on_toggle_mode)
        self.bg_btn.clicked.connect(self._on_toggle_background)
        self.upload_btn.clicked.connect(self._on_upload_video)
        self.roi_btn.clicked.connect(self._on_toggle_roi_mode)
        self.zone_btn.clicked.connect(self._on_toggle_zone_mode)
        self.line_btn.clicked.connect(self._on_set_line_mode)

        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.detection_conf_slider.valueChanged.connect(self._on_detection_conf_changed)

        # Forward selection signals from video widget
        self.video_widget.roi_selected.connect(self.roi_selected)
        self.video_widget.define_zone_selected.connect(self.define_zone_selected)
        self.video_widget.line_selected.connect(self._on_line_finished)
        self.video_widget.line_selected.connect(self.count_line_selected)

        # Start in live camera mode: hide upload button
        self.upload_btn.hide()
        self._update_playback_controls()

    # ----- External update API -----

    def update_frames(self, original_frame, processed_frame) -> None:
        """
        Update video widgets. When background removal is enabled, both original
        and processed frames are shown. Otherwise only the bottom widget is used.
        """
        if original_frame is not None:
            self.original_video_widget.show_frame(original_frame)
        if processed_frame is not None:
            self.video_widget.show_frame(processed_frame)

    def update_stats(self, stats: PotatoStats) -> None:
        self.total_label.setText(f"Total: {stats.total}")
        self.good_label.setText(f"Good: {stats.good}")
        self.defected_label.setText(f"Defected: {stats.defected}")
        self.damaged_label.setText(f"Damaged: {stats.damaged}")
        self.diseased_label.setText(f"Diseased-fungal: {stats.diseased_fungal}")
        self.sprouted_label.setText(f"Sprouted: {stats.sprouted}")
        self.defect_rate_label.setText(f"Defect rate: {stats.defect_rate:.1f}%")

        self._check_alarm(stats)

    # ----- Playback / state helpers -----

    def on_processor_state_changed(self, state: str) -> None:
        """
        Called by the processing pipeline when playback state changes.
        Used to update the Stop/Resume/Restart button label.
        """
        self._video_state = state
        self._update_stop_button_label()

    def _check_alarm(self, stats: PotatoStats) -> None:
        if stats.defect_rate > self.ui_config.defect_alarm_threshold:
            if not self.alarm_active:
                self.alarm_active = True
                self.statusBar().showMessage("ALARM: Defect rate exceeded threshold!", 5000)
                # Placeholder for real factory integration (PLC, sound, etc.)
        else:
            if self.alarm_active:
                self.alarm_active = False
                self.statusBar().clearMessage()

    # ----- Slider handlers -----

    def _on_brightness_changed(self, value: int) -> None:
        # Map [50,150] -> [0.5,1.5]
        alpha = value / 100.0
        self.brightness_changed.emit(alpha)

    def _on_contrast_changed(self, value: int) -> None:
        # Map [-50,50] -> [-50,50] directly as beta
        self.contrast_changed.emit(float(value))

    def _on_detection_conf_changed(self, value: int) -> None:
        # Map [10,90] -> [0.1,0.9]
        conf = value / 100.0
        self.detection_conf_changed.emit(conf)

    def _on_upload_video(self) -> None:
        filters = "Video Files (*.mp4 *.avi *.mov)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video File", "", filters
        )
        if path:
            self.video_file_selected.emit(path)
            self._has_video_file = True
            self._update_playback_controls()

    def _on_toggle_mode(self) -> None:
        # Toggle between LiveCam and Upload modes
        self._upload_mode = not self._upload_mode
        if self._upload_mode:
            self.mode_btn.setText("Switch to LiveCam")
            self.upload_btn.show()
            self.mode_changed.emit("upload")
        else:
            self.mode_btn.setText("Switch to Upload")
            self.upload_btn.hide()
            self.mode_changed.emit("live")
        # Clear frames so placeholder is visible when no active source
        self.video_widget.clear_frame()
        self.original_video_widget.clear_frame()
        self._update_playback_controls()
        self._video_state = "stopped"
        self._update_stop_button_label()

    def _on_toggle_background(self) -> None:
        enabled = self.bg_btn.isChecked()
        self.background_toggled.emit(enabled)
        # Show / hide original panel when background removal is active
        if enabled:
            self.original_video_widget.show()
        else:
            self.original_video_widget.hide()

    def _on_toggle_roi_mode(self) -> None:
        if self.video_widget._mode != "roi":
            self.roi_btn.setText("End Set ROI")
            self.video_widget.start_roi_selection()
        else:
            self.roi_btn.setText("Set ROI")
            self.video_widget.finish_roi_selection()

    def _on_toggle_zone_mode(self) -> None:
        if self.video_widget._mode != "zone":
            self.zone_btn.setText("End Define Zone")
            self.video_widget.start_zone_selection()
        else:
            self.zone_btn.setText("Set Define Zone")
            self.video_widget.finish_zone_selection()

    def _on_set_line_mode(self) -> None:
        # Enter line creation mode and keep button visually active
        self.line_btn.setChecked(True)
        self.video_widget.start_line_selection()

    def _on_line_finished(self, seg) -> None:
        # Line has been fully defined (two points); exit active state
        self.line_btn.setChecked(False)

    def _update_playback_controls(self) -> None:
        """
        Show LIVE indicator in camera mode, and timeline slider in upload mode
        (only once a video file is selected).
        """
        if self._upload_mode and self._has_video_file:
            self.timeline_slider.show()
            self.timeline_slider.setEnabled(True)
            self.live_label.hide()
        else:
            self.timeline_slider.hide()
            self.timeline_slider.setEnabled(False)
            self.live_label.show()

    def _update_stop_button_label(self) -> None:
        """
        Update the Stop button label based on current playback state and mode.
        """
        # Only uploaded video uses Resume/Restart semantics; live camera keeps "Stop".
        if self._upload_mode and self._has_video_file:
            if self._video_state == "paused":
                self.stop_btn.setText("Resume")
            elif self._video_state == "ended":
                self.stop_btn.setText("Restart")
            else:
                self.stop_btn.setText("Stop")
        else:
            self.stop_btn.setText("Stop")

    def set_timeline_position(self, current: int, total: int) -> None:
        if total <= 0:
            return
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(total - 1)
        self.timeline_slider.setValue(max(0, min(current, total - 1)))
        self.timeline_slider.blockSignals(False)

    def _on_timeline_released(self) -> None:
        if not self.timeline_slider.isEnabled():
            return
        frame_idx = self.timeline_slider.value()
        self.seek_requested.emit(frame_idx)

    def _on_stop_clicked(self) -> None:
        """
        Context-aware handler for the Stop button:
        - When playing: pause (Stop)
        - When paused (upload mode): resume
        - When ended (upload mode): restart from beginning (same as Start)
        """
        if self._upload_mode and self._has_video_file:
            if self._video_state == "playing":
                self.stop_requested.emit()
            elif self._video_state == "paused":
                self.resume_requested.emit()
            elif self._video_state == "ended":
                # Behave like Start from the beginning
                self.start_requested.emit()
            else:
                # Fallback: treat as stop
                self.stop_requested.emit()
        else:
            # Live camera or no file: simple stop/pause
            if self._video_state == "playing":
                self.stop_requested.emit()
            elif self._video_state == "paused":
                self.resume_requested.emit()
            else:
                self.stop_requested.emit()


def create_app_window() -> MainWindow:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1024, 720)
    window.show()
    return window

