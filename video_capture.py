from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import os


@dataclass
class CaptureConfig:
    """Configuration for video capture."""

    source: Union[int, str] = 0  # 0 = default camera, "rtsp://...", "file.mp4"
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None


class VideoCapture:
    """
    Unified wrapper for:
    - USB camera (index: 0, 1, ...)
    - IP camera (RTSP/HTTP URL)
    - Video file (mp4, avi, ...)
    """

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_file: bool = isinstance(config.source, str) and os.path.isfile(
            config.source
        )

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.config.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.config.source}")

        if self.config.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def read(self):
        if self.cap is None:
            raise RuntimeError("Capture device not opened. Call open() first.")
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # --- Helpers for video files (timeline support) ---

    def is_file_source(self) -> bool:
        return self._is_file

    def get_frame_count(self) -> int:
        if not self.is_opened():
            return 0
        if not self.is_file_source():
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_index(self) -> int:
        if not self.is_opened():
            return 0
        if not self.is_file_source():
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek_to_frame(self, index: int) -> None:
        if not self.is_opened():
            return
        if not self.is_file_source():
            return
        index = max(0, index)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)

