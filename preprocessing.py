from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration options for preprocessing and calibration."""

    resize_width: int = 640
    resize_height: int = 640
    use_gaussian_blur: bool = True
    use_median_blur: bool = False
    blur_kernel_size: int = 5

    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    brightness: float = 1.0  # gain (alpha)
    contrast: float = 0.0  # bias (beta, -100..100 typical)

    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class Preprocessor:
    """
    Handles image preprocessing for unstable factory lighting and ROI selection.

    Supports:
    - Resize
    - Noise reduction (Gaussian / Median)
    - Contrast enhancement (CLAHE)
    - Brightness / Contrast calibration
    - ROI cropping
    """

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self.config.roi = (x, y, w, h)

    def auto_calibrate(self, frame) -> None:
        """
        Simple automatic calibration stub:
        - Estimate brightness from grayscale mean
        - Adjust brightness/contrast heuristically
        - Use full frame as ROI (user can override)
        """
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))

        # Target mean around mid-level 128
        if mean_intensity < 100:
            self.config.brightness = 1.2
            self.config.contrast = 20
        elif mean_intensity > 150:
            self.config.brightness = 0.8
            self.config.contrast = -20
        else:
            self.config.brightness = 1.0
            self.config.contrast = 0.0

        h, w = frame.shape[:2]
        self.config.roi = (0, 0, w, h)

    def apply(self, frame):
        cfg = self.config

        # ROI cropping (conveyor area)
        if cfg.roi is not None:
            x, y, w, h = cfg.roi
            x2, y2 = x + w, y + h
            x, y = max(0, x), max(0, y)
            frame = frame[y:y2, x:x2]

        # Resize
        frame = cv2.resize(frame, (cfg.resize_width, cfg.resize_height))

        # Noise reduction
        if cfg.use_gaussian_blur:
            k = cfg.blur_kernel_size if cfg.blur_kernel_size % 2 == 1 else cfg.blur_kernel_size + 1
            frame = cv2.GaussianBlur(frame, (k, k), 0)
        elif cfg.use_median_blur:
            k = cfg.blur_kernel_size if cfg.blur_kernel_size % 2 == 1 else cfg.blur_kernel_size + 1
            frame = cv2.medianBlur(frame, k)

        # Contrast enhancement via CLAHE (on L channel)
        if cfg.use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid_size
            )
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Brightness / contrast adjustment (manual or auto)
        frame = cv2.convertScaleAbs(frame, alpha=cfg.brightness, beta=cfg.contrast)

        return frame

