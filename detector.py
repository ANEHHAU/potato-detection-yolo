import os
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO


class DetectionResult:
    """Single detection result."""

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        cls_id: int,
        cls_name: str,
        confidence: float,
        track_id: Optional[int] = None,
    ) -> None:
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.confidence = confidence
        self.track_id = track_id

    @property
    def center(self) -> Tuple[int, int]:
        cx = int((self.x1 + self.x2) / 2)
        cy = int((self.y1 + self.y2) / 2)
        return cx, cy


class PotatoDetector:
    """
    YOLOv8-based detector wrapper.

    This class is responsible ONLY for running the YOLO model and returning
    structured detection results. Tracking IDs are optionally attached when
    used together with a tracking pipeline (e.g. ByteTrack).
    """

    def __init__(
        self,
        model_path: str = "best2.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "",  # "", "cpu", "cuda"
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at '{model_path}'")

        self.model = YOLO(model_path)
        if device:
            # Let Ultralytics choose device if not specified
            self.model.to(device)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Original class names from the model
        self.class_names = self.model.names

        # Optional human-friendly / localized names (example: Vietnamese)
        self.localized_names = {
            "Damaged potato": "Khoai nứt",
            "Defected potato": "Khoai biến dạng",
            "Diseased-fungal potato": "Khoai nấm bệnh",
            "Potato": "Khoai tốt",
            "Sprouted potato": "Khoai mọc mầm",
        }

    def get_display_name(self, cls_name: str) -> str:
        return self.localized_names.get(cls_name, cls_name)

    def detect(self, frame) -> List[DetectionResult]:
        """
        Run pure detection (no tracking IDs).
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: List[DetectionResult] = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.class_names[cls_id]
                detections.append(
                    DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        cls_id=cls_id,
                        cls_name=cls_name,
                        confidence=conf,
                        track_id=None,
                    )
                )

        return detections

    def track(self, frame, persist: bool = True, tracker_config: str = "bytetrack.yaml") -> List[DetectionResult]:
        """
        Run detection + tracking using Ultralytics built-in ByteTrack integration.

        This satisfies the requirement of using ByteTrack as the default tracker
        while keeping the interface clean. Tracking IDs are attached to results.
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            persist=persist,
            tracker=tracker_config,
            verbose=False,
        )

        detections: List[DetectionResult] = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            if boxes.id is None:
                continue

            for box, track_id in zip(boxes, boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.class_names[cls_id]

                detections.append(
                    DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        cls_id=cls_id,
                        cls_name=cls_name,
                        confidence=conf,
                        track_id=int(track_id),
                    )
                )

        return detections


def draw_detections(frame, detections: List[DetectionResult]) -> None:
    """
    Simple visualization helper for debugging without the full UI.
    """
    for det in detections:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"{det.cls_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"ID:{det.track_id} {label}"
        cv2.putText(
            frame,
            label,
            (det.x1, max(0, det.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

