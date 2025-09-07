"""Optional phone detector using YOLOv8n.

If the YOLO weights file is unavailable or the ultralytics library fails to
import, the detector silently disables itself.  When enabled, it detects
``cell phone`` objects in the lower portion of the frame (the ``phone zone``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore


class PhoneDetector:
    """Phone detector using YOLOv8n.

    Parameters
    ----------
    model_path:
        Path to the YOLOv8n weights file (e.g. ``yolov8n.pt``).  If the file
        does not exist or YOLO cannot be imported, phone detection will be
        disabled.
    conf_thresh:
        Confidence threshold for detections.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.35) -> None:
        self.model = None
        self.conf_thresh = conf_thresh
        # Attempt to load the model
        if YOLO is None:
            return
        weights = Path(model_path)
        if not weights.exists():
            return
        try:
            self.model = YOLO(str(weights))
        except Exception:
            self.model = None

    def available(self) -> bool:
        """Return ``True`` if the detector is operational."""
        return self.model is not None

    def detect(self, frame: 'cv2.typing.MatLike', phone_zone_ratio: float = 0.55) -> List[Tuple[int, int, int, int]]:
        """Detect phones in the lower portion of the frame.

        Parameters
        ----------
        frame:
            BGR image in which to run detection.
        phone_zone_ratio:
            Fraction of the frame height considered the ``phone zone``.  Only
            detections whose centre y‑coordinate falls below this fraction are
            returned.

        Returns
        -------
        list of bounding boxes (x1, y1, x2, y2)
            Detected phone bounding boxes.  Empty if detection is disabled or no
            phones are found.
        """
        if not self.available():
            return []
        h, w = frame.shape[:2]
        zone_y = int(h * phone_zone_ratio)
        boxes: List[Tuple[int, int, int, int]] = []
        try:
            results = self.model.predict(source=frame, imgsz=416, conf=self.conf_thresh, verbose=False)
        except Exception:
            return []
        # Iterate over detections in the first (and only) result
        for result in results:
            if not hasattr(result, 'boxes'):
                continue
            for b in result.boxes:
                cls_name = result.names.get(int(b.cls), '')
                # Accept common labels for phones
                if cls_name not in {"cell phone", "mobile phone", "phone"}:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cy = int((y1 + y2) / 2)
                # Only consider objects in the lower part of the frame
                if cy >= zone_y:
                    boxes.append((x1, y1, x2, y2))
        return boxes