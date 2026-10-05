import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig, ModelConfig


@dataclass
class DetectionItem:
    x: int
    y: int
    w: int
    h: int
    class_id: int
    class_name: str
    confidence: float
    model_name: str = "general"


class SingleModelEngine:
    """Loads and runs inference for a single YOLO model."""

    def __init__(self, name: str, config: ModelConfig):
        self.name = name
        self.config = config
        self._model = None
        self._error: Optional[str] = None
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._error is None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _load_model(self):
        model_path = Path(self.config.path)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parent
            model_path = (project_root / model_path).resolve()

        if not model_path.exists():
            self._error = f"Model file not found: {model_path}"
            return

        try:
            ultralytics = importlib.import_module("ultralytics")
            YOLO = ultralytics.YOLO
            self._model = YOLO(str(model_path))
            self._error = None
            print(f"[DetectionEngine] Loaded {self.name} model from {model_path}")
        except Exception as exc:
            self._model = None
            self._error = f"Failed to load {self.name} model: {exc}"
            print(f"[DetectionEngine] ERROR ({self.name}): {self._error}")

    def detect(self, frame: np.ndarray) -> List[DetectionItem]:
        if not self.is_ready:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.config.confidence,
                iou=0.45,
                verbose=False,
                imgsz=self.config.imgsz,
            )
        except Exception as exc:
            print(f"[DetectionEngine] {self.name} predict exception: {exc}")
            return []

        if not results:
            return []

        detections: List[DetectionItem] = []
        det = results[0]
        if det.boxes is None:
            return detections

        names = det.names if hasattr(det, "names") else getattr(self._model, "names", {})

        xyxy = det.boxes.xyxy.cpu().numpy().astype(int)
        confs = (
            det.boxes.conf.cpu().numpy()
            if det.boxes.conf is not None
            else np.ones((len(xyxy),), dtype=float)
        )
        classes = (
            det.boxes.cls.cpu().numpy().astype(int)
            if det.boxes.cls is not None
            else np.zeros((len(xyxy),), dtype=int)
        )

        for idx, (x1, y1, x2, y2) in enumerate(xyxy):
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            class_id = int(classes[idx])
            class_name = (
                str(names.get(class_id, f"class_{class_id}"))
                if isinstance(names, dict)
                else str(class_id)
            )
            confidence = float(confs[idx])

            detections.append(
                DetectionItem(
                    x=int(x1),
                    y=int(y1),
                    w=int(w),
                    h=int(h),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    model_name=self.name,
                )
            )

        return detections


class DetectionEngine:
    """Unified engine for all Motherson vision models."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models: Dict[str, SingleModelEngine] = {}
        self.reload_all_models()

    def reload_all_models(self):
        self.models = {
            "scratch": SingleModelEngine("scratch", self.config.scratch_config),
            "cap": SingleModelEngine("cap", self.config.cap_config),
            "defect_2": SingleModelEngine("defect_2", self.config.defect_2_config),
            "part_presence": SingleModelEngine("part_presence", self.config.part_presence_config),
            "barcode": SingleModelEngine("barcode", self.config.barcode_config),
        }

    def detect_model(self, model_key: str, frame: np.ndarray) -> List[DetectionItem]:
        if model_key in self.models:
            return self.models[model_key].detect(frame)
        return []

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[DetectionItem],
        draw_part_presence: bool = False,
    ) -> np.ndarray:
        """Annotates image with bounding boxes.
        
        Rules:
        - cap_model: 'cap' and 'foam' -> GREEN box (0, 255, 0)
                      'no_cap' -> RED box (0, 0, 255)
        - defect_2 / scratch: RED box (0, 0, 255)
        - barcode: AMBER/YELLOW box (0, 215, 255)
        """
        out = frame.copy()
        box_thickness = 3
        text_scale = 0.8
        text_thickness = 2
        text_padding = 10

        for item in detections:
            # Skip part_presence bounding box unless explicitly requested
            if item.model_name == "part_presence" and not draw_part_presence:
                continue

            cname = item.class_name.strip().lower()

            # Determine box color
            if item.model_name == "cap":
                if "no_cap" in cname or "nocap" in cname:
                    color = (0, 0, 255)  # RED
                elif "cap" in cname or "foam" in cname:
                    color = (0, 255, 0)  # GREEN
                else:
                    color = (0, 255, 0)  # Default GREEN for cap model good items
            elif item.model_name == "barcode":
                color = (0, 215, 255)  # YELLOW/GOLD
            else:
                color = (0, 0, 255)  # RED for defects/scratch

            x, y, w, h = item.x, item.y, item.w, item.h
            cv2.rectangle(out, (x, y), (x + w, y + h), color, box_thickness)

            label = f"{item.class_name} ({item.confidence*100:.0f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            y1 = max(0, y - text_h - baseline - text_padding)
            y2 = max(0, y - 2)
            x2 = x + text_w + text_padding

            cv2.rectangle(out, (x, y1), (x2, y2), color, -1)
            cv2.putText(
                out,
                label,
                (x + (text_padding // 2), y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return out
