"""
Detector: YOLOv8n wrapped with SAHI tiled inference for small-object detection.

Why SAHI?
  Drone footage typically captures persons at very small pixel sizes (< 20px).
  Standard full-frame YOLO inference often misses these because they never fill
  the receptive field. SAHI slices each frame into overlapping tiles so small
  objects appear proportionally larger to the network.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np


class Detector:
    """YOLOv8n + SAHI detector returning person detections per frame."""

    def __init__(self, cfg: dict) -> None:
        self.conf = cfg["detector"]["conf_thresh"]
        self.iou = cfg["detector"]["iou_thresh"]
        self.device = cfg["detector"]["device"]

        det = cfg["detector"]
        if "person_class_ids" in det:
            self.person_cls_ids: list[int] = list(det["person_class_ids"])
        else:
            # backward compat with single-id config key
            self.person_cls_ids = [int(det["person_class_id"])]
        self._person_cls_set = set(self.person_cls_ids)

        weights = det["weights"]
        self.sahi_cfg = cfg["sahi"]

        self._model = None
        self._sahi_model = None
        self._weights = weights
        self._use_sahi = self.sahi_cfg["enabled"]

        self._load(weights)

    def _load(self, weights: str) -> None:
        from ultralytics import YOLO

        # Fall back to pretrained YOLOv8n if fine-tuned weights not yet available
        if not Path(weights).exists():
            print(f"[Detector] '{weights}' not found -- using pretrained yolov8n.pt")
            weights = "yolov8n.pt"

        self._model = YOLO(weights)

        if self._use_sahi:
            try:
                from sahi import AutoDetectionModel
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=weights,
                    confidence_threshold=self.conf,
                    device=self.device,
                )
                print("[Detector] SAHI tiled inference enabled.")
            except ImportError:
                print("[Detector] sahi not installed -- falling back to full-frame inference.")
                self._use_sahi = False

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame.

        Returns:
            np.ndarray of shape (N, 5): [x1, y1, x2, y2, conf]
        """
        if self._use_sahi and self._sahi_model is not None:
            return self._detect_sahi(frame)
        return self._detect_full(frame)

    def _detect_sahi(self, frame: np.ndarray) -> np.ndarray:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            frame,
            self._sahi_model,
            slice_height=self.sahi_cfg["slice_height"],
            slice_width=self.sahi_cfg["slice_width"],
            overlap_height_ratio=self.sahi_cfg["overlap_height_ratio"],
            overlap_width_ratio=self.sahi_cfg["overlap_width_ratio"],
            postprocess_type=self.sahi_cfg["postprocess_type"],
            postprocess_match_threshold=self.sahi_cfg["postprocess_match_threshold"],
            verbose=0,
        )

        detections = []
        for obj in result.object_prediction_list:
            if obj.category.id not in self._person_cls_set:
                continue
            bb = obj.bbox
            detections.append([bb.minx, bb.miny, bb.maxx, bb.maxy, obj.score.value])

        return np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)

    def _detect_full(self, frame: np.ndarray) -> np.ndarray:
        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.person_cls_ids,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return np.empty((0, 5), dtype=np.float32)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.concatenate([xyxy, confs], axis=1).astype(np.float32)
