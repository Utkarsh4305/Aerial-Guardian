"""
Visualizer: draws bounding boxes, track IDs, and fading trajectory tails.
"""
from __future__ import annotations

import cv2
import numpy as np


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per track ID using a hash."""
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    ]
    color_bgr = palette[track_id % len(palette)]
    return (color_bgr[2], color_bgr[1], color_bgr[0])  # RGB -> BGR


class Visualizer:
    def __init__(self, cfg: dict) -> None:
        vis_cfg = cfg["visualizer"]
        self.tail_length: int = vis_cfg["tail_length"]
        self.box_thickness: int = vis_cfg["box_thickness"]
        self.font_scale: float = vis_cfg["font_scale"]

    def draw(
        self,
        frame: np.ndarray,
        tracks,
    ) -> np.ndarray:
        """
        Draw detections on a copy of frame.

        Args:
            frame:  BGR image.
            tracks: list of Track objects from ByteTracker.

        Returns:
            Annotated BGR frame.
        """
        out = frame.copy()
        overlay = out.copy()

        for track in tracks:
            tid = track.track_id
            color = _id_color(tid)
            box = track.to_xyxy().astype(int)
            x1, y1, x2, y2 = box

            # --- Trajectory tail ---
            history = track.history[-self.tail_length:]
            for i in range(1, len(history)):
                alpha = i / len(history)
                thick = max(1, int(self.box_thickness * alpha))
                pt1 = history[i - 1]
                pt2 = history[i]
                cv2.line(overlay, pt1, pt2, color, thick)

            # --- Bounding box ---
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.box_thickness)

            # --- ID label ---
            label = f"#{tid}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            label_bg_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(overlay, (x1, label_bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                overlay, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Blend overlay for semi-transparent tails
        cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)
        return out
