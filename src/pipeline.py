"""
End-to-end pipeline: frame source -> detector -> tracker -> visualizer -> writer.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.detector import Detector
from src.tracker import ByteTracker
from src.visualizer import Visualizer


def _frame_source(source: str) -> Generator[np.ndarray, None, None]:
    """
    Yield BGR frames from:
      - a video file (.mp4, .avi, ...)
      - a directory of images (sorted by name)
    """
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        for f in files:
            frame = cv2.imread(str(f))
            if frame is not None:
                yield frame
    else:
        cap = cv2.VideoCapture(str(p))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()


def _get_frame_count(source: str) -> int:
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return len([f for f in p.iterdir() if f.suffix.lower() in exts])
    cap = cv2.VideoCapture(str(p))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _get_frame_size(source: str) -> tuple[int, int]:
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        if files:
            frame = cv2.imread(str(files[0]))
            if frame is not None:
                h, w = frame.shape[:2]
                return w, h
        return (1280, 720)
    cap = cv2.VideoCapture(str(p))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


class Pipeline:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.tracker = ByteTracker(cfg)
        self.visualizer = Visualizer(cfg)

    def run(
        self,
        source: str,
        output_path: Optional[str] = None,
        mot_output_path: Optional[str] = None,
    ) -> dict:
        """
        Process a video or image sequence.

        Args:
            source:          Video file or image-sequence directory.
            output_path:     Path for annotated output video (optional).
            mot_output_path: Path for MOT-format tracking results .txt (optional).
                             Format per line: <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1

        Returns:
            dict with 'fps', 'total_frames', 'total_tracks', 'output', 'mot_output'
        """
        out_cfg = self.cfg["output"]
        w, h = _get_frame_size(source)
        total_frames = _get_frame_count(source)

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*out_cfg["codec"])
            writer = cv2.VideoWriter(output_path, fourcc, out_cfg["fps"], (w, h))

        mot_lines: list[str] = []

        self.tracker.reset()
        seen_ids: set[int] = set()
        frame_times: list[float] = []
        frame_id = 0

        with tqdm(total=total_frames, desc="Processing", unit="fr") as pbar:
            for frame in _frame_source(source):
                frame_id += 1
                t0 = time.perf_counter()

                detections = self.detector.detect(frame)
                active_tracks = self.tracker.update(detections, frame)
                annotated = self.visualizer.draw(frame, active_tracks)

                elapsed = time.perf_counter() - t0
                frame_times.append(elapsed)

                for t in active_tracks:
                    seen_ids.add(t.track_id)
                    if mot_output_path is not None:
                        x1, y1, x2, y2 = t.to_xyxy()
                        bw = max(0.0, float(x2 - x1))
                        bh = max(0.0, float(y2 - y1))
                        mot_lines.append(
                            f"{frame_id},{t.track_id},{x1:.2f},{y1:.2f},{bw:.2f},{bh:.2f},{t.conf:.4f},-1,-1,-1"
                        )

                if writer is not None:
                    writer.write(annotated)

                pbar.update(1)
                pbar.set_postfix(fps=f"{1/elapsed:.1f}", tracks=len(active_tracks))

        if writer is not None:
            writer.release()

        if mot_output_path is not None:
            Path(mot_output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(mot_output_path).write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""))

        avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

        return {
            "fps": avg_fps,
            "total_frames": len(frame_times),
            "total_tracks": len(seen_ids),
            "output": output_path,
            "mot_output": mot_output_path,
        }
