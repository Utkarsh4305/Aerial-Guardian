"""
Benchmark FPS and report hardware.

Runs the full pipeline (detector + tracker + visualizer) over a video/sequence
and reports per-component timing breakdowns.

Usage:
    python scripts/benchmark.py --source data/raw/VisDrone2019-MOT-val/sequences/uav0000013_00000_v
    python scripts/benchmark.py --source clip.mp4 --no-sahi
"""
import argparse
import platform
import time
import sys
from pathlib import Path

# Ensure project root is on sys.path when script is run as scripts/benchmark.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--frames", type=int, default=200, help="Max frames to benchmark")
    p.add_argument("--no-sahi", action="store_true")
    return p.parse_args()


def hw_info() -> str:
    info = [
        f"Platform : {platform.platform()}",
        f"Processor: {platform.processor()}",
        f"Python   : {sys.version.split()[0]}",
    ]
    try:
        import torch
        info.append(f"PyTorch  : {torch.__version__}")
        if torch.cuda.is_available():
            info.append(f"CUDA GPU : {torch.cuda.get_device_name(0)}")
        else:
            info.append("CUDA GPU : not available")
    except ImportError:
        pass
    return "\n  ".join(info)


def load_frames(source: str, max_frames: int) -> list:
    frames = []
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])[:max_frames]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
    else:
        cap = cv2.VideoCapture(str(p))
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    return frames


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.no_sahi:
        cfg["sahi"]["enabled"] = False

    print(f"\n=== Aerial Guardian Benchmark ===")
    print(f"  {hw_info()}")
    print(f"  SAHI: {cfg['sahi']['enabled']}\n")

    print("Loading frames ...")
    frames = load_frames(args.source, args.frames)
    if not frames:
        print("No frames found -- check --source path.")
        return
    print(f"Loaded {len(frames)} frames  ({frames[0].shape[1]}x{frames[0].shape[0]})\n")

    weights_path = cfg["detector"]["weights"]
    if not __import__("pathlib").Path(weights_path).exists():
        print(f"[Weights] '{weights_path}' not found -- downloading ...")
        from scripts.download_weights import download as _dl
        _dl(weights_path)
        print()

    from src.detector import Detector
    from src.tracker import ByteTracker
    from src.visualizer import Visualizer

    detector = Detector(cfg)
    tracker = ByteTracker(cfg)
    visualizer = Visualizer(cfg)

    det_times, track_times, vis_times = [], [], []

    for frame in frames:
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()
        tracks = tracker.update(dets, frame)
        t2 = time.perf_counter()
        _ = visualizer.draw(frame, tracks)
        t3 = time.perf_counter()

        det_times.append(t1 - t0)
        track_times.append(t2 - t1)
        vis_times.append(t3 - t2)

    def ms(times):
        return f"{1000 * sum(times) / len(times):.1f} ms"

    total = [d + tr + v for d, tr, v in zip(det_times, track_times, vis_times)]
    fps = len(total) / sum(total)

    print("=== Timing Breakdown (per frame, averaged) ===")
    print(f"  Detection  : {ms(det_times)}")
    print(f"  Tracking   : {ms(track_times)}")
    print(f"  Visualize  : {ms(vis_times)}")
    print(f"  Total      : {ms(total)}")
    print(f"\n  FPS        : {fps:.2f}")
    print()


if __name__ == "__main__":
    main()
