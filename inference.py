"""
Aerial Guardian -- inference entrypoint.

Usage:
    python inference.py --source data/raw/VisDrone2019-MOT-val/sequences/uav0000013_00000_v
    python inference.py --source my_drone_clip.mp4 --output output/result.mp4
    python inference.py --source my_clip.mp4 --no-sahi   # faster, lower recall
"""
import argparse
import platform
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Aerial Guardian -- person detection & tracking")
    p.add_argument("--source", required=True, help="Video file or image sequence directory")
    p.add_argument("--output", default="output/result.mp4", help="Output video path")
    p.add_argument("--mot-output", default=None, help="Write MOT-format results to this .txt path")
    p.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    p.add_argument("--weights", default=None, help="Override model weights path")
    p.add_argument("--device", default=None, help="cpu | cuda | mps")
    p.add_argument("--no-sahi", action="store_true", help="Disable SAHI (faster, lower recall)")
    p.add_argument("--conf", type=float, default=None, help="Override detection confidence")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.weights:
        cfg["detector"]["weights"] = args.weights
    if args.device:
        cfg["detector"]["device"] = args.device
    if args.no_sahi:
        cfg["sahi"]["enabled"] = False
    if args.conf is not None:
        cfg["detector"]["conf_thresh"] = args.conf

    print("\n=== Aerial Guardian ===")
    print(f"  Source  : {args.source}")
    print(f"  Output  : {args.output}")
    print(f"  Device  : {cfg['detector']['device']}")
    print(f"  SAHI    : {cfg['sahi']['enabled']}")
    print(f"  Conf    : {cfg['detector']['conf_thresh']}")
    print(f"  Python  : {sys.version.split()[0]}  |  Platform: {platform.platform()}")
    print()

    # Auto-download weights if the configured path doesn't exist
    weights_path = cfg["detector"]["weights"]
    if not __import__("pathlib").Path(weights_path).exists():
        print(f"[Weights] '{weights_path}' not found -- downloading pre-trained VisDrone weights ...")
        from scripts.download_weights import download as _dl
        _dl(weights_path)
        print()

    from src.pipeline import Pipeline
    pipe = Pipeline(cfg)
    mot_out = getattr(args, "mot_output", None)
    stats = pipe.run(args.source, args.output, mot_output_path=mot_out)

    print(f"\n--- Results ---")
    print(f"  Frames processed : {stats['total_frames']}")
    print(f"  Unique track IDs : {stats['total_tracks']}")
    print(f"  Average FPS      : {stats['fps']:.2f}")
    if stats["output"]:
        print(f"  Output video     : {stats['output']}")
    if stats.get("mot_output"):
        print(f"  MOT results      : {stats['mot_output']}")
    print()


if __name__ == "__main__":
    main()
