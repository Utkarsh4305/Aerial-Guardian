"""Generate annotated output video for a single sequence."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.pipeline import Pipeline

seq_dir = "VisDrone2019-MOT-val/sequences/uav0000086_00000_v"
out_video = "results/uav0000086_00000_v_tracked.mp4"

pipe = Pipeline(cfg)
stats = pipe.run(seq_dir, output_path=out_video, mot_output_path=None)
print(f"Done: frames={stats['total_frames']} tracks={stats['total_tracks']} fps={stats['fps']:.2f}")
print(f"Video: {out_video}")
