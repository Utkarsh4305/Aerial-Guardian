"""Run inference on sequences not yet in results/val/."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

from src.pipeline import Pipeline

results_dir = Path("results/val")
results_dir.mkdir(parents=True, exist_ok=True)

seq_root = Path("VisDrone2019-MOT-val/sequences")
sequences = sorted([d for d in seq_root.iterdir() if d.is_dir()])

pipe = Pipeline(cfg)

for seq_dir in sequences:
    mot_path = results_dir / f"{seq_dir.name}.txt"
    if mot_path.exists():
        print(f"[{seq_dir.name}] already done, skipping.")
        continue
    print(f"[{seq_dir.name}]", flush=True)
    stats = pipe.run(str(seq_dir), output_path=None, mot_output_path=str(mot_path))
    print(
        f"  frames={stats['total_frames']}  tracks={stats['total_tracks']}"
        f"  fps={stats['fps']:.1f}  -> {mot_path.name}",
        flush=True,
    )

print("ALL DONE")
