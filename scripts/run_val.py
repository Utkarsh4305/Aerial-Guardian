"""
Run inference over all VisDrone MOT validation sequences and write MOT-format results.

After this, run evaluate.py to compute MOTA / IDF1.

Usage:
    python scripts/run_val.py \
        --sequences data/raw/VisDrone2019-MOT-val/sequences \
        --results   results/val \
        [--no-sahi] [--device cpu] [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when script is run as scripts/run_val.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", default="data/raw/VisDrone2019-MOT-val/sequences",
                   help="Directory containing one sub-directory per sequence")
    p.add_argument("--results", default="results/val",
                   help="Output directory for MOT-format .txt files")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--no-sahi", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.no_sahi:
        cfg["sahi"]["enabled"] = False
    if args.device:
        cfg["detector"]["device"] = args.device

    seq_root = Path(args.sequences)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d for d in seq_root.iterdir() if d.is_dir()])
    if not sequences:
        print(f"No sequence directories found under '{seq_root}'")
        return

    print(f"Found {len(sequences)} sequences -> results in '{results_dir}'\n")

    weights_path = cfg["detector"]["weights"]
    if not __import__("pathlib").Path(weights_path).exists():
        print(f"[Weights] '{weights_path}' not found -- downloading ...")
        from scripts.download_weights import download as _dl  # noqa: E402
        _dl(weights_path)
        print()

    from src.pipeline import Pipeline

    pipe = Pipeline(cfg)  # load model once, reuse for all sequences

    for seq_dir in sequences:
        mot_path = results_dir / f"{seq_dir.name}.txt"
        print(f"[{seq_dir.name}]")
        stats = pipe.run(
            str(seq_dir),
            output_path=None,
            mot_output_path=str(mot_path),
        )
        print(
            f"  frames={stats['total_frames']}  tracks={stats['total_tracks']}"
            f"  fps={stats['fps']:.1f}  -> {mot_path.name}\n"
        )

    print("Done. Now run:")
    print(
        f"  python scripts/evaluate.py "
        f"--gt data/raw/VisDrone2019-MOT-val/annotations "
        f"--pred {results_dir}"
    )


if __name__ == "__main__":
    main()
