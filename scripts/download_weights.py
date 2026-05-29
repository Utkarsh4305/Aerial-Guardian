"""
Download VisDrone fine-tuned YOLOv8n weights from HuggingFace.

Model: mshamrai/yolov8n-visdrone  (6.2 MB)
  - Trained on VisDrone 2019 dataset (aerial/UAV footage)
  - 10 classes: pedestrian, people, bicycle, car, van, truck,
                tricycle, awning-tricycle, bus, motor
  - We use classes 0 (pedestrian) and 1 (people) as "person"

Usage:
    python scripts/download_weights.py [--dest weights/yolov8n_visdrone.pt]
"""
import argparse
import sys
from pathlib import Path

HF_URL = "https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/best.pt"


def download(dest: str) -> None:
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"[Weights] Already downloaded at {out}  (delete to re-download)")
        return

    print(f"[Weights] Downloading VisDrone YOLOv8n from HuggingFace ...")
    print(f"          Source : {HF_URL}")
    print(f"          Dest   : {out}")

    try:
        import requests
        from tqdm import tqdm

        r = requests.get(HF_URL, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(out, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc="yolov8n_visdrone.pt"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                pbar.update(len(chunk))

    except ImportError:
        # requests not available -- fall back to stdlib
        import urllib.request

        def _reporthook(count, block_size, total_size):
            done = count * block_size
            if total_size > 0:
                pct = min(100, done * 100 // total_size)
                mb = done / 1_048_576
                sys.stdout.write(f"\r  {pct:3d}%  {mb:.1f} MB")
                sys.stdout.flush()

        urllib.request.urlretrieve(HF_URL, out, reporthook=_reporthook)
        print()

    size_mb = out.stat().st_size / 1_048_576
    print(f"[Weights] Saved {size_mb:.1f} MB -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="weights/yolov8n_visdrone.pt",
                   help="Destination path for the weights file")
    args = p.parse_args()
    download(args.dest)
