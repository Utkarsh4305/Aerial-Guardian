"""
Convert VisDrone MOT annotations to YOLO detection format (person class only).

VisDrone MOT annotation columns:
  frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
  score, object_category, truncation, occlusion

object_category values:
  0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
  6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others

We keep categories 1 (pedestrian) and 2 (people) -> both become class 0 = person.

Output YOLO label format per line:
  <class> <cx_norm> <cy_norm> <w_norm> <h_norm>

Usage:
  python scripts/convert_visdrone.py \
      --src  data/raw/VisDrone2019-MOT-val \
      --dest data/yolo_format
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

PERSON_CATS = {1, 2}  # pedestrian + people


def convert_sequence(seq_dir: Path, out_img_dir: Path, out_lbl_dir: Path) -> int:
    ann_dir = seq_dir / "annotations"
    img_dir = seq_dir / "sequences" / seq_dir.name

    if not img_dir.exists():
        # Some releases store images directly under seq_dir/sequences/<name>
        # or under seq_dir directly
        img_dir = seq_dir / "sequences"
        if not img_dir.exists():
            img_dir = seq_dir

    ann_files = sorted(ann_dir.glob("*.txt")) if ann_dir.exists() else []
    if not ann_files:
        return 0

    count = 0
    for ann_file in tqdm(ann_files, desc=seq_dir.name, leave=False):
        frame_annotations: dict[int, list] = {}
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    continue
                frame_idx = int(parts[0])
                cat = int(parts[7])
                if cat not in PERSON_CATS:
                    continue
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                frame_annotations.setdefault(frame_idx, []).append((x, y, w, h))

        # Find image files for this annotation
        seq_name = ann_file.stem
        frames_dir = img_dir / seq_name if (img_dir / seq_name).exists() else img_dir
        img_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))

        for img_path in img_files:
            frame_idx = int(img_path.stem)
            if frame_idx not in frame_annotations:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            stem = f"{seq_dir.name}_{img_path.stem}"
            dst_img = out_img_dir / img_path.name.replace(img_path.stem, stem)
            dst_lbl = out_lbl_dir / f"{stem}.txt"

            shutil.copy2(img_path, dst_img)

            with open(dst_lbl, "w") as lf:
                for (x, y, w, h) in frame_annotations[frame_idx]:
                    cx = (x + w / 2) / iw
                    cy = (y + h / 2) / ih
                    wn = w / iw
                    hn = h / ih
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    wn = max(0.0, min(1.0, wn))
                    hn = max(0.0, min(1.0, hn))
                    lf.write(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
            count += 1
    return count


def write_dataset_yaml(dest: Path) -> None:
    yaml_path = dest / "dataset.yaml"
    content = f"""path: {dest.resolve()}
train: images
val: images

nc: 1
names: ['person']
"""
    yaml_path.write_text(content)
    print(f"Written {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/VisDrone2019-MOT-val")
    parser.add_argument("--dest", default="data/yolo_format")
    args = parser.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    out_img = dest / "images"
    out_lbl = dest / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    sequences = [d for d in src.iterdir() if d.is_dir()]
    total = 0
    for seq in sequences:
        total += convert_sequence(seq, out_img, out_lbl)

    print(f"Converted {total} labeled frames -> {dest}")
    write_dataset_yaml(dest)


if __name__ == "__main__":
    main()
