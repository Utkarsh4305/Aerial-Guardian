"""
Fine-tune YOLOv8n on the VisDrone person dataset.

This script adapts the base COCO-pretrained YOLOv8n to aerial drone footage
where persons appear very small. Key training choices:
  - imgsz=640: standard resolution; SAHI handles small objects at inference time
  - mosaic augmentation (default in YOLOv8): helps with small-object diversity
  - close_mosaic=10: disable mosaic last 10 epochs for stable convergence
  - Single class (person): keeps the head focused on the target category

After training, exports to ONNX for edge (Jetson) deployment reference.

Usage:
    python scripts/train.py [--data data/yolo_format/dataset.yaml] [--epochs 50]
"""
import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/yolo_format/dataset.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu", help="cpu | 0 | cuda")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="visdrone_person")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Augmentation tuned for aerial small objects
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,       # slight rotation -- drones tilt
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        # Small-object friendly anchor-free config
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    # Save best weights to weights/ directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        import shutil
        dest = weights_dir / "yolov8n_visdrone.pt"
        shutil.copy2(best, dest)
        print(f"\n[Train] Best weights saved -> {dest}")
        print("[Train] NOTE: this is a single-class (person) model.")
        print("[Train]   In configs/config.yaml set:  person_class_ids: [0]")

        # Export to ONNX for Jetson TensorRT path
        print("[Train] Exporting to ONNX ...")
        from ultralytics import YOLO as _YOLO
        export_model = _YOLO(str(dest))
        export_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"[Train] ONNX model saved -> {dest.with_suffix('.onnx')}")
    else:
        print("[Train] WARNING: best.pt not found -- check training output.")


if __name__ == "__main__":
    main()
