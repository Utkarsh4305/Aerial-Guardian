"""
Evaluate tracking results against VisDrone ground-truth using MOT metrics.

Requires: pip install motmetrics

Reads:
  - Ground-truth from VisDrone MOT annotation files (one .txt per sequence)
  - Tracker output from MOT-format result files (one .txt per sequence)

Outputs per-sequence and overall: MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs

Usage:
    # Run inference first to generate MOT results:
    python inference.py \
        --source data/raw/VisDrone2019-MOT-val/sequences/uav0000013_00000_v \
        --mot-output results/uav0000013_00000_v.txt

    # Evaluate all sequences:
    python scripts/evaluate.py \
        --gt   data/raw/VisDrone2019-MOT-val/annotations \
        --pred results \
        --iou  0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

PERSON_CATS = {1, 2}


def _parse_gt_file(path: Path) -> tuple[dict[int, list], dict[int, list]]:
    """Parse VisDrone MOT annotation.

    Returns (gt_data, ignore_data) where:
      gt_data     = {frame_id: [(x1,y1,x2,y2,gt_id), ...]}  (person categories 1,2 with score==1)
      ignore_data = {frame_id: [(x1,y1,x2,y2), ...]}         (cat==0 or score==0 regions)
    """
    gt_data: dict[int, list] = {}
    ignore_data: dict[int, list] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            frame_idx = int(parts[0])
            target_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = int(parts[6])
            cat = int(parts[7])
            if w <= 0 or h <= 0:
                continue
            box = (x, y, x + w, y + h)
            if cat in PERSON_CATS and score == 1:
                gt_data.setdefault(frame_idx, []).append(box + (target_id,))
            else:
                # cat==0 (ignore region) or score==0 (annotated as ignored)
                ignore_data.setdefault(frame_idx, []).append(box)
    return gt_data, ignore_data


def _parse_pred_file(path: Path) -> dict[int, list]:
    """Parse MOT-format prediction; returns {frame_id: [(x1,y1,x2,y2,track_id,conf), ...]}"""
    data: dict[int, list] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            if w <= 0 or h <= 0:
                continue
            data.setdefault(frame_idx, []).append((x, y, x + w, y + h, track_id, conf))
    return data


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix (AxB) between two sets of [x1,y1,x2,y2] boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = a[:, None, :]
    bb = b[None, :, :]
    ix1 = np.maximum(aa[..., 0], bb[..., 0])
    iy1 = np.maximum(aa[..., 1], bb[..., 1])
    ix2 = np.minimum(aa[..., 2], bb[..., 2])
    iy2 = np.minimum(aa[..., 3], bb[..., 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (aa[..., 2] - aa[..., 0]) * (aa[..., 3] - aa[..., 1])
    area_b = (bb[..., 2] - bb[..., 0]) * (bb[..., 3] - bb[..., 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def _max_iou_with_regions(box: np.ndarray, regions: list) -> float:
    """Return max IoU of a single [x1,y1,x2,y2] box against a list of region boxes."""
    if not regions:
        return 0.0
    reg_arr = np.array(regions, dtype=np.float32)
    iou_row = _iou(box[None], reg_arr)
    return float(iou_row.max())


def evaluate_sequence(
    gt: dict[int, list],
    pred: dict[int, list],
    iou_thresh: float = 0.5,
    ignore: dict[int, list] | None = None,
) -> dict:
    """
    Compute CLEAR MOT + IDF1 for one sequence.

    Returns dict with TP, FP, FN, ID_switches, total_gt, MOTA, MOTP, IDF1, MT, ML.
    Predictions that fall inside ignored regions (IoU >= iou_thresh) are not counted as FP.
    """
    if ignore is None:
        ignore = {}
    all_frames = sorted(set(gt.keys()) | set(pred.keys()))

    TP = FP = FN = ID_switches = 0
    iou_sum = 0.0
    match_count = 0

    # id-level: for IDF1 -- count per (gt_id, pred_id) overlap
    idtp: dict[tuple, int] = {}
    idfp: dict[int, int] = {}
    idfn: dict[int, int] = {}

    prev_gt2pred: dict[int, int] = {}

    track_lengths: dict[int, int] = {}
    track_covered: dict[int, int] = {}

    for frame in all_frames:
        gt_boxes_raw = gt.get(frame, [])
        pred_boxes_raw = pred.get(frame, [])

        for _, _, _, _, gt_id in gt_boxes_raw:
            track_lengths[gt_id] = track_lengths.get(gt_id, 0) + 1

        ignore_boxes = ignore.get(frame, [])

        if not gt_boxes_raw:
            for x1, y1, x2, y2, pid, _ in pred_boxes_raw:
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                if _max_iou_with_regions(box, ignore_boxes) >= iou_thresh:
                    continue  # prediction inside ignored region — not a FP
                FP += 1
                idfp[pid] = idfp.get(pid, 0) + 1
            continue

        if not pred_boxes_raw:
            FN += len(gt_boxes_raw)
            for _, _, _, _, gid in gt_boxes_raw:
                idfn[gid] = idfn.get(gid, 0) + 1
            continue

        gt_arr = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in gt_boxes_raw])
        pred_arr = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _, _ in pred_boxes_raw])
        gt_ids = [r[4] for r in gt_boxes_raw]
        pred_ids = [r[4] for r in pred_boxes_raw]

        iou_mat = _iou(gt_arr, pred_arr)

        # Greedy matching (highest IoU first, above threshold)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        gt2pred: dict[int, int] = {}

        flat_order = np.argsort(iou_mat.ravel())[::-1]
        for idx in flat_order:
            gi, pi = divmod(int(idx), len(pred_ids))
            if iou_mat[gi, pi] < iou_thresh:
                break
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            gt2pred[gt_ids[gi]] = pred_ids[pi]
            iou_sum += iou_mat[gi, pi]
            match_count += 1
            pair = (gt_ids[gi], pred_ids[pi])
            idtp[pair] = idtp.get(pair, 0) + 1
            track_covered[gt_ids[gi]] = track_covered.get(gt_ids[gi], 0) + 1

        TP += len(matched_gt)
        FN += len(gt_ids) - len(matched_gt)

        # ID switches
        for gid, pid in gt2pred.items():
            if gid in prev_gt2pred and prev_gt2pred[gid] != pid:
                ID_switches += 1

        for gi in range(len(gt_ids)):
            if gi not in matched_gt:
                idfn[gt_ids[gi]] = idfn.get(gt_ids[gi], 0) + 1
        for pi in range(len(pred_ids)):
            if pi not in matched_pred:
                x1, y1, x2, y2 = pred_arr[pi]
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
                if _max_iou_with_regions(box, ignore_boxes) >= iou_thresh:
                    continue  # suppressed by ignored region
                FP += 1
                idfp[pred_ids[pi]] = idfp.get(pred_ids[pi], 0) + 1

        prev_gt2pred = gt2pred

    total_gt = TP + FN
    MOTA = 1.0 - (FN + FP + ID_switches) / max(total_gt, 1)
    MOTP = iou_sum / max(match_count, 1)

    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    total_idtp = sum(idtp.values())
    total_idfp = sum(idfp.values())
    total_idfn = sum(idfn.values())
    IDF1 = (2 * total_idtp) / max(2 * total_idtp + total_idfp + total_idfn, 1)

    # MT (mostly tracked >= 80%) / ML (mostly lost <= 20%)
    MT = sum(1 for gid, length in track_lengths.items()
             if track_covered.get(gid, 0) / length >= 0.8)
    ML = sum(1 for gid, length in track_lengths.items()
             if track_covered.get(gid, 0) / length <= 0.2)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "IDs": ID_switches,
        "GT": total_gt,
        "MOTA": MOTA,
        "MOTP": MOTP,
        "IDF1": IDF1,
        "MT": MT,
        "ML": ML,
        "num_tracks": len(track_lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Aerial Guardian tracking results")
    parser.add_argument("--gt", required=True, help="VisDrone annotations directory (*.txt per sequence)")
    parser.add_argument("--pred", required=True, help="Tracker results directory (*.txt per sequence)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a true positive")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)

    gt_files = {p.stem: p for p in gt_dir.glob("*.txt")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}

    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        print(f"No matching sequence names found between '{gt_dir}' and '{pred_dir}'.")
        print(f"GT sequences   : {sorted(gt_files)[:5]} ...")
        print(f"Pred sequences : {sorted(pred_files)[:5]} ...")
        return

    print(f"\nEvaluating {len(common)} sequence(s) at IoU={args.iou}\n")
    header = f"{'Sequence':<40} {'MOTA':>7} {'MOTP':>7} {'IDF1':>7} {'MT':>5} {'ML':>5} {'FP':>7} {'FN':>7} {'IDs':>5}"
    print(header)
    print("-" * len(header))

    totals = {"TP": 0, "FP": 0, "FN": 0, "IDs": 0, "GT": 0,
              "MT": 0, "ML": 0, "num_tracks": 0, "iou_sum": 0.0, "matches": 0}

    for seq in common:
        gt_data, ignore_data = _parse_gt_file(gt_files[seq])
        pred_data = _parse_pred_file(pred_files[seq])
        m = evaluate_sequence(gt_data, pred_data, iou_thresh=args.iou, ignore=ignore_data)

        print(
            f"{seq:<40} {m['MOTA']:>7.3f} {m['MOTP']:>7.3f} {m['IDF1']:>7.3f}"
            f" {m['MT']:>5d} {m['ML']:>5d} {m['FP']:>7d} {m['FN']:>7d} {m['IDs']:>5d}"
        )
        for k in ("TP", "FP", "FN", "IDs", "GT", "MT", "ML", "num_tracks"):
            totals[k] += m[k]

    if len(common) > 1:
        print("-" * len(header))
        overall_mota = 1.0 - (totals["FN"] + totals["FP"] + totals["IDs"]) / max(totals["GT"], 1)
        print(
            f"{'OVERALL':<40} {overall_mota:>7.3f} {'':>7} {'':>7}"
            f" {totals['MT']:>5d} {totals['ML']:>5d} {totals['FP']:>7d} {totals['FN']:>7d} {totals['IDs']:>5d}"
        )

    print()


if __name__ == "__main__":
    main()
