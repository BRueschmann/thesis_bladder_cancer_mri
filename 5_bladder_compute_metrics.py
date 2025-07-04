#!/usr/bin/env python3
"""
Compute Dice and HD95 for binary bladder-tumour masks
and optionally list which header field (SIZE / SPACING / ORIGIN /
DIRECTION) forces each prediction to be resampled.

Usage examples
--------------
# just the metrics, no extra output
python 5_compute_metrics.py --pred_dir PRED --gt_dir GT

# as above but show first 30 header mismatches
python 5_compute_metrics.py --pred_dir PRED --gt_dir GT \
        --show-header-diffs --diff-limit 30
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm




# ───────────────────────── CLI ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", required=True,
                    help="Folder with <case>.nii.gz predictions")
parser.add_argument("--gt_dir",   required=True,
                    help="Root folder with <case>/seg.nii.gz ground-truth")
parser.add_argument("--out_dir",  default=None,
                    help="Output folder for the .npy metrics "
                         "(default: ../predictions/result_metrics)")
parser.add_argument("--inspect",  type=int, default=0,
                    metavar="N", help="Print full headers for first N cases")

# NEW: list header differences that cause resampling
parser.add_argument("--show-header-diffs", action="store_true",
                    help="Print the first diff-limit mismatches (size/spacing/origin/direction)")
parser.add_argument("--diff-limit", type=int, default=20,
                    help="How many mismatching cases to list (default 20)")
args = parser.parse_args()


# ────────────────────── helper fns ──────────────────────
def to_bin(a): return (a > 0).astype(np.uint8)

def dice_hd95(gt, pred, spacing):
    d = metric.binary.dc(pred, gt)
    try:
        h = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    except RuntimeError:
        h = np.inf
    return d, h

def resample_to_ref(mov, ref):
    return sitk.Resample(mov, ref, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def hdr(img):
    return (img.GetSize(), img.GetSpacing(),
            img.GetOrigin(), img.GetDirection())




# ───────────────────── path handling ────────────────────
pred_root = Path(args.pred_dir).resolve()
gt_root   = Path(args.gt_dir).resolve()
if not pred_root.is_dir():
    raise FileNotFoundError(pred_root)
if not gt_root.is_dir():
    raise FileNotFoundError(gt_root)

out_dir = (Path(args.out_dir) if args.out_dir
           else pred_root.parent / "result_metrics")
out_dir.mkdir(parents=True, exist_ok=True)

# extract pure case IDs (strip .nii.gz)
case_ids = sorted(
    Path(p).name.removesuffix(".nii.gz")  # Python ≥3.9
    for p in glob.glob(str(pred_root / "*.nii.gz"))
)
if not case_ids:
    raise RuntimeError(f"No predictions found in {pred_root}")

results      = np.zeros((len(case_ids), 1, 2), dtype=np.float32)
cnt_resample = 0
hdr_issues   = []         # (case, reason) pairs for optional listing

# ───────────────────────── loop ─────────────────────────
for i, case in enumerate(tqdm(case_ids, desc="Cases")):
    gt_path   = gt_root / case / "seg.nii.gz"
    pred_path = pred_root / f"{case}.nii.gz"

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    gt_img   = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))

    if args.inspect and i < args.inspect:
        print(f"\n[{case}] GT hdr : {hdr(gt_img)}")
        print(f"[{case}] PR hdr : {hdr(pred_img)}")

    if hdr(gt_img) != hdr(pred_img):
        if args.show_header_diffs:
            reason = hdr_mismatch(pred_img, gt_img)
            hdr_issues.append((case, reason))
        pred_img  = resample_to_ref(pred_img, gt_img)
        cnt_resample += 1
        assert hdr(gt_img) == hdr(pred_img), \
            f"Header mismatch persists after resample for {case}"

    gt_arr   = to_bin(sitk.GetArrayFromImage(gt_img))
    pred_arr = to_bin(sitk.GetArrayFromImage(pred_img))
    spacing  = gt_img.GetSpacing()[::-1]              # z,y,x order

    results[i, 0, :] = dice_hd95(gt_arr, pred_arr, spacing)

# ─────────────────── save + report ──────────────────────
np_path = out_dir / f"{pred_root.name}.npy"
np.save(np_path, results)

print(f"\nSaved per-case metrics → {np_path}")
print(f"Cases evaluated        : {len(case_ids)}")
print(f"Cases resampled        : {cnt_resample}")
print(f"Mean Dice              : {results[:,0,0].mean():.4f}")
print(f"Mean HD95 (mm)         : {results[:,0,1].mean():.2f}")
