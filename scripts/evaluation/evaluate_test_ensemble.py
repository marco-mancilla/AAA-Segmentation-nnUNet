#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import numpy as np
import SimpleITK as sitk


EXPECTED_CASES_DEFAULT = 20

OUTPUT_COLUMNS = [
    "case",
    "dice",
    "iou",
    "precision",
    "recall",
    "tp",
    "fp",
    "fn",
    "tn",
    "n_pred",
    "n_ref",
    "volume_difference",
    "volume_difference_pct",
    "volume_pred_ml",
    "volume_ref_ml",
    "volume_difference_ml",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen EXP01 five-fold ensemble predictions against the "
            "original test reference masks supplied with the dataset."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing AAA_TEST_XXX.nii.gz ensemble predictions.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory containing AAA_TEST_XXX.nii.gz dataset reference masks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "metrics" / "exp01_test_ensemble_metrics.csv",
        help=(
            "Output CSV path. Default: "
            "results/metrics/exp01_test_ensemble_metrics.csv"
        ),
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=EXPECTED_CASES_DEFAULT,
        help=f"Expected number of test cases (default: {EXPECTED_CASES_DEFAULT}).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for geometry comparisons (default: 1e-6).",
    )
    return parser.parse_args()


def almost_equal_sequence(a, b, tol: float) -> bool:
    return (
        len(a) == len(b)
        and all(
            math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=tol)
            for x, y in zip(a, b)
        )
    )


def safe_divide(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def validate_binary_mask(arr: np.ndarray, case: str, kind: str) -> None:
    labels = set(np.unique(arr).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{case}: {kind} is not binary. Found labels: {sorted(labels)}"
        )


def geometry_matches(pred: sitk.Image, ref: sitk.Image, tol: float) -> bool:
    return (
        pred.GetSize() == ref.GetSize()
        and almost_equal_sequence(pred.GetSpacing(), ref.GetSpacing(), tol)
        and almost_equal_sequence(pred.GetOrigin(), ref.GetOrigin(), tol)
        and almost_equal_sequence(pred.GetDirection(), ref.GetDirection(), tol)
    )


def main() -> int:
    args = parse_args()

    predictions_dir = args.predictions_dir.resolve()
    ground_truth_dir = args.ground_truth_dir.resolve()
    output_path = args.output.resolve()

    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {ground_truth_dir}")

    prediction_paths = sorted(predictions_dir.glob("AAA_TEST_*.nii.gz"))
    gt_paths = sorted(ground_truth_dir.glob("AAA_TEST_*.nii.gz"))

    if len(prediction_paths) != args.expected_cases:
        raise ValueError(
            f"Found {len(prediction_paths)} predictions; "
            f"expected {args.expected_cases}."
        )
    if len(gt_paths) != args.expected_cases:
        raise ValueError(
            f"Found {len(gt_paths)} ground-truth masks; "
            f"expected {args.expected_cases}."
        )

    pred_by_name = {p.name: p for p in prediction_paths}
    gt_by_name = {p.name: p for p in gt_paths}

    if len(pred_by_name) != len(prediction_paths):
        raise ValueError("Duplicate prediction filenames detected.")
    if len(gt_by_name) != len(gt_paths):
        raise ValueError("Duplicate ground-truth filenames detected.")

    missing_gt = sorted(set(pred_by_name) - set(gt_by_name))
    missing_pred = sorted(set(gt_by_name) - set(pred_by_name))

    if missing_gt:
        raise ValueError(
            "Predictions without matching ground truth: " + ", ".join(missing_gt)
        )
    if missing_pred:
        raise ValueError(
            "Ground-truth masks without matching prediction: "
            + ", ".join(missing_pred)
        )

    rows: list[dict[str, object]] = []

    print("=" * 78)
    print("EXP01 test ensemble evaluation vs dataset reference")
    print("=" * 78)
    print(f"Predictions:   {predictions_dir}")
    print(f"Ground truth:  {ground_truth_dir}")
    print()

    for name in sorted(pred_by_name):
        pred_path = pred_by_name[name]
        gt_path = gt_by_name[name]
        case = name.removesuffix(".nii.gz")

        pred_img = sitk.ReadImage(str(pred_path))
        gt_img = sitk.ReadImage(str(gt_path))

        if not geometry_matches(pred_img, gt_img, args.tol):
            raise ValueError(
                f"{case}: prediction and reference geometry do not match. "
                "Evaluation stopped; no implicit resampling is performed."
            )

        pred_arr = sitk.GetArrayFromImage(pred_img)
        gt_arr = sitk.GetArrayFromImage(gt_img)

        validate_binary_mask(pred_arr, case, "prediction")
        validate_binary_mask(gt_arr, case, "ground truth")

        pred_fg = pred_arr == 1
        ref_fg = gt_arr == 1

        tp = int(np.logical_and(pred_fg, ref_fg).sum())
        fp = int(np.logical_and(pred_fg, ~ref_fg).sum())
        fn = int(np.logical_and(~pred_fg, ref_fg).sum())
        tn = int(np.logical_and(~pred_fg, ~ref_fg).sum())

        n_pred = int(pred_fg.sum())
        n_ref = int(ref_fg.sum())

        dice = safe_divide(2 * tp, 2 * tp + fp + fn)
        iou = safe_divide(tp, tp + fp + fn)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)

        volume_difference = n_pred - n_ref
        volume_difference_pct = (
            safe_divide(volume_difference, n_ref) * 100.0
            if n_ref != 0
            else 0.0
        )

        spacing = gt_img.GetSpacing()
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

        volume_pred_ml = n_pred * voxel_volume_mm3 / 1000.0
        volume_ref_ml = n_ref * voxel_volume_mm3 / 1000.0
        volume_difference_ml = volume_pred_ml - volume_ref_ml

        rows.append(
            {
                "case": case,
                "dice": f"{dice:.15g}",
                "iou": f"{iou:.15g}",
                "precision": f"{precision:.15g}",
                "recall": f"{recall:.15g}",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_pred": n_pred,
                "n_ref": n_ref,
                "volume_difference": volume_difference,
                "volume_difference_pct": f"{volume_difference_pct:.15g}",
                "volume_pred_ml": f"{volume_pred_ml:.15g}",
                "volume_ref_ml": f"{volume_ref_ml:.15g}",
                "volume_difference_ml": f"{volume_difference_ml:.15g}",
            }
        )

        print(
            f"{case}: Dice={dice:.6f}, IoU={iou:.6f}, "
            f"FP={fp}, FN={fn}, ΔV={volume_difference_pct:+.2f}%"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    dice_values = [float(row["dice"]) for row in rows]
    iou_values = [float(row["iou"]) for row in rows]
    precision_values = [float(row["precision"]) for row in rows]
    recall_values = [float(row["recall"]) for row in rows]
    volume_pct_values = [float(row["volume_difference_pct"]) for row in rows]

    dice_stats = describe(dice_values)
    iou_stats = describe(iou_values)

    lowest = sorted(rows, key=lambda row: float(row["dice"]))[:5]

    print()
    print("=" * 78)
    print("EXP01 test ensemble summary")
    print("=" * 78)
    print(f"Cases:      {len(rows)}")
    print()
    print("Dice")
    print(f"  Mean:     {dice_stats['mean']:.10f}")
    print(f"  Median:   {dice_stats['median']:.10f}")
    print(f"  SD:       {dice_stats['sd']:.10f}")
    print(f"  Minimum:  {dice_stats['min']:.10f}")
    print(f"  Maximum:  {dice_stats['max']:.10f}")
    print()
    print("IoU")
    print(f"  Mean:     {iou_stats['mean']:.10f}")
    print(f"  Median:   {iou_stats['median']:.10f}")
    print(f"  SD:       {iou_stats['sd']:.10f}")
    print(f"  Minimum:  {iou_stats['min']:.10f}")
    print(f"  Maximum:  {iou_stats['max']:.10f}")
    print()
    print(f"Mean precision:          {statistics.fmean(precision_values):.10f}")
    print(f"Mean recall:             {statistics.fmean(recall_values):.10f}")
    print(
        f"Mean volume difference:  "
        f"{statistics.fmean(volume_pct_values):+.4f}%"
    )
    print()
    print("Lowest 5 Dice:")
    for row in lowest:
        print(
            f"  {row['case']}: "
            f"Dice={float(row['dice']):.6f}, "
            f"IoU={float(row['iou']):.6f}, "
            f"FP={row['fp']}, FN={row['fn']}, "
            f"ΔV={float(row['volume_difference_pct']):+.2f}%"
        )

    print()
    print("[OK] Evaluation completed.")
    print("[OK] No resampling or post-hoc correction was applied.")
    print("[OK] CSV written to:")
    print(f"  {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
