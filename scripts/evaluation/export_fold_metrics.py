#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

DEFAULT_DATASET = "Dataset001_AAA"
DEFAULT_TRAINER = "nnUNetTrainer"
DEFAULT_PLANS = "nnUNetPlans"
DEFAULT_CONFIGURATION = "3d_fullres"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            "Load the nnU-Net environment before running this script."
        )
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export validation metrics from one nnU-Net fold to CSV."
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def case_id_from_path(path_value: str) -> str:
    name = Path(path_value).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def main() -> int:
    args = parse_args()

    nnunet_results = require_env_path("nnUNet_results")
    experiment_dir = (
        nnunet_results
        / args.dataset
        / f"{args.trainer}__{args.plans}__{args.configuration}"
    )
    summary_path = experiment_dir / f"fold_{args.fold}" / "validation" / "summary.json"

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Validation summary not found:\n  {summary_path}\n\n"
            f"Make sure Fold {args.fold} completed validation."
        )

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    cases = summary.get("metric_per_case")
    if not cases:
        raise RuntimeError("summary.json does not contain metric_per_case.")

    rows = []

    for item in cases:
        metrics = item.get("metrics", {})
        foreground = metrics.get("1") or metrics.get("foreground")
        if foreground is None:
            raise KeyError("Could not find foreground metrics for label 1.")

        reference_file = item.get("reference_file", "")
        prediction_file = item.get("prediction_file", "")
        case = case_id_from_path(reference_file or prediction_file)

        n_pred = foreground.get("n_pred")
        n_ref = foreground.get("n_ref")

        volume_difference = None
        volume_difference_pct = None
        if n_pred is not None and n_ref is not None:
            volume_difference = n_pred - n_ref
            if n_ref != 0:
                volume_difference_pct = 100.0 * volume_difference / n_ref

        tp = foreground.get("TP")
        fp = foreground.get("FP")
        fn = foreground.get("FN")
        tn = foreground.get("TN")

        precision = None
        recall = None
        if tp is not None and fp is not None and (tp + fp) > 0:
            precision = tp / (tp + fp)
        if tp is not None and fn is not None and (tp + fn) > 0:
            recall = tp / (tp + fn)

        rows.append({
            "case": case,
            "fold": args.fold,
            "dice": foreground.get("Dice"),
            "iou": foreground.get("IoU"),
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n_pred": n_pred,
            "n_ref": n_ref,
            "volume_difference": volume_difference,
            "volume_difference_pct": volume_difference_pct,
        })

    rows.sort(key=lambda r: (float("inf") if r["dice"] is None else r["dice"], r["case"]))

    output = (
        args.output.expanduser().resolve()
        if args.output
        else repo_root() / "results" / "metrics" / f"fold_{args.fold}_metrics.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dice_values = [r["dice"] for r in rows if r["dice"] is not None]

    print("=" * 72)
    print(f"Fold {args.fold} validation metrics")
    print("=" * 72)
    print(f"Cases:   {len(rows)}")
    print(f"Mean:    {statistics.mean(dice_values):.10f}")
    print(f"Median:  {statistics.median(dice_values):.10f}")
    if len(dice_values) > 1:
        print(f"SD:      {statistics.stdev(dice_values):.10f}")
    print(f"Minimum: {min(dice_values):.10f}")
    print(f"Maximum: {max(dice_values):.10f}")
    print()
    print("Lowest 10 Dice:")
    for row in rows[:10]:
        print(
            f"  {row['case']}: Dice={row['dice']:.6f}, "
            f"IoU={row['iou']:.6f}, FP={row['fp']}, FN={row['fn']}"
        )
    print()
    print(f"[OK] CSV written to:\n  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
