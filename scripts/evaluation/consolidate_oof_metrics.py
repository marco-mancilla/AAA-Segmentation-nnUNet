#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path


EXPECTED_FOLDS = tuple(range(5))
EXPECTED_CASES_PER_FOLD = 40
EXPECTED_TOTAL_CASES = 200

OUTPUT_COLUMNS = [
    "case",
    "fold",
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
]

CORE_REQUIRED_COLUMNS = {
    "case",
    "dice",
    "iou",
    "tp",
    "fp",
    "fn",
    "tn",
    "n_pred",
    "n_ref",
    "volume_difference",
    "volume_difference_pct",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Consolidate the five EXP01 fold validation CSV files "
            "into a single 200-case out-of-fold metrics table."
        )
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=repo_root / "results" / "metrics",
        help="Directory containing fold_<N>_metrics.csv files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "metrics" / "exp01_oof_metrics.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def case_number(case_id: str) -> int:
    try:
        return int(case_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return sys.maxsize


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_row(
    row: dict[str, str],
    expected_fold: int,
    source_name: str,
) -> dict[str, str]:

    missing = CORE_REQUIRED_COLUMNS.difference(row.keys())

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )

    case_id = row["case"].strip()

    if not case_id:
        raise ValueError(f"Empty case identifier found in {source_name}")

    # Validate fold if the source CSV already contains one.
    source_fold = row.get("fold", "").strip()

    if source_fold:
        try:
            parsed_fold = int(source_fold)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fold value for {case_id} in {source_name}: "
                f"{source_fold!r}"
            ) from exc

        if parsed_fold != expected_fold:
            raise ValueError(
                f"{case_id} in {source_name} reports fold {parsed_fold}, "
                f"expected {expected_fold}."
            )

    try:
        tp = float(row["tp"])
        fp = float(row["fp"])
        fn = float(row["fn"])
    except ValueError as exc:
        raise ValueError(
            f"Invalid TP/FP/FN values for {case_id} in {source_name}"
        ) from exc

    # Older Fold 0 exports may not contain precision/recall.
    # Reconstruct them deterministically from the confusion counts.
    precision_value = row.get("precision", "").strip()
    recall_value = row.get("recall", "").strip()

    if precision_value:
        precision = float(precision_value)
    else:
        precision = safe_divide(tp, tp + fp)

    if recall_value:
        recall = float(recall_value)
    else:
        recall = safe_divide(tp, tp + fn)

    normalized = {
        "case": case_id,
        "fold": str(expected_fold),
        "dice": row["dice"].strip(),
        "iou": row["iou"].strip(),
        "precision": f"{precision:.15g}",
        "recall": f"{recall:.15g}",
        "tp": row["tp"].strip(),
        "fp": row["fp"].strip(),
        "fn": row["fn"].strip(),
        "tn": row["tn"].strip(),
        "n_pred": row["n_pred"].strip(),
        "n_ref": row["n_ref"].strip(),
        "volume_difference": row["volume_difference"].strip(),
        "volume_difference_pct": row["volume_difference_pct"].strip(),
    }

    # Validate all numeric output columns.
    for column in OUTPUT_COLUMNS:
        if column in {"case", "fold"}:
            continue

        try:
            float(normalized[column])
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for '{column}' in "
                f"{case_id} from {source_name}: "
                f"{normalized[column]!r}"
            ) from exc

    return normalized


def load_fold(path: Path, expected_fold: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold metrics file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")

        raw_rows = list(reader)

    if len(raw_rows) != EXPECTED_CASES_PER_FOLD:
        raise ValueError(
            f"Fold {expected_fold} contains {len(raw_rows)} rows; "
            f"expected {EXPECTED_CASES_PER_FOLD}."
        )

    return [
        normalize_row(row, expected_fold, path.name)
        for row in raw_rows
    ]


def numeric_values(
    rows: list[dict[str, str]],
    column: str,
) -> list[float]:

    values: list[float] = []

    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value for '{column}' "
                f"in case {row.get('case', '<unknown>')}."
            ) from exc

    return values


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    args = parse_args()

    metrics_dir = args.metrics_dir.resolve()
    output_path = args.output.resolve()

    all_rows: list[dict[str, str]] = []

    print("=" * 72)
    print("EXP01 out-of-fold metrics consolidation")
    print("=" * 72)

    for fold in EXPECTED_FOLDS:
        fold_path = metrics_dir / f"fold_{fold}_metrics.csv"

        rows = load_fold(fold_path, fold)
        all_rows.extend(rows)

        print(f"Fold {fold}: {len(rows):3d} cases [OK]")

    if len(all_rows) != EXPECTED_TOTAL_CASES:
        raise ValueError(
            f"Combined table contains {len(all_rows)} rows; "
            f"expected {EXPECTED_TOTAL_CASES}."
        )

    case_ids = [row["case"] for row in all_rows]

    duplicates = sorted(
        case_id
        for case_id in set(case_ids)
        if case_ids.count(case_id) > 1
    )

    if duplicates:
        raise ValueError(
            "Duplicate OOF cases detected: "
            + ", ".join(duplicates)
        )

    expected_cases = {
        f"AAA_{i:03d}"
        for i in range(1, EXPECTED_TOTAL_CASES + 1)
    }

    actual_cases = set(case_ids)

    missing_cases = sorted(expected_cases - actual_cases)
    unexpected_cases = sorted(actual_cases - expected_cases)

    if missing_cases:
        raise ValueError(
            "Missing expected cases: "
            + ", ".join(missing_cases)
        )

    if unexpected_cases:
        raise ValueError(
            "Unexpected cases found: "
            + ", ".join(unexpected_cases)
        )

    # AAA_001 ... AAA_200
    all_rows.sort(
        key=lambda row: case_number(row["case"])
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(all_rows)

    dice = numeric_values(all_rows, "dice")
    iou = numeric_values(all_rows, "iou")

    dice_stats = describe(dice)
    iou_stats = describe(iou)

    lowest = sorted(
        all_rows,
        key=lambda row: float(row["dice"]),
    )[:10]

    print()
    print("=" * 72)
    print("EXP01 complete OOF validation")
    print("=" * 72)

    print(f"Cases:       {len(all_rows)}")
    print(f"Unique:      {len(set(case_ids))}")
    print(
        f"Coverage:    {len(all_rows)}/"
        f"{EXPECTED_TOTAL_CASES} (100%)"
    )

    print()
    print("Dice")
    print(f"  Mean:      {dice_stats['mean']:.10f}")
    print(f"  Median:    {dice_stats['median']:.10f}")
    print(f"  SD:        {dice_stats['sd']:.10f}")
    print(f"  Minimum:   {dice_stats['min']:.10f}")
    print(f"  Maximum:   {dice_stats['max']:.10f}")

    print()
    print("IoU")
    print(f"  Mean:      {iou_stats['mean']:.10f}")
    print(f"  Median:    {iou_stats['median']:.10f}")
    print(f"  SD:        {iou_stats['sd']:.10f}")
    print(f"  Minimum:   {iou_stats['min']:.10f}")
    print(f"  Maximum:   {iou_stats['max']:.10f}")

    print()
    print("Lowest 10 OOF Dice:")

    for row in lowest:
        print(
            f"  {row['case']} (fold {row['fold']}): "
            f"Dice={float(row['dice']):.6f}, "
            f"IoU={float(row['iou']):.6f}, "
            f"FP={row['fp']}, "
            f"FN={row['fn']}"
        )

    print()
    print("[OK] All five folds consolidated successfully.")
    print("[OK] 200 unique out-of-fold cases verified.")
    print("[OK] CSV written to:")
    print(f"  {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())