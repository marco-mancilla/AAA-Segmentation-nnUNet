#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import SimpleITK as sitk


EXPECTED_CASES_DEFAULT = 20
ALLOWED_LABELS = {0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate EXP01 test ensemble predictions without using test ground truth."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--expected-cases", type=int, default=EXPECTED_CASES_DEFAULT)
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def almost_equal_sequence(a, b, tol: float) -> bool:
    return (
        len(a) == len(b)
        and all(
            math.isclose(float(x), float(y), rel_tol=0.0, abs_tol=tol)
            for x, y in zip(a, b)
        )
    )


def case_id_from_image(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected test image filename: {path.name}")
    return path.name[:-len(suffix)]


def main() -> int:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    predictions_dir = args.predictions_dir.resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    report_path = (
        args.report.resolve()
        if args.report
        else predictions_dir / "EXP01_5fold_ensemble_validation.csv"
    )
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else predictions_dir / "EXP01_5fold_ensemble_SHA256.csv"
    )

    image_paths = sorted(images_dir.glob("AAA_TEST_*_0000.nii.gz"))
    if len(image_paths) != args.expected_cases:
        raise ValueError(
            f"Found {len(image_paths)} test images; expected {args.expected_cases}."
        )

    case_ids = [case_id_from_image(p) for p in image_paths]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate test case identifiers detected.")

    expected_prediction_names = {f"{case_id}.nii.gz" for case_id in case_ids}
    prediction_paths = sorted(predictions_dir.glob("AAA_TEST_*.nii.gz"))
    actual_prediction_names = {p.name for p in prediction_paths}

    missing = sorted(expected_prediction_names - actual_prediction_names)
    unexpected = sorted(actual_prediction_names - expected_prediction_names)

    if missing:
        raise ValueError("Missing predictions: " + ", ".join(missing))
    if unexpected:
        raise ValueError("Unexpected predictions: " + ", ".join(unexpected))
    if len(prediction_paths) != args.expected_cases:
        raise ValueError(
            f"Found {len(prediction_paths)} predictions; expected {args.expected_cases}."
        )

    rows = []
    failures = []

    print("=" * 78)
    print("EXP01 test prediction validation")
    print("=" * 78)

    for image_path in image_paths:
        case_id = case_id_from_image(image_path)
        prediction_path = predictions_dir / f"{case_id}.nii.gz"

        image = sitk.ReadImage(str(image_path))
        prediction = sitk.ReadImage(str(prediction_path))

        size_match = image.GetSize() == prediction.GetSize()
        spacing_match = almost_equal_sequence(
            image.GetSpacing(), prediction.GetSpacing(), args.tol
        )
        origin_match = almost_equal_sequence(
            image.GetOrigin(), prediction.GetOrigin(), args.tol
        )
        direction_match = almost_equal_sequence(
            image.GetDirection(), prediction.GetDirection(), args.tol
        )
        geometry_match = all(
            [size_match, spacing_match, origin_match, direction_match]
        )

        arr = sitk.GetArrayViewFromImage(prediction)
        labels = sorted(int(v) for v in set(arr.ravel().tolist()))
        binary = set(labels).issubset(ALLOWED_LABELS)
        foreground_voxels = int((arr == 1).sum())
        nonempty = foreground_voxels > 0

        spacing = prediction.GetSpacing()
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        segmented_volume_ml = foreground_voxels * voxel_volume_mm3 / 1000.0
        digest = sha256_file(prediction_path)

        if not geometry_match:
            failures.append(f"{case_id}: geometry mismatch")
        if not binary:
            failures.append(f"{case_id}: non-binary labels {labels}")
        if not nonempty:
            failures.append(f"{case_id}: empty foreground")

        rows.append(
            {
                "case": case_id,
                "image_file": image_path.name,
                "prediction_file": prediction_path.name,
                "sha256": digest,
                "size_match": size_match,
                "spacing_match": spacing_match,
                "origin_match": origin_match,
                "direction_match": direction_match,
                "geometry_match": geometry_match,
                "labels": ";".join(str(v) for v in labels),
                "binary": binary,
                "nonempty": nonempty,
                "foreground_voxels": foreground_voxels,
                "segmented_volume_ml": f"{segmented_volume_ml:.10f}",
            }
        )

        status = "OK" if geometry_match and binary and nonempty else "FAIL"
        print(
            f"{case_id}: {status} | labels={labels} | "
            f"foreground={foreground_voxels} | volume={segmented_volume_ml:.3f} mL"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_columns = [
        "case",
        "image_file",
        "prediction_file",
        "sha256",
        "size_match",
        "spacing_match",
        "origin_match",
        "direction_match",
        "geometry_match",
        "labels",
        "binary",
        "nonempty",
        "foreground_voxels",
        "segmented_volume_ml",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "prediction_file", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case": row["case"],
                    "prediction_file": row["prediction_file"],
                    "sha256": row["sha256"],
                }
            )

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"Cases checked:   {len(rows)}")
    print(f"Geometry OK:     {sum(bool(r['geometry_match']) for r in rows)}/{len(rows)}")
    print(f"Binary masks:    {sum(bool(r['binary']) for r in rows)}/{len(rows)}")
    print(f"Non-empty masks: {sum(bool(r['nonempty']) for r in rows)}/{len(rows)}")
    print(f"Report:          {report_path}")
    print(f"Manifest:        {manifest_path}")

    if failures:
        print()
        print("[ERROR] Validation failures detected:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print()
    print("[OK] All test predictions passed structural validation.")
    print("[OK] No test ground-truth labels were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
