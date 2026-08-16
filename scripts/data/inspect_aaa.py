#!/usr/bin/env python3
"""
Inspect and validate one AAA image/segmentation pair.

Works with any image format supported by SimpleITK (including NRRD and NIfTI).
It is useful before dataset conversion to confirm that image and label geometry
match and that the segmentation contains the expected labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def close_tuple(a, b, atol: float = 1e-5) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=atol))


def geometry_report(image: sitk.Image, label: sitk.Image) -> dict:
    return {
        "same_size": image.GetSize() == label.GetSize(),
        "same_spacing": close_tuple(image.GetSpacing(), label.GetSpacing()),
        "same_origin": close_tuple(image.GetOrigin(), label.GetOrigin()),
        "same_direction": close_tuple(image.GetDirection(), label.GetDirection()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one CT image and segmentation mask used by the AAA dataset."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to CT image (.nrrd/.nii.gz/etc.).")
    parser.add_argument("--label", required=True, type=Path, help="Path to segmentation label.")
    parser.add_argument(
        "--expected-labels",
        default="0,1",
        help="Comma-separated expected label values (default: 0,1).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        help="Optional path where the inspection report will be written as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    label_path = args.label.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")

    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))

    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    unique_labels, counts = np.unique(label_np, return_counts=True)
    label_counts = {
        str(int(v) if float(v).is_integer() else float(v)): int(c)
        for v, c in zip(unique_labels.astype(float), counts)
    }

    expected = {float(x.strip()) for x in args.expected_labels.split(",") if x.strip()}
    observed = {float(x) for x in unique_labels.tolist()}

    geometry = geometry_report(image, label)

    report = {
        "image": str(image_path),
        "label": str(label_path),
        "image_size_xyz": list(image.GetSize()),
        "label_size_xyz": list(label.GetSize()),
        "image_array_shape_zyx": list(image_np.shape),
        "label_array_shape_zyx": list(label_np.shape),
        "image_spacing_xyz": list(image.GetSpacing()),
        "label_spacing_xyz": list(label.GetSpacing()),
        "image_origin_xyz": list(image.GetOrigin()),
        "label_origin_xyz": list(label.GetOrigin()),
        "image_direction": list(image.GetDirection()),
        "label_direction": list(label.GetDirection()),
        "image_intensity": {
            "min": float(np.min(image_np)),
            "max": float(np.max(image_np)),
            "mean": float(np.mean(image_np)),
            "std": float(np.std(image_np)),
        },
        "labels": sorted(float(x) for x in observed),
        "label_counts": label_counts,
        "foreground_voxels": int(np.count_nonzero(label_np)),
        "geometry": geometry,
        "expected_labels": sorted(expected),
        "labels_match_expected": observed == expected,
    }

    print("=" * 72)
    print("AAA image/label inspection")
    print("=" * 72)
    print(f"Image:      {image_path}")
    print(f"Label:      {label_path}")
    print(f"Size XYZ:   image={image.GetSize()} label={label.GetSize()}")
    print(f"Spacing:    image={image.GetSpacing()} label={label.GetSpacing()}")
    print(f"Origin:     image={image.GetOrigin()} label={label.GetOrigin()}")
    print(f"Array ZYX:  image={image_np.shape} label={label_np.shape}")
    print(f"Labels:     {sorted(observed)}")
    print(f"Label voxels: {label_counts}")
    print(f"Foreground voxels: {report['foreground_voxels']}")
    print(
        "Intensity: "
        f"min={report['image_intensity']['min']:.3f}, "
        f"max={report['image_intensity']['max']:.3f}, "
        f"mean={report['image_intensity']['mean']:.3f}, "
        f"std={report['image_intensity']['std']:.3f}"
    )
    print("Geometry:")
    for key, value in geometry.items():
        print(f"  {key}: {value}")
    print(f"Labels match expected {sorted(expected)}: {report['labels_match_expected']}")

    valid_geometry = all(geometry.values())
    valid_labels = report["labels_match_expected"] and report["foreground_voxels"] > 0

    if args.json_output:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[OK] JSON report written to: {output}")

    if not valid_geometry:
        print("[ERROR] Image and label geometry do not match.")
        return 2
    if not valid_labels:
        print("[ERROR] Label values are unexpected or the foreground is empty.")
        return 3

    print("[OK] Pair passed the basic inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
