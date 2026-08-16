#!/usr/bin/env python3
"""
Prepare the public AAA Figshare dataset as Dataset001_AAA for nnU-Net v2.

Experiment subset:
- pre-operative contrast-enhanced CT only
- 200 source training cases from AAA_Train
- 20 source test cases from AAA_Test

The source data are NEVER modified. NRRD files are read with SimpleITK and
written as compressed NIfTI (.nii.gz) into the nnU-Net workspace.

Important:
The source folders reuse some CE identifiers across train/test. This script
reports those identifier overlaps and performs content checks for overlapping
IDs. If an image pair is byte-identical or voxel-identical across train/test,
the script stops by default to protect against accidental data leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


DATASET_NAME = "Dataset001_AAA"
TRAIN_EXPECTED = 200
TEST_EXPECTED = 20
CASE_RE = re.compile(r"^CE(\d+)_CE\.nrrd$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def voxel_hash(image: sitk.Image) -> str:
    """Hash voxel content plus geometry, independent of the source file container."""
    array = sitk.GetArrayFromImage(image)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(array).tobytes())
    h.update(str(tuple(image.GetSize())).encode())
    h.update(str(tuple(image.GetSpacing())).encode())
    h.update(str(tuple(image.GetOrigin())).encode())
    h.update(str(tuple(image.GetDirection())).encode())
    return h.hexdigest()


def numeric_id(path: Path) -> int:
    match = CASE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected CE filename: {path.name}")
    return int(match.group(1))


def source_id(path: Path) -> str:
    return f"CE{numeric_id(path):03d}"


def find_ce_images(folder: Path) -> list[Path]:
    candidates = [p for p in folder.glob("CE*_CE.nrrd") if CASE_RE.match(p.name)]
    return sorted(candidates, key=numeric_id)


def label_for(image_path: Path) -> Path:
    sid = source_id(image_path)
    return image_path.with_name(f"{sid}_gt-label.nrrd")


def allclose_tuple(a: Iterable[float], b: Iterable[float], atol: float = 1e-5) -> bool:
    return bool(np.allclose(np.asarray(tuple(a), dtype=float), np.asarray(tuple(b), dtype=float), atol=atol))


def validate_pair(image: sitk.Image, label: sitk.Image, case_name: str) -> tuple[np.ndarray, list[int]]:
    problems = []

    if image.GetSize() != label.GetSize():
        problems.append(f"size image={image.GetSize()} label={label.GetSize()}")
    if not allclose_tuple(image.GetSpacing(), label.GetSpacing()):
        problems.append(f"spacing image={image.GetSpacing()} label={label.GetSpacing()}")
    if not allclose_tuple(image.GetOrigin(), label.GetOrigin()):
        problems.append(f"origin image={image.GetOrigin()} label={label.GetOrigin()}")
    if not allclose_tuple(image.GetDirection(), label.GetDirection()):
        problems.append("direction differs")

    label_np = sitk.GetArrayFromImage(label)
    unique = sorted(int(x) for x in np.unique(label_np).tolist())

    if set(unique) != {0, 1}:
        problems.append(f"expected labels [0, 1], found {unique}")
    if int(np.count_nonzero(label_np == 1)) == 0:
        problems.append("foreground label 1 is empty")

    if problems:
        raise ValueError(f"{case_name}: " + "; ".join(problems))

    return label_np, unique


def resolve_dataset_root(path: Path) -> Path:
    root = path.expanduser().resolve()

    if (root / "AAA_Train").is_dir() and (root / "AAA_Test").is_dir():
        return root

    matches = []
    if root.is_dir():
        for candidate in root.rglob("AAA_Train"):
            parent = candidate.parent
            if (parent / "AAA_Test").is_dir():
                matches.append(parent)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        raise FileNotFoundError(
            f"Could not find AAA_Train and AAA_Test below: {root}\n"
            "Pass --dataset-root pointing to the extracted Figshare dataset "
            "or to a parent directory that contains it."
        )

    formatted = "\n".join(f"  - {p}" for p in unique_matches)
    raise RuntimeError(
        "Multiple candidate dataset roots were found. "
        "Please pass the exact one with --dataset-root:\n" + formatted
    )


def default_workspace() -> Path:
    env_raw = os.environ.get("nnUNet_raw")
    if env_raw:
        return Path(env_raw).expanduser().resolve().parent
    return Path(__file__).resolve().parents[2].parent / "AAA-Segmentation-workspace"


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\n"
                "Use --overwrite only if you intentionally want to rebuild Dataset001_AAA."
            )
        shutil.rmtree(path)


def compare_overlapping_ids(
    train_by_id: dict[str, Path],
    test_by_id: dict[str, Path],
) -> list[dict]:
    overlap = sorted(set(train_by_id) & set(test_by_id))
    report = []

    for sid in overlap:
        train_img_path = train_by_id[sid]
        test_img_path = test_by_id[sid]
        train_lbl_path = label_for(train_img_path)
        test_lbl_path = label_for(test_img_path)

        train_img = sitk.ReadImage(str(train_img_path))
        test_img = sitk.ReadImage(str(test_img_path))
        train_lbl = sitk.ReadImage(str(train_lbl_path))
        test_lbl = sitk.ReadImage(str(test_lbl_path))

        record = {
            "source_id": sid,
            "image_file_sha256_equal": sha256_file(train_img_path) == sha256_file(test_img_path),
            "label_file_sha256_equal": sha256_file(train_lbl_path) == sha256_file(test_lbl_path),
            "image_voxel_geometry_hash_equal": voxel_hash(train_img) == voxel_hash(test_img),
            "label_voxel_geometry_hash_equal": voxel_hash(train_lbl) == voxel_hash(test_lbl),
        }
        report.append(record)

    return report


def write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    data = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "AAA": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def write_mapping_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def convert_case(
    image_path: Path,
    label_path: Path,
    out_image: Path,
    out_label: Path,
    case_name: str,
) -> dict:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    label_np, unique = validate_pair(image, label, case_name)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(image, str(out_image), useCompression=True)
    sitk.WriteImage(label, str(out_label), useCompression=True)

    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "labels": unique,
        "foreground_voxels": int(np.count_nonzero(label_np == 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the contrast-enhanced pre-operative AAA Figshare subset to nnU-Net v2."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Extracted dataset directory containing AAA_Train and AAA_Test, "
             "or a parent directory containing that structure.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace created by scripts/setup_workspace.py. "
             "If omitted, infer it from $nnUNet_raw when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild Dataset001_AAA if the output directory already contains files.",
    )
    parser.add_argument(
        "--no-strict-counts",
        action="store_true",
        help=f"Do not require exactly {TRAIN_EXPECTED} CE train and {TEST_EXPECTED} CE test cases.",
    )
    parser.add_argument(
        "--allow-identical-overlap",
        action="store_true",
        help="Continue even if an overlapping source ID is identical across train/test. "
             "Not recommended; intended only for deliberate methodological investigation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    workspace = (
        args.workspace.expanduser().resolve()
        if args.workspace is not None
        else default_workspace().resolve()
    )

    train_dir = dataset_root / "AAA_Train"
    test_dir = dataset_root / "AAA_Test"

    train_images = find_ce_images(train_dir)
    test_images = find_ce_images(test_dir)

    print("=" * 72)
    print("AAA dataset preparation for nnU-Net v2")
    print("=" * 72)
    print(f"Source dataset: {dataset_root}")
    print(f"Workspace:      {workspace}")
    print(f"CE train cases: {len(train_images)}")
    print(f"CE test cases:  {len(test_images)}")

    if not args.no_strict_counts:
        if len(train_images) != TRAIN_EXPECTED:
            raise RuntimeError(
                f"Expected {TRAIN_EXPECTED} CE training cases, found {len(train_images)}."
            )
        if len(test_images) != TEST_EXPECTED:
            raise RuntimeError(
                f"Expected {TEST_EXPECTED} CE test cases, found {len(test_images)}."
            )

    missing = []
    for image_path in train_images + test_images:
        label_path = label_for(image_path)
        if not label_path.is_file():
            missing.append(str(label_path))
    if missing:
        raise FileNotFoundError(
            "Missing segmentation labels:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    train_by_id = {source_id(p): p for p in train_images}
    test_by_id = {source_id(p): p for p in test_images}
    overlap_report = compare_overlapping_ids(train_by_id, test_by_id)

    print()
    print("Source-ID overlap check")
    print("-" * 72)
    if overlap_report:
        for item in overlap_report:
            print(
                f"{item['source_id']}: "
                f"image-file={item['image_file_sha256_equal']} "
                f"label-file={item['label_file_sha256_equal']} "
                f"image-content={item['image_voxel_geometry_hash_equal']} "
                f"label-content={item['label_voxel_geometry_hash_equal']}"
            )
    else:
        print("No source identifiers overlap between CE train and CE test.")

    dangerous = [
        item for item in overlap_report
        if item["image_file_sha256_equal"]
        or item["image_voxel_geometry_hash_equal"]
        or item["label_file_sha256_equal"]
        or item["label_voxel_geometry_hash_equal"]
    ]
    if dangerous and not args.allow_identical_overlap:
        raise RuntimeError(
            "Potential train/test leakage detected: one or more overlapping source IDs "
            "contain identical image or label content. Review the overlap report before "
            "using the test set. To override intentionally, use --allow-identical-overlap."
        )

    nnunet_raw = workspace / "nnUNet_raw"
    dataset_dir = nnunet_raw / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    test_gt = workspace / "data" / "test_ground_truth"
    report_dir = workspace / "data" / "reports"

    ensure_empty_or_overwrite(dataset_dir, args.overwrite)
    if test_gt.exists() and any(test_gt.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Test ground-truth directory is not empty: {test_gt}\n"
                "Use --overwrite only if you intentionally want to rebuild it."
            )
        shutil.rmtree(test_gt)

    for directory in (images_tr, labels_tr, images_ts, test_gt, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    case_reports = []

    print()
    print("Converting CE training cases")
    print("-" * 72)
    for index, image_path in enumerate(train_images, start=1):
        sid = source_id(image_path)
        nn_id = f"AAA_{index:03d}"
        label_path = label_for(image_path)

        stats = convert_case(
            image_path,
            label_path,
            images_tr / f"{nn_id}_0000.nii.gz",
            labels_tr / f"{nn_id}.nii.gz",
            nn_id,
        )

        mapping_rows.append({
            "split": "train",
            "source_id": sid,
            "nnunet_id": nn_id,
            "source_image": f"AAA_Train/{image_path.name}",
            "source_label": f"AAA_Train/{label_path.name}",
        })
        case_reports.append({"split": "train", "source_id": sid, "nnunet_id": nn_id, **stats})
        print(
            f"[{index:03d}/{len(train_images):03d}] {sid} -> {nn_id} | "
            f"size={tuple(stats['size_xyz'])} labels={stats['labels']}"
        )

    print()
    print("Converting CE test cases")
    print("-" * 72)
    for index, image_path in enumerate(test_images, start=1):
        sid = source_id(image_path)
        nn_id = f"AAA_TEST_{index:03d}"
        label_path = label_for(image_path)

        # Validate and write test image + held-out reference.
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        label_np, unique = validate_pair(image, label, nn_id)

        sitk.WriteImage(image, str(images_ts / f"{nn_id}_0000.nii.gz"), useCompression=True)
        sitk.WriteImage(label, str(test_gt / f"{nn_id}.nii.gz"), useCompression=True)

        stats = {
            "size_xyz": list(image.GetSize()),
            "spacing_xyz": list(image.GetSpacing()),
            "labels": unique,
            "foreground_voxels": int(np.count_nonzero(label_np == 1)),
        }
        mapping_rows.append({
            "split": "test",
            "source_id": sid,
            "nnunet_id": nn_id,
            "source_image": f"AAA_Test/{image_path.name}",
            "source_label": f"AAA_Test/{label_path.name}",
        })
        case_reports.append({"split": "test", "source_id": sid, "nnunet_id": nn_id, **stats})
        print(
            f"[{index:03d}/{len(test_images):03d}] {sid} -> {nn_id} | "
            f"size={tuple(stats['size_xyz'])} labels={stats['labels']}"
        )

    write_dataset_json(dataset_dir, len(train_images))
    write_mapping_csv(report_dir / "case_mapping.csv", mapping_rows)

    report = {
        "dataset_name": DATASET_NAME,
        "source_dataset_root": str(dataset_root),
        "subset": "pre-operative contrast-enhanced CT",
        "train_cases": len(train_images),
        "test_cases": len(test_images),
        "source_id_overlap": overlap_report,
        "outputs": {
            "dataset_dir": str(dataset_dir),
            "test_ground_truth": str(test_gt),
            "mapping_csv": str(report_dir / "case_mapping.csv"),
        },
        "cases": case_reports,
    }
    (report_dir / "aaa_dataset_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("Dataset preparation complete")
    print("=" * 72)
    print(f"nnU-Net dataset:   {dataset_dir}")
    print(f"Training images:   {len(list(images_tr.glob('*_0000.nii.gz')))}")
    print(f"Training labels:   {len(list(labels_tr.glob('*.nii.gz')))}")
    print(f"Test images:       {len(list(images_ts.glob('*_0000.nii.gz')))}")
    print(f"Test ground truth: {len(list(test_gt.glob('*.nii.gz')))}")
    print(f"Report:            {report_dir / 'aaa_dataset_report.json'}")
    print(f"Mapping:           {report_dir / 'case_mapping.csv'}")
    print()
    print("Next:")
    print("  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
