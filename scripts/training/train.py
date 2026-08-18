#!/usr/bin/env python3
"""
Cross-platform training launcher for AAA-Segmentation-nnUNet.

Validates the local nnU-Net environment and exact cross-validation split
before delegating training to nnUNetv2_train.

Examples:
    python scripts/training/train.py --fold 1
    python scripts/training/train.py --fold 1 --resume
    python scripts/training/train.py --fold 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = "Dataset001_AAA"
DEFAULT_CONFIGURATION = "3d_fullres"
VALID_FOLDS = (0, 1, 2, 3, 4)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set.\n"
            "Load the nnU-Net environment for your workspace before training."
        )
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} points to a non-existing path: {path}")
    return path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_exact_split(dataset: str, preprocessed_root: Path) -> Path:
    repo_split = repo_root() / "configs" / dataset / "splits_final.json"
    workspace_split = preprocessed_root / dataset / "splits_final.json"

    if not repo_split.is_file():
        raise FileNotFoundError(
            f"Reference split file is missing from the repository:\n  {repo_split}"
        )

    if not workspace_split.is_file():
        workspace = preprocessed_root.parent
        raise FileNotFoundError(
            f"Workspace split file is missing:\n  {workspace_split}\n\n"
            "To reproduce the exact EXP01 fold membership, run:\n"
            f'  python scripts/setup_workspace.py --workspace "{workspace}" --sync-splits'
        )

    if load_json(repo_split) != load_json(workspace_split):
        raise RuntimeError(
            "The workspace splits_final.json does not match the version-controlled "
            "EXP01 split.\n"
            f"Repository: {repo_split}\n"
            f"Workspace:  {workspace_split}\n\n"
            "Training was stopped to avoid silently using different fold membership."
        )

    return workspace_split


def validate_dataset(dataset: str, preprocessed_root: Path) -> Path:
    dataset_dir = preprocessed_root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed dataset not found:\n  {dataset_dir}\n\n"
            "Run nnUNetv2_plan_and_preprocess before training."
        )
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and launch one nnU-Net fold for the AAA baseline experiment."
    )
    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=VALID_FOLDS,
        help="Cross-validation fold to train (0-4).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"nnU-Net dataset name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--configuration",
        default=DEFAULT_CONFIGURATION,
        help=f"nnU-Net configuration (default: {DEFAULT_CONFIGURATION}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint by passing --c to nnUNetv2_train.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate everything and print the command without starting training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    trainer_exe = shutil.which("nnUNetv2_train")
    if trainer_exe is None:
        raise RuntimeError(
            "nnUNetv2_train was not found on PATH.\n"
            "Activate the Python environment where nnU-Net v2 is installed."
        )

    nnunet_raw = require_env_path("nnUNet_raw")
    nnunet_preprocessed = require_env_path("nnUNet_preprocessed")
    nnunet_results = require_env_path("nnUNet_results")

    dataset_dir = validate_dataset(args.dataset, nnunet_preprocessed)
    split_path = validate_exact_split(args.dataset, nnunet_preprocessed)

    command = [
        trainer_exe,
        args.dataset,
        args.configuration,
        str(args.fold),
    ]
    if args.resume:
        command.append("--c")

    print("=" * 72)
    print("AAA-Segmentation nnU-Net training launcher")
    print("=" * 72)
    print(f"Dataset:             {args.dataset}")
    print(f"Configuration:       {args.configuration}")
    print(f"Fold:                {args.fold}")
    print(f"Resume:              {args.resume}")
    print(f"nnUNet_raw:          {nnunet_raw}")
    print(f"nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"nnUNet_results:      {nnunet_results}")
    print(f"Dataset dir:         {dataset_dir}")
    print(f"Verified split:      {split_path}")
    print(f"Trainer:             {trainer_exe}")
    print()
    print("Command:")
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in command))

    if args.dry_run:
        print()
        print("[OK] Dry run complete. Training was not started.")
        return 0

    print()
    print("Starting training...")
    print("-" * 72)
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTraining launcher interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
