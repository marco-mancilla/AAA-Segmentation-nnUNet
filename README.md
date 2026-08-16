# AAA Segmentation with nnU-Net v2

Reproducible code and experiment metadata for abdominal aortic aneurysm (AAA)
segmentation in contrast-enhanced CT using **nnU-Net v2**.

This repository contains the **code, configuration, and lightweight experiment
results** required to reproduce the computational workflow. Raw medical images,
preprocessed nnU-Net data, model checkpoints, and prediction volumes are kept
outside Git.

> **Research scope:** the baseline experiment (`EXP01`) uses the pre-operative
> contrast-enhanced (CE) CT subset of the public AAA dataset by Siriapisith,
> Kusakunniran, and Haddawy.

## Repository layout

```text
AAA-Segmentation-nnUNet/
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   └── Dataset001_AAA/
│       ├── dataset.json
│       ├── dataset_fingerprint.json
│       ├── nnUNetPlans.json
│       └── splits_final.json
├── environment/
│   ├── README.md
│   ├── requirements.txt
│   └── requirements-lock.txt
├── data/
│   └── .gitkeep
├── scripts/
│   ├── setup_workspace.py
│   ├── data/
│   │   ├── inspect_aaa.py
│   │   └── prepare_aaa_dataset.py
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── experiments/
│   └── exp01_baseline/
├── results/
│   ├── metrics/
│   └── tables/
├── figures/
│   └── diagnostics/
└── tests/
```

The runtime workspace is intentionally **separate from the Git repository**:

```text
parent-directory/
├── AAA-Segmentation-nnUNet/       # Git repository
└── AAA-Segmentation-workspace/    # generated locally; not committed
    ├── .venv/
    ├── nnUNet_raw/
    ├── nnUNet_preprocessed/
    ├── nnUNet_results/
    ├── data/
    │   ├── source/
    │   ├── reports/
    │   └── test_ground_truth/
    └── logs/
```

This design minimizes the risk of accidentally committing medical images,
large preprocessed files, predictions, or model weights.

---

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd AAA-Segmentation-nnUNet
```

Python **3.10 or newer** is recommended by current nnU-Net documentation.

---

## 2. Create the workspace and virtual environment

The same Python setup script works on Windows and Linux:

```bash
python scripts/setup_workspace.py
```

By default it creates:

```text
../AAA-Segmentation-workspace
```

To choose another location:

```bash
python scripts/setup_workspace.py --workspace /path/to/AAA-Segmentation-workspace
```

### Windows PowerShell

Activate the virtual environment:

```powershell
& "..\AAA-Segmentation-workspace\.venv\Scripts\Activate.ps1"
```

Load the nnU-Net environment variables:

```powershell
. "..\AAA-Segmentation-workspace\activate_nnunet.ps1"
```

### Linux

Activate the virtual environment:

```bash
source ../AAA-Segmentation-workspace/.venv/bin/activate
```

Load the nnU-Net environment variables:

```bash
source ../AAA-Segmentation-workspace/activate_nnunet.sh
```

The helper defines the three paths required by nnU-Net:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

---

## 3. Install PyTorch and project dependencies

Install a PyTorch build appropriate for your operating system and GPU first:

https://pytorch.org/get-started/locally/

Then install the project dependencies:

```bash
python -m pip install -r environment/requirements.txt
```

The environment used for the reference experiment is recorded in:

```text
environment/requirements-lock.txt
```

The lock file is provided for provenance. A byte-for-byte identical Python
environment may not be portable across operating systems or GPU/CUDA versions,
so use it as a reference rather than blindly installing it on every platform.

Verify the installation:

```bash
python -c "import torch, nnunetv2, SimpleITK; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 4. Download the public AAA dataset

Dataset:

**Thanongchai Siriapisith, Worapan Kusakunniran, Peter Haddawy.  
“A 3D deep learning approach incorporating coordinate information to improve
the segmentation of pre- and post-operative abdominal aortic aneurysm.”**

DOI:

https://doi.org/10.6084/m9.figshare.19090052

Figshare reports the following relevant pre-operative subsets:

- Contrast-enhanced CT: 200 training cases
- Contrast-enhanced CT: 20 test cases
- Non-contrast CT: 200 training cases
- Non-contrast CT: 20 test cases

`EXP01` uses **only the pre-operative contrast-enhanced CT subset**.

The dataset is distributed by its authors under **CC BY 4.0**. The MIT license
in this repository applies to repository code, not to third-party dataset files.

Download and extract the dataset somewhere outside Git, for example:

```text
AAA-Segmentation-workspace/
└── data/
    └── source/
        └── AAA_dataset/
            ├── AAA_Train/
            │   ├── CE001_CE.nrrd
            │   ├── CE001_gt-label.nrrd
            │   └── ...
            └── AAA_Test/
                ├── CE021_CE.nrrd
                ├── CE021_gt-label.nrrd
                └── ...
```

The preparation script can also receive a parent directory and will locate a
single nested directory containing both `AAA_Train` and `AAA_Test`.

---

## 5. Inspect a source image/label pair

`inspect_aaa.py` is a lightweight sanity checker. It reports:

- image and label dimensions;
- voxel spacing;
- origin and direction;
- NumPy array shape;
- CT intensity statistics;
- segmentation labels and voxel counts;
- whether image/label geometry matches.

Example:

```bash
python scripts/data/inspect_aaa.py \
  --image "../AAA-Segmentation-workspace/data/source/AAA_dataset/AAA_Train/CE001_CE.nrrd" \
  --label "../AAA-Segmentation-workspace/data/source/AAA_dataset/AAA_Train/CE001_gt-label.nrrd"
```

On Windows PowerShell the same command can be written on one line:

```powershell
python .\scripts\data\inspect_aaa.py --image "..\AAA-Segmentation-workspace\data\source\AAA_dataset\AAA_Train\CE001_CE.nrrd" --label "..\AAA-Segmentation-workspace\data\source\AAA_dataset\AAA_Train\CE001_gt-label.nrrd"
```

A valid binary segmentation is expected to contain labels `0` and `1`.

---

## 6. Prepare `Dataset001_AAA`

Run:

### Windows PowerShell

```powershell
python .\scripts\data\prepare_aaa_dataset.py `
  --dataset-root "..\AAA-Segmentation-workspace\data\source\AAA_dataset" `
  --workspace "..\AAA-Segmentation-workspace"
```

### Linux

```bash
python scripts/data/prepare_aaa_dataset.py \
  --dataset-root ../AAA-Segmentation-workspace/data/source/AAA_dataset \
  --workspace ../AAA-Segmentation-workspace
```

The script:

1. selects only files matching `CE*_CE.nrrd`;
2. requires the corresponding `CE###_gt-label.nrrd`;
3. validates image/label geometry;
4. verifies binary labels `{0, 1}` and non-empty foreground;
5. converts NRRD to compressed NIfTI using SimpleITK;
6. maps source training cases to `AAA_001` … `AAA_200`;
7. maps source test cases to `AAA_TEST_001` … `AAA_TEST_020`;
8. creates the nnU-Net v2 `dataset.json`;
9. stores test reference masks outside `labelsTr`;
10. writes a case mapping and preparation report under `workspace/data/reports/`.

Generated nnU-Net structure:

```text
nnUNet_raw/
└── Dataset001_AAA/
    ├── imagesTr/
    │   ├── AAA_001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── AAA_001.nii.gz
    │   └── ...
    ├── imagesTs/
    │   ├── AAA_TEST_001_0000.nii.gz
    │   └── ...
    └── dataset.json
```

Test references are kept separately:

```text
data/test_ground_truth/
├── AAA_TEST_001.nii.gz
└── ...
```

### Source-ID overlap safeguard

The downloaded dataset uses some CE identifiers in both `AAA_Train` and
`AAA_Test`. Identifier equality alone does not establish whether two files are
the same study. The preparation script therefore checks overlapping IDs using
both file SHA-256 and voxel+geometry hashes.

If identical image or label content is detected across train/test, preparation
stops by default so the potential leakage can be investigated before reporting
a held-out test result.

---

## 7. Plan and preprocess with nnU-Net

After loading the nnU-Net environment variables:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

The reference experiment uses:

```text
Dataset001_AAA
configuration: 3d_fullres
trainer: nnUNetTrainer
plans: nnUNetPlans
```

The generated planning/fingerprint files from the reference run are preserved
under `configs/Dataset001_AAA/` for auditability.

---

## 8. Reproduce the exact 5-fold split

The exact cross-validation split used by `EXP01` is version-controlled at:

```text
configs/Dataset001_AAA/splits_final.json
```

After planning/preprocessing, copy it into the preprocessed dataset with the
cross-platform helper:

```bash
python scripts/setup_workspace.py --sync-splits
```

If you used a custom workspace:

```bash
python scripts/setup_workspace.py \
  --workspace /path/to/AAA-Segmentation-workspace \
  --sync-splits
```

This step must be completed **before starting training** if exact fold
membership is required.

---

## 9. Train the five folds

```bash
nnUNetv2_train Dataset001_AAA 3d_fullres 0
nnUNetv2_train Dataset001_AAA 3d_fullres 1
nnUNetv2_train Dataset001_AAA 3d_fullres 2
nnUNetv2_train Dataset001_AAA 3d_fullres 3
nnUNetv2_train Dataset001_AAA 3d_fullres 4
```

If a run is interrupted, nnU-Net can continue from its latest saved checkpoint:

```bash
nnUNetv2_train Dataset001_AAA 3d_fullres <FOLD> --c
```

### Reference EXP01 status

At the time this README section was written:

| Fold | Status | Mean validation Dice | Mean validation IoU |
|---|---|---:|---:|
| 0 | Completed | 0.9738874871 | 0.9493091210 |
| 1 | Pending | — | — |
| 2 | Pending | — | — |
| 3 | Pending | — | — |
| 4 | Pending | — | — |

Per-case Fold 0 metrics are available at:

```text
results/metrics/fold_0_metrics.csv
```

---

## Reproducibility policy

The Git repository should contain:

- source code;
- small configuration files;
- exact CV splits;
- experiment metadata;
- lightweight metrics/tables;
- instructions required to reconstruct the workflow.

The Git repository should **not** contain:

- original medical images;
- generated `nnUNet_raw`, `nnUNet_preprocessed`, or `nnUNet_results` trees;
- NIfTI/NRRD prediction volumes;
- model checkpoints (`.pth`, `.pt`, `.ckpt`);
- local virtual environments;
- secrets or machine-specific environment files.

Large model artifacts can be distributed separately if needed.

---

## Licenses and citation

Repository code is released under the license in `LICENSE` (MIT).

The AAA dataset is a third-party dataset and retains its own **CC BY 4.0**
license and attribution requirements. Cite the dataset authors and DOI when
using it.

nnU-Net is developed by the Division of Medical Image Computing at the German
Cancer Research Center (DKFZ). Please follow the nnU-Net project citation
instructions when publishing results produced with nnU-Net.
