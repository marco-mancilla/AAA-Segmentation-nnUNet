# AAA Segmentation with nnU-Net v2

Reproducible code, configuration, experiment metadata, and lightweight results for abdominal aortic aneurysm (AAA) segmentation in contrast-enhanced CT using **nnU-Net v2**.

This repository is intentionally separated from the runtime workspace. It contains the files needed to reproduce and audit the computational workflow, while raw medical images, preprocessed nnU-Net data, model checkpoints, and prediction volumes remain outside Git.

> **Research scope — EXP01:** the baseline experiment uses only the **pre-operative contrast-enhanced CT (CECT)** subset of the public AAA dataset by Siriapisith, Kusakunniran, and Haddawy. **Non-contrast CT (NCCT) and post-EVAR data are not included in EXP01.**

---

## Current EXP01 status

`EXP01` is a five-fold cross-validation baseline using the standard `3d_fullres` nnU-Net configuration.

| Fold | Status | Mean validation Dice | Mean validation IoU |
|---|---|---:|---:|
| 0 | Completed | 0.9738874871 | 0.9493091210 |
| 1 | Completed | 0.9696128138 | 0.9413580927 |
| 2 | Completed | 0.9767462827 | 0.9546255451 |
| 3 | Completed | 0.9746482835 | 0.9506707151 |
| 4 | In Progress | — | — |

**Out-of-fold validation completed:** 160 / 200 cases (**80%**).

Four of the five cross-validation folds have completed training and final validation. Fold 4 remains pending.

Per-case metrics currently available:

```text
results/metrics/fold_0_metrics.csv
results/metrics/fold_1_metrics.csv
results/metrics/fold_2_metrics.csv
results/metrics/fold_3_metrics.csv

The experiment remains **in progress** until all five folds are trained and validated.

---

## Reference environment

The reference `EXP01` run uses:

| Component | Reference |
|---|---|
| Python | 3.10 |
| nnU-Net | **v2.8.1** |
| PyTorch | 2.13.0+cu130 |
| CUDA reported by PyTorch | 13.0 |
| Reference GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| nnU-Net configuration | `3d_fullres` |
| Trainer | `nnUNetTrainer` |
| Plans | `nnUNetPlans` |
| Cross-validation | 5 folds |
| Training cases per fold | 160 |
| Validation cases per fold | 40 |
| Epochs per fold | 1000 |

The GPU and CUDA entries describe the **reference machine**, not a requirement for every compatible installation.

Portable dependencies are listed in `environment/requirements.txt`, while the reference environment snapshot is preserved in `environment/requirements-lock.txt`.

---

## Repository layout

```text
AAA-Segmentation-nnUNet/
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
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
│   └── README.md
├── scripts/
│   ├── setup_workspace.py
│   ├── data/
│   │   ├── inspect_aaa.py
│   │   └── prepare_aaa_dataset.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── export_fold_metrics.py
│   └── inference/
├── experiments/
│   └── exp01_baseline/
│       ├── experiment.yaml
│       ├── commands.ps1
│       └── commands.sh
├── results/
│   ├── metrics/
│   │   ├── fold_0_metrics.csv
│   │   ├── fold_1_metrics.csv
│   │   ├── fold_2_metrics.csv
│   │   └── fold_3_metrics.csv
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

This separation minimizes the risk of accidentally committing medical images, large preprocessed data, predictions, checkpoints, or machine-specific files.

---

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd AAA-Segmentation-nnUNet
```

Python 3.10 is used by the reference experiment.

---

## 2. Create the workspace and virtual environment

The same setup helper works on Windows and Linux:

```bash
python scripts/setup_workspace.py
```

By default, it creates `../AAA-Segmentation-workspace`.

To choose another location:

```bash
python scripts/setup_workspace.py --workspace /path/to/AAA-Segmentation-workspace
```

The setup creates the runtime workspace, virtual environment, nnU-Net directory structure, activation helpers, and workspace metadata.

### Windows PowerShell

```powershell
& "..\AAA-Segmentation-workspace\.venv\Scripts\Activate.ps1"
. "..\AAA-Segmentation-workspace\activate_nnunet.ps1"
```

### Linux

```bash
source ../AAA-Segmentation-workspace/.venv/bin/activate
source ../AAA-Segmentation-workspace/activate_nnunet.sh
```

The helper defines the three paths required by nnU-Net:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

---

## 3. Install PyTorch and project dependencies

Install a PyTorch build appropriate for the operating system, GPU, driver, and CUDA environment first:

https://pytorch.org/get-started/locally/

Then install the project dependencies:

```bash
python -m pip install -r environment/requirements.txt
```

Verify the installation:

```bash
python -c "import torch, nnunetv2, SimpleITK; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

To confirm the installed nnU-Net package version:

```bash
python -m pip show nnunetv2
```

The reference experiment was performed with **nnU-Net v2.8.1**.

---

## 4. Download the public AAA dataset

Dataset:

**Thanongchai Siriapisith, Worapan Kusakunniran, Peter Haddawy.  
“A 3D deep learning approach incorporating coordinate information to improve the segmentation of pre- and post-operative abdominal aortic aneurysm.”**

DOI:

https://doi.org/10.6084/m9.figshare.19090052

`EXP01` intentionally selects only the **pre-operative contrast-enhanced CT** subset:

- 200 CE training/cross-validation cases;
- 20 CE test cases.

Although the source dataset also provides other data, including non-contrast and post-operative material, those images are **outside the scope of EXP01**.

The dataset is distributed by its authors under **CC BY 4.0**. The MIT license in this repository applies to repository code, not to third-party dataset files.

### Case-level interpretation

Within this repository, each CT volume is handled as an individual **case**. The current computational pipeline does not use patient identifiers, study dates, or longitudinal links. Therefore, repository documentation should not infer patient-level uniqueness or longitudinal follow-up unless that relation is explicitly established from the source dataset metadata.

---

## 5. Inspect a source image/label pair

`scripts/data/inspect_aaa.py` reports image/label dimensions, voxel spacing, origin/direction, CT intensity statistics, segmentation labels, voxel counts, and geometry consistency.

Example:

```bash
python scripts/data/inspect_aaa.py \
  --image "../AAA-Segmentation-workspace/data/source/AAA_dataset/AAA_Train/CE001_CE.nrrd" \
  --label "../AAA-Segmentation-workspace/data/source/AAA_dataset/AAA_Train/CE001_gt-label.nrrd"
```

A valid binary segmentation is expected to contain labels `0` and `1`.

---

## 6. Prepare `Dataset001_AAA`

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

The preparation script selects the CE subset, validates image/label geometry, verifies binary labels, converts NRRD to compressed NIfTI, creates nnU-Net IDs, generates `dataset.json`, keeps test references outside `labelsTr`, and writes mapping/report files.

### Source-ID overlap safeguard

Some CE identifiers occur in both `AAA_Train` and `AAA_Test`. Identifier equality alone does not establish whether two files represent the same study or patient.

The preparation pipeline compares overlapping source IDs using file SHA-256 hashes and voxel+geometry hashes for both images and labels. For the currently prepared CE subset, no byte-identical or voxel+geometry-identical volumes were detected among the overlapping source identifiers.

This check does **not** establish patient-level independence.

---

## 7. Plan and preprocess with nnU-Net

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Reference configuration:

```text
Dataset:       Dataset001_AAA
Configuration: 3d_fullres
Trainer:       nnUNetTrainer
Plans:         nnUNetPlans
Batch size:    2
Patch size:    [224, 56, 192]
```

The complete generated planning/fingerprint files are preserved under `configs/Dataset001_AAA/` and should be treated as the authoritative configuration record.

---

## 8. Reproduce the exact 5-fold split

The exact split used by `EXP01` is version-controlled at:

```text
configs/Dataset001_AAA/splits_final.json
```

Each fold contains 160 training and 40 validation cases. Across all five folds, each of the 200 cross-validation cases is used once as an out-of-fold validation case.

Synchronize the split into the preprocessed workspace:

```bash
python scripts/setup_workspace.py --sync-splits
```

For a custom workspace:

```bash
python scripts/setup_workspace.py \
  --workspace /path/to/AAA-Segmentation-workspace \
  --sync-splits
```

---

## 9. Train the folds reproducibly

Preferred interface:

```bash
python scripts/training/train.py --fold <FOLD>
```

Example:

```bash
python scripts/training/train.py --fold 2
```

Before delegating to `nnUNetv2_train`, the launcher verifies the nnU-Net executable, required environment variables, preprocessed dataset, `splits_final.json`, equality between workspace and version-controlled splits, and the requested fold.

### Dry run

```bash
python scripts/training/train.py --fold 2 --dry-run
```

### Resume an interrupted fold

```bash
python scripts/training/train.py --fold <FOLD> --resume
```

Equivalent direct nnU-Net command:

```bash
nnUNetv2_train Dataset001_AAA 3d_fullres <FOLD>
```

`EXP01` keeps the same dataset, split definition, trainer, plans, and configuration across all five folds.

---

## 10. Export per-case validation metrics

After a fold completes:

```bash
python scripts/evaluation/export_fold_metrics.py --fold <FOLD>
```

Example:

```bash
python scripts/evaluation/export_fold_metrics.py --fold 1
```

The exporter reads the fold validation `summary.json` and writes:

```text
results/metrics/fold_<N>_metrics.csv
```

The exported table contains case ID, fold, Dice, IoU, precision, recall, TP, FP, FN, TN, predicted/reference foreground voxel counts, volume difference, and percentage volume difference.

It also prints fold-level descriptive statistics and the lowest-Dice cases for quick inspection.

The final cross-validation analysis will consolidate the five fold CSV files into a 200-case out-of-fold result set.

---

## 11. Evaluation strategy

Dice is treated as a segmentation overlap metric, not as clinical accuracy.

The planned analysis extends beyond mean Dice to include case-level Dice/IoU, false-positive and false-negative burden, predicted-vs-reference volume differences, surface/distance metrics such as HD95 and ASD/ASSD, spatial inspection, and characterization of oversegmentation, undersegmentation, fragmentation, false positives, false negatives, and boundary errors.

A later stage will examine the possible effect of segmentation errors on geometric or clinically relevant measurements using a separately defined reproducible measurement methodology.

---

## Reproducibility policy

The Git repository should contain:

- source code;
- small configuration files;
- exact cross-validation splits;
- experiment metadata;
- lightweight metrics and tables;
- documentation required to reconstruct the workflow.

The Git repository should **not** contain:

- original medical images;
- generated `nnUNet_raw`, `nnUNet_preprocessed`, or `nnUNet_results` trees;
- NIfTI/NRRD prediction volumes;
- model checkpoints (`.pth`, `.pt`, `.ckpt`);
- probability archives such as `.npz`;
- local virtual environments;
- secrets or machine-specific environment files.

Large model artifacts can be distributed separately if needed.

---

## Licenses and citation

Repository code is released under the [MIT License](LICENSE).

The AAA dataset is a third-party dataset distributed under **CC BY 4.0**.
Dataset files are not redistributed in this repository and remain subject to
their original license and attribution requirements.

If you use the dataset, please cite:

> Siriapisith, T., Kusakunniran, W., & Haddawy, P.  
> *A 3D deep learning approach incorporating coordinate information to improve
> the segmentation of pre- and post-operative abdominal aortic aneurysm.*  
> Dataset: https://doi.org/10.6084/m9.figshare.19090052

If you use nnU-Net, please cite:

> Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021).  
> *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.*  
> Nature Methods, 18, 203–211.

nnU-Net is developed by the Division of Medical Image Computing at the
German Cancer Research Center (DKFZ).