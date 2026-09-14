# AAA Segmentation with nnU-Net v2

Reproducible code, configuration, experiment metadata, and lightweight results for abdominal aortic aneurysm (AAA) segmentation in contrast-enhanced CT using **nnU-Net v2**.

This repository is intentionally separated from the runtime workspace. It contains the files needed to reproduce and audit the computational workflow, while raw medical images, preprocessed nnU-Net data, model checkpoints, and prediction volumes remain outside Git.

> **Research scope — EXP01:** the baseline experiment uses only the **pre-operative contrast-enhanced CT (CECT)** subset of the public AAA dataset by Siriapisith, Kusakunniran, and Haddawy. **Non-contrast CT (NCCT) and post-EVAR data are not included in EXP01.**

---

## Current EXP01 status

`EXP01` is a five-fold cross-validation baseline using the standard `3d_fullres` nnU-Net configuration.

| Fold | Status | Mean validation Dice | Mean validation IoU |
| ---: | :--- | ---: | ---: |
| 0 | Completed | 0.9738874871 | 0.9493091210 |
| 1 | Completed | 0.9696128138 | 0.9413580927 |
| 2 | Completed | 0.9767462827 | 0.9546255451 |
| 3 | Completed | 0.9746482835 | 0.9506707151 |
| 4 | Completed | 0.9739619610 | 0.9495494120 |

**Five-fold cross-validation completed:** 200 / 200 out-of-fold cases (**100%**).

Each of the 200 cross-validation cases was evaluated exactly once by the fold model that did not use that case for training. The normalized per-case OOF table is stored at:

```text
results/metrics/exp01_oof_metrics.csv
```

### Complete cross-validation result

| Metric | Mean | Median | SD | Minimum | Maximum |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Dice | 0.9737713656 | 0.9765188599 | 0.0110437081 | 0.9133852911 | 0.9893980449 |
| IoU | 0.9491025772 | 0.9541151484 | 0.0204693108 | 0.8405788029 | 0.9790185343 |

The aggregate statistics above are computed directly from the **200 unique out-of-fold predictions**.

### Frozen five-fold ensemble on the held-out test set

After cross-validation was completed, the five final fold checkpoints were used together for inference on the **20 reserved CECT test cases**. The test predictions were frozen before evaluation by storing SHA-256 hashes for every prediction.

Structural validation of the exported predictions:

| Check | Result |
| :--- | ---: |
| Test predictions present | 20 / 20 |
| Geometry matched to source CT | 20 / 20 |
| Binary masks | 20 / 20 |
| Non-empty foreground masks | 20 / 20 |

The frozen prediction identities are recorded in:

```text
results/manifests/exp01_test_ensemble_sha256.csv
```

The structural validation report is stored in:

```text
results/tables/exp01_test_ensemble_validation.csv
```

The prediction volumes themselves are **not** committed to Git.

### Test ensemble result vs dataset reference masks

The frozen five-fold ensemble was evaluated against the original reference masks supplied with the 20 test cases. No resampling or post-hoc correction was applied during this evaluation.

| Metric | Value |
| :--- | ---: |
| Cases | 20 |
| Mean Dice | 0.9808049315 |
| Median Dice | 0.9810233683 |
| Dice SD | 0.0055996805 |
| Minimum Dice | 0.9672762766 |
| Maximum Dice | 0.9922366389 |
| Mean IoU | 0.9623890360 |
| Median IoU | 0.9627535598 |
| IoU SD | 0.0107581146 |
| Mean precision | 0.9763024622 |
| Mean recall | 0.9855119700 |
| Mean signed volume difference | +0.9606% |

Per-case test metrics are stored in:

```text
results/metrics/exp01_test_ensemble_metrics.csv
```

The lowest-Dice test cases were `AAA_TEST_015` (0.967276), `AAA_TEST_013` (0.971166), `AAA_TEST_010` (0.975991), `AAA_TEST_014` (0.976654), and `AAA_TEST_019` (0.978007).

These results quantify agreement with the **dataset reference masks**. They should not be interpreted as clinical accuracy or as a substitute for independent expert annotation.

### Independent expert annotation

Independent radiologist segmentation of the same 20 CECT test cases is being prepared/performed separately. The radiologist receives the CT images without the original dataset masks or model predictions. Once those annotations are available, the planned comparisons are:

- ensemble vs radiologist;
- dataset reference vs radiologist;
- ensemble vs dataset reference (already completed).

## Reference environment

The reference `EXP01` run uses:

| **ComponentReference**    |                                  |
| ------------------------- | -------------------------------- |
| Python                    | 3.10                             |
| nnU-Net                   | **v2.8.1**                       |
| PyTorch                   | 2.13.0+cu130                     |
| CUDA reported by PyTorch  | 13.0                             |
| Reference GPU             | NVIDIA GeForce RTX 5060 TI 16 GB |
| nnU-Net configuration     | `3d_fullres`                     |
| Trainer                   | `nnUNetTrainer`                  |
| Plans                     | `nnUNetPlans`                    |
| Cross-validation          | 5 folds                          |
| Training cases per fold   | 160                              |
| Validation cases per fold | 40                               |
| Epochs per fold           | 1000                             |

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
│   │   ├── export_fold_metrics.py
│   │   ├── consolidate_oof_metrics.py
│   │   ├── validate_test_predictions.py
│   │   └── evaluate_test_ensemble.py
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
│   │   ├── fold_3_metrics.csv
│   │   ├── fold_4_metrics.csv
│   │   ├── exp01_oof_metrics.csv
│   │   └── exp01_test_ensemble_metrics.csv
│   ├── manifests/
│   │   └── exp01_test_ensemble_sha256.csv
│   └── tables/
│       └── exp01_test_ensemble_validation.csv
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
    │   ├── test_ground_truth/
    │   └── test_predictions/
    └── logs/
```

This separation minimizes the risk of accidentally committing medical images, large preprocessed data, predictions, checkpoints, or machine-specific files.

## 1. Clone the repository

```
git clone <YOUR-REPOSITORY-URL>
cd AAA-Segmentation-nnUNet
```

Python 3.10 is used by the reference experiment.

---

## 2. Create the workspace and virtual environment

The same setup helper works on Windows and Linux:

```
python scripts/setup_workspace.py
```

By default, it creates `../AAA-Segmentation-workspace`.

To choose another location:

```
python scripts/setup_workspace.py --workspace /path/to/AAA-Segmentation-workspace
```

The setup creates the runtime workspace, virtual environment, nnU-Net directory structure, activation helpers, and workspace metadata.

### Windows PowerShell

```
& "..\AAA-Segmentation-workspace\.venv\Scripts\Activate.ps1"
. "..\AAA-Segmentation-workspace\activate_nnunet.ps1"
```

### Linux

```
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

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then install the project dependencies:

```
python -m pip install -r environment/requirements.txt
```

Verify the installation:

```
python -c "import torch, nnunetv2, SimpleITK; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

To confirm the installed nnU-Net package version:

```
python -m pip show nnunetv2
```

The reference experiment was performed with **nnU-Net v2.8.1**.

---

## 4. Download the public AAA dataset

Dataset:

**Thanongchai Siriapisith, Worapan Kusakunniran, Peter Haddawy.**
**“A 3D deep learning approach incorporating coordinate information to improve the segmentation of pre- and post-operative abdominal aortic aneurysm.”**

DOI:

[https://doi.org/10.6084/m9.figshare.19090052](https://doi.org/10.6084/m9.figshare.19090052)

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

```
python scripts/data/inspect_aaa.py \
  --image "../AAA-Segmentation-workspace/data/source/AAA_dataset/AAA_Train/CE001_CE.nrrd" \
  --label "../AAA-Segmentation-workspace/data/source/AAA_dataset/AAA_Train/CE001_gt-label.nrrd"
```

A valid binary segmentation is expected to contain labels `0` and `1`.

---

## 6. Prepare `Dataset001_AAA`

### Windows PowerShell

```
python .\scripts\data\prepare_aaa_dataset.py `
  --dataset-root "..\AAA-Segmentation-workspace\data\source\AAA_dataset" `
  --workspace "..\AAA-Segmentation-workspace"
```

### Linux

```
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

```
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Reference configuration:

```
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

```
configs/Dataset001_AAA/splits_final.json

```

Each fold contains 160 training and 40 validation cases. Across all five folds, each of the 200 cross-validation cases is used once as an out-of-fold validation case.

Synchronize the split into the preprocessed workspace:

```
python scripts/setup_workspace.py --sync-splits
```

For a custom workspace:

```
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

Before training, the launcher verifies the active Python environment, required nnU-Net environment variables, the preprocessed dataset, `splits_final.json`, equality between the workspace and version-controlled split definitions, and the requested fold.

The launcher delegates training through the **active Python interpreter**:

```bash
python -m nnunetv2.run.run_training Dataset001_AAA 3d_fullres <FOLD>
```

This avoids depending on a separate console launcher and makes the Python environment used for training explicit.

### Dry run

```bash
python scripts/training/train.py --fold 2 --dry-run
```

### Resume an interrupted fold

```bash
python scripts/training/train.py --fold <FOLD> --resume
```

`EXP01` keeps the same dataset, split definition, trainer, plans, configuration, and 1000-epoch training schedule across all five folds.

## 10. Export and consolidate cross-validation metrics

After a fold completes:

```bash
python scripts/evaluation/export_fold_metrics.py --fold <FOLD>
```

The exporter reads the corresponding validation `summary.json` and writes:

```text
results/metrics/fold_<N>_metrics.csv
```

The exported tables contain case ID, fold, Dice, IoU, precision, recall, TP, FP, FN, TN, predicted/reference foreground voxel counts, volume difference, and percentage volume difference.

After all five folds are available, consolidate the 200 OOF cases with:

```bash
python scripts/evaluation/consolidate_oof_metrics.py
```

This generates:

```text
results/metrics/exp01_oof_metrics.csv
```

The consolidation step validates five 40-case fold files, inserts/reconstructs compatible fields when needed, checks unique case coverage, and produces a normalized 200-case OOF table.

---

## 11. Generate the frozen five-fold test ensemble

The final test prediction uses the five completed `3d_fullres` fold checkpoints together. nnU-Net combines the fold predictions internally; the checkpoint files are not manually merged.

Conceptually, the inference call uses:

```text
dataset:       Dataset001_AAA
configuration: 3d_fullres
folds:         0 1 2 3 4
checkpoint:    checkpoint_final.pth
```

For the reference Windows run, the nnU-Net prediction entry point was invoked through the active Python environment:

```powershell
python -c "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()" `
  -i "<TEST_IMAGES_DIR>" `
  -o "<TEST_OUTPUT_DIR>" `
  -d Dataset001_AAA `
  -c 3d_fullres `
  -f 0 1 2 3 4 `
  -chk checkpoint_final.pth
```

The run produced 20/20 test segmentations. Prediction NIfTI files remain outside Git.

---

## 12. Validate and freeze test predictions

Before inspecting performance against the test references, validate the exported masks:

```bash
python scripts/evaluation/validate_test_predictions.py \
  --images-dir "<TEST_IMAGES_DIR>" \
  --predictions-dir "<TEST_OUTPUT_DIR>"
```

The script checks expected case coverage, source-CT geometry, binary labels, non-empty foreground, foreground volume, and SHA-256 for every prediction.

It writes:

```text
results/manifests/exp01_test_ensemble_sha256.csv
results/tables/exp01_test_ensemble_validation.csv
```

The manifest provides a lightweight record of the exact frozen prediction files without redistributing the prediction volumes.

---

## 13. Evaluate the test ensemble against dataset references

After the predictions are frozen, evaluate them against the original 20 test reference masks:

```bash
python scripts/evaluation/evaluate_test_ensemble.py \
  --predictions-dir "<TEST_OUTPUT_DIR>" \
  --ground-truth-dir "<TEST_GROUND_TRUTH_DIR>"
```

The evaluator requires matching geometry and binary masks and intentionally performs **no implicit resampling or post-hoc correction**.

It writes:

```text
results/metrics/exp01_test_ensemble_metrics.csv
```

Per-case outputs include Dice, IoU, precision, recall, TP, FP, FN, TN, predicted/reference foreground counts, and signed volume differences.

---

## 14. Evaluation strategy

Dice is treated as a segmentation overlap metric, not as clinical accuracy.

Completed evaluation stages currently include:

- 200-case five-fold OOF overlap evaluation;
- frozen five-fold ensemble inference on 20 reserved test CECT cases;
- structural validation and SHA-256 freezing of all 20 test predictions;
- ensemble-vs-dataset-reference test evaluation.

The next analysis stage extends beyond overlap metrics to surface/distance measures such as **HD95** and **ASD/ASSD**, followed by spatial inspection and characterization of oversegmentation, undersegmentation, false positives, false negatives, fragmentation, and boundary errors.

A separate expert-annotation stage will compare the ensemble and original dataset references with independent radiologist segmentations on the same 20 test CTs.

A later stage will examine the possible effect of segmentation errors on geometric or clinically relevant measurements using a separately defined reproducible measurement methodology.

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

Repository code is released under the [MIT License](https://github.com/marco-mancilla/AAA-Segmentation-nnUNet/blob/main/LICENSE).

The AAA dataset is a third-party dataset distributed under **CC BY 4.0**. Dataset files are not redistributed in this repository and remain subject to their original license and attribution requirements.

If you use the dataset, please cite:

> Siriapisith, T., Kusakunniran, W., & Haddawy, P.
> *A 3D deep learning approach incorporating coordinate information to improve the segmentation of pre- and post-operative abdominal aortic aneurysm.*
> Dataset: [https://doi.org/10.6084/m9.figshare.19090052](https://doi.org/10.6084/m9.figshare.19090052)

If you use nnU-Net, please cite:

> Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021).
> *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.*
> Nature Methods, 18, 203–211.

nnU-Net is developed by the Division of Medical Image Computing at the German Cancer Research Center (DKFZ).
