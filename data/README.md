# Data

Medical imaging data are not stored in this Git repository.

## Dataset

This project uses the public abdominal aortic aneurysm (AAA) dataset by:

Thanongchai Siriapisith, Worapan Kusakunniran, and Peter Haddawy.

DOI:

https://doi.org/10.6084/m9.figshare.19090052

The baseline experiment (EXP01) uses only the pre-operative
contrast-enhanced CT subset.

Expected source structure:

AAA_dataset/
├── AAA_Train/
└── AAA_Test/

Dataset files must remain outside Git.

Use:

python scripts/data/prepare_aaa_dataset.py --help

to prepare Dataset001_AAA for nnU-Net v2.

The dataset retains its original CC BY 4.0 license.
The MIT license of this repository applies only to repository code.
