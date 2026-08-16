# EXP01 - nnU-Net baseline

# Planning and preprocessing
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Five-fold cross-validation
nnUNetv2_train Dataset001_AAA 3d_fullres 0
nnUNetv2_train Dataset001_AAA 3d_fullres 1
nnUNetv2_train Dataset001_AAA 3d_fullres 2
nnUNetv2_train Dataset001_AAA 3d_fullres 3
nnUNetv2_train Dataset001_AAA 3d_fullres 4

# Resume interrupted training
# nnUNetv2_train Dataset001_AAA 3d_fullres <FOLD> --c
