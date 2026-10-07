# Data

No data is stored in this repository. Download the public datasets yourself and respect their licences:

- KiTS23 (kidney, tumour, cyst CT segmentation): https://kits-challenge.org/2023/
- KIPA22 (kidney, tumour, artery, vein CT segmentation): https://kipa22.grand-challenge.org/

Expected layout when running the notebooks locally:

```
data/
  kits23/case_00000/{imaging.nii.gz, segmentation.nii.gz}
  kipa22/...
```

Model checkpoints (nnU-Net / BA-Net / ResUNet folds) and mesh exports (.obj/.glb) are also excluded because of size.
