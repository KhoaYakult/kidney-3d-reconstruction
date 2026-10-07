# Kidney CT: segmentation and 3D surface reconstruction

Research code from ASIA LAB for AI-assisted analysis of kidney tumours. Given an abdominal CT scan, the goal is to
(1) label the kidney, tumour, cyst and (for KIPA22) renal artery and vein voxel by voxel, and
(2) turn those labels into smooth 3D surfaces that a clinician can rotate and inspect in a browser.

> Status: research prototype. The reconstruction part is the more developed one. The KiTS23 training notebook is a short sanity run, not a converged model (see "Results and limits").

**My part:** surveying current state-of-the-art deep learning models for kidney CT, then applying them and building the segmentation and 3D reconstruction pipelines described below for the kidney 3D reconstruction problem.

## The problem in one picture

```
CT volume (.nii.gz)
   |  Part 1  Segmentation
   |    nnU-Net-style coarse stage  ->  BA-Net (boundary-aware) refinement stage
   v
multi-label mask
   KiTS23 : 1 kidney, 2 tumour/mass, 3 cyst
   KIPA22 : 1 renal vein, 2 kidney, 3 renal artery, 4 tumour
   |  Part 2  3D reconstruction (three methods compared)
   |    A. Marching Cubes (scikit-image), Gaussian + Taubin smoothing
   |    B. MONAI isotropic resampling + VTK Marching Cubes + decimation (PyVista)
   |    C. Neural implicit surfaces: SDF + SIREN network, and NeRP
   v
interactive HTML (Plotly) and meshes (.obj / .glb / .stl)
```

Why three reconstruction methods: plain Marching Cubes leaves voxel "staircase" artefacts and, on CT with thick slices,
flattens the z-axis. B fixes the anisotropy by resampling to 1 mm cubes and is much faster (VTK in C++).
C fits a continuous function to the mask, so the surface can be extracted at any resolution and looks smooth
without post-smoothing.

## Datasets

| Dataset | Content | Used for |
|---|---|---|
| KiTS23 | CT with kidney, tumour, cyst labels (163 cases used here: 114 train / 24 val / 25 test) | Notebooks 01-02, demos |
| KIPA22 | 130 contrast-enhanced abdominal CTs, labels: renal vein, kidney, renal artery, tumour | NeRP scripts, Marching Cubes script |

Data is not included. See `data/README.md`.

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/01_nnunet_banet_two_stage_kits23.ipynb` | Full 2-stage pipeline written in PyTorch/MONAI: residual-encoder 3D U-Net with deep supervision (stage 1), crop ROI, then BA-Net with attention gates and a boundary head (stage 2) |
| `notebooks/02_banet_training_kits23.ipynb` | BA-Net (6.2M parameters) trained and evaluated on KiTS23 with Dice, IoU and HD95 (Kaggle run) |
| `notebooks/03_recon_marching_cubes_skimage.ipynb` | Marching Cubes with auto-crop, Taubin smoothing, decimation, small-component removal |
| `notebooks/04_recon_monai_pyvista.ipynb` | MONAI resampling + VTK pipeline, exports HTML and `.glb` |
| `notebooks/05_recon_neural_sdf_siren.ipynb` | Mask -> signed distance field -> SIREN MLP -> mesh at arbitrary resolution |
| `src/marching_cubes/` | Standalone Marching Cubes script with Plotly HTML output |
| `src/nerp/` | NeRP-based reconstruction (fit a SIREN to the volume, then extract meshes, multi-label colouring) |
| `results/3d_demos/` | Three small interactive 3D examples, open the HTML in a browser |
| `results/nerp_training/` | Snapshots of NeRP reconstructions every 5k iterations |

## Pipelines and model architectures

### Part 1. Segmentation (notebooks 01 and 02)

**Preprocessing (notebook 01, nnU-Net conventions):** reorient to RAS, resample to 1.5 x 1.5 x 2.0 mm (bilinear for CT, nearest for labels), clip CT to [-100, 400] HU and scale to [0, 1], crop to the foreground, store as float16/uint8 tensors. Split 70/15/15 with a fixed seed (163 cases: 114 / 24 / 25).

**Training samples:** patches drawn with a 1:1 foreground/background ratio (`RandCropByPosNegLabeld`), with flips, 90-degree rotations, affine (rotation and scale), Gaussian noise, and intensity shift/scale.

**Two-stage inference:**

```
full volume --(Stage 1, sliding window 128^3, Gaussian blending, overlap 0.5)--> coarse mask
coarse mask --> bounding box + 20-voxel margin --> ROI crop
ROI crop    --(Stage 2, sliding window 96^3)--> fine mask --> paste back into full volume
```

| | Stage 1: `nnUNet3D` | Stage 2: `BANet3D` |
|---|---|---|
| Role | Coarse kidney / tumour / cyst mask on the full volume | Refine boundaries inside the ROI |
| Backbone | 3D residual-encoder U-Net: stem + 4 stride-2 stages, bottleneck, 4 transposed-conv decoder stages with skip concatenation | Same residual encoder, bottleneck at 16x base width |
| Block | Residual block: 2 x (Conv3d 3x3x3, InstanceNorm, LeakyReLU 0.01) plus 1x1 skip | Same |
| Decoder extras | Deep supervision: 4 output heads (coarse to fine) | Additive attention gates (Oktay et al., 2018) on every skip connection, plus a lightweight boundary decoder |
| Outputs | Logits (4 heads in training, finest one at inference) | Segmentation logits and a 1-channel boundary logit |
| Boundary to segmentation link | - | Boundary decoder produces an attention map that scales the finest decoder features by (1 + attention) |
| Base width | 32 | 32 |
| Loss | Dice + cross-entropy (0.5 / 0.5, background excluded), weighted over scales 0.5, 0.25, 0.125, 0.0625 | 0.6 x (Dice + CE) + 0.4 x boundary BCE (positive weight 10); boundary ground truth = mask XOR its 3D erosion |
| Optimiser | AdamW, lr 1e-4, weight decay 1e-5, 10-epoch warm-up then polynomial decay (power 0.9), mixed precision, gradient clipping 1.0 | AdamW, lr 5e-5, same schedule |

Metrics: Dice, IoU (derived from Dice), and 95th-percentile Hausdorff distance (HD95) per class, computed on the pasted-back full-size predictions.

Notebook 02 is a standalone, lighter BA-Net experiment on KiTS23: max-pool U-Net with residual double-conv blocks, 4 levels, attention gates, segmentation head plus boundary head (6.2M parameters, base width 16, 64^3 patches, boundary weight 0.4). The configuration lists 300 epochs, but the saved run stops after 5.

The original KIPA22 workspace also holds ResUNet and BA-Net checkpoints in nnU-Net v1 format (4 folds, 4 classes: vein, kidney, artery, tumour). They were obtained from an external source (not trained in this project) and are not part of this repository. The BA-Net design follows "Boundary-Aware Network for Kidney Parsing" (arXiv:2208.13338).

### Part 2. 3D reconstruction (notebooks 03 to 05, `src/`)

All methods take a multi-label NIfTI mask and build one surface per label (kidney, tumour, cyst, vessels), then render them together as a Plotly HTML scene and, for some, export `.glb`/`.obj`.

| Method | Where | Steps | Why |
|---|---|---|---|
| A. Enhanced Marching Cubes | notebook 03, `src/marching_cubes/` | auto-crop -> morphological closing/opening -> Gaussian blur (sigma 0.5-0.8 per label) -> scikit-image Marching Cubes at level 0.5 with real voxel spacing -> Taubin smoothing (10-15 iterations) -> mesh decimation -> drop small components | Simple baseline |
| B. MONAI + PyVista (VTK) | notebook 04 | MONAI `Spacing` to 1 mm isotropic (nearest) -> auto-crop -> per label PyVista image -> VTK contour (Marching Cubes) -> Taubin smoothing (20-30 iterations) -> quadric decimation (50-70%) -> small-component filter -> mesh statistics (vertices, faces, surface area, volume if watertight) -> HTML + GLB | Removes z-axis flattening on thick-slice CT and runs much faster in C++ |
| C1. Neural SDF with SIREN | notebook 05 | 1 mm resampling -> auto-crop -> per label signed distance field (Euclidean distance transform, positive inside) -> sample points (50% near surface within 5 mm, 30% random, 20% on surface) -> train SIREN (coordinates -> SDF; 4-5 layers, width 256 or 128, omega_0 = 30) with L1 loss + 0.1 x Eikonal loss, Adam lr 1e-4 with cosine decay, 300-600 epochs -> query a 2x finer grid -> Marching Cubes at SDF = 0 -> HTML + GLB | Continuous surface, no staircase, sub-voxel detail, mesh at any resolution |
| C2. NeRP-based INR | `src/nerp/` | normalise the volume to [0, 1] -> fit a SIREN (3D coordinates -> value; optional Gaussian Fourier feature encoding; 5 layers, width 256, L2 loss, Adam lr 1e-4, 25k iterations, snapshots every 5k) -> query the trained network on a 384^3 grid -> per-label Marching Cubes with light smoothing -> colour-coded multi-label HTML and meshes | Super-resolution from the ~161^3 input and a compact representation of the volume |

Network types in NeRP's `networks.py` (upstream, not included): SIREN (sine activations with the SIREN initialisation and omega_0 = 30) and a ReLU MLP with Fourier feature encoding (FFN) for comparison; both end with a sigmoid.

## How to run

Segmentation notebooks and the 3D notebooks were written for Kaggle GPU notebooks. Paths such as `/kaggle/input/...` must be changed for local runs.

Marching Cubes script:

```bash
pip install nibabel SimpleITK trimesh scikit-image plotly scipy
python src/marching_cubes/marching_cubes_plotly.py path/to/mask.nii.gz
```

NeRP route (run from `src/nerp`, after copying the upstream files described in "Attribution and licences"):

```bash
pip install -r requirements.txt trimesh plotly
cp path/to/mask.nii.gz 13.nii.gz
python convert_to_npz.py                                   # -> data/ct_data/kidney_13.npz
python train_image_regression_3d.py --config configs/kidney.yaml   # trains the SIREN, saves checkpoints
python nerp_mutilable_reconstruct.py                       # multi-label mesh + HTML
```

Edit the configuration block at the top of `nerp_mutilable_reconstruct.py` (input file, checkpoint, resolution) before running.

## Results and limits

- Notebook 02 holds a 5-epoch Kaggle run: mean Dice 0.148 (kidney 0.435, mass 0.007, cyst 0.001). It checks that the pipeline runs end to end and is not a converged result.
- Notebook 01 is saved without outputs.
- The 3D reconstructions are compared visually. No surface-distance comparison between methods is reported yet.

## Attribution and licences

- **NeRP.** `src/nerp/` builds on NeRP: L. Shen, J. Pauly, L. Xing, "NeRP: Implicit Neural Representation Learning with Prior Embedding for Sparsely Sampled Image Reconstruction", IEEE TNNLS, 2022 (https://github.com/liyues/NeRP). The upstream code is copyrighted by Stanford University for research use only, has no redistribution licence, and a patent application exists. For that reason the files adapted from upstream (`networks.py`, `data.py`, `train_image_regression_3d.py`, `utils.py` and the upstream sample configs) are **not included** in this repository. To run the NeRP route, clone upstream, copy those files into `src/nerp/` and apply your own settings. Only the scripts written for this project (data conversion, mesh extraction, multi-label reconstruction, `kidney.yaml`) are published here.
- **KiTS23.** N. Heller et al., "2023 Kidney and Kidney Tumor Segmentation Challenge", Zenodo, doi:10.5281/zenodo.7840134, data released under CC BY-NC-SA 4.0 (non-commercial, share-alike; see https://github.com/neheller/kits23). The dataset itself is not included. The 3D demos in `results/3d_demos/` are derived from KiTS23 labels and are shared under the same non-commercial terms.
- **KIPA22.** Challenge dataset of the KiPA 2022 challenge (renal vein, kidney, renal artery, tumour). The dataset is not included.
- **BA-Net.** The boundary-aware design (shared encoder, boundary decoder, segmentation decoder with boundary attention) follows "Boundary-Aware Network for Kidney Parsing" (arXiv:2208.13338). The implementations in the notebooks are written for this project.
- **Attention gate.** Oktay et al., "Attention U-Net", 2018. **SIREN.** Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", 2020.
- No patient-identifying information is included.
