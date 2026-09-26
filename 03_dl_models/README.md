# 03 — 2.5D deep-learning models

Training and inference code for the 2.5D deep-learning networks compared in the manuscript. A 2.5D
input is built from one time point: the axial slice through the largest tumour cross-section
together with the corresponding sagittal and coronal slices, stacked as three channels.

## The 32 candidate networks

The manuscript compares **32 candidate 2.5D networks** at both time points (Figure 1C, pre-nICT, and
Figure 1D, post-nICT). The list below is the exact set, declared in `dl_config.yaml` under
`candidates`:

```
SimpleViT, ViT, mobilenet_v3_small, resnet34, shufflenet_v2_x1_0, mobilenet_v2, resnet50,
mnasnet0_5, resnext50_32x4d, densenet121, resnet18, squeezenet1_0, wide_resnet101_2,
densenet161, googlenet, shufflenet_v2_x0_5, resnet152, alexnet, vgg16, densenet169,
densenet201, inception_v3, wide_resnet50_2, resnet101, vgg13_bn, vgg16_bn, vgg13, mnasnet1_0,
squeezenet1_1, vgg19, mobilenet_v3_large, vgg19_bn
```

`models_2p5d.py` provides a zoo of 50 architectures (the 32 candidates listed above plus the other
torchvision backbones that were available). `run_candidate_comparison.py` trains every candidate in
the list and collects the results into `candidate_comparison.csv`.

## Inputs

- A manifest (csv) with the columns `ID`, `pre_path`, `post_path` and the label column (`pCR`).
  Paths may be absolute or relative to the manifest.
- Optional tumour masks in `--mask-dir`, named `<ID>_mask.nii.gz` (or `.npy`). Without a mask the
  central slice is used.
- The images themselves are NIfTI (`.nii` / `.nii.gz`) or NumPy (`.npy`) volumes.

## Outputs

- `best.pt` — the checkpoint with the best validation AUC (written to the run directory only and
  never committed).
- `history.csv` — loss and validation AUC per epoch.
- `run_config.json` — model, seed, device, best epoch and best validation AUC.
- `candidate_comparison.csv` — validation AUC for every candidate 2.5D network.
- `features_dl.csv` — deep-learning features extracted by `extract_dl_features.py`.

## Scripts

| Script | Purpose |
| --- | --- |
| `datasets.py` | 2.5D input construction (axial / sagittal / coronal), dataset and manifest handling |
| `models_2p5d.py` | Zoo of 50 network architectures and the candidate-selection interface |
| `train_dl.py` | Training loop with fixed seeds, early stopping and a `--smoke-test` mode |
| `run_candidate_comparison.py` | Trains every candidate in the config and summarises the comparison |
| `extract_dl_features.py` | Forward-hook feature extraction from a trained network |
| `dl_config.yaml` | The 32 candidates, training hyper-parameters and seed |

## Usage

```bash
# train one candidate
python train_dl.py --manifest manifest.csv --mask-dir masks/ \
                   --config dl_config.yaml --output runs/resnet18 --model resnet18

# train all 32 candidates and collect the comparison table
python run_candidate_comparison.py --manifest manifest.csv --mask-dir masks/ \
                                   --config dl_config.yaml --output runs/comparison

# extract deep-learning features with the selected network
python extract_dl_features.py --manifest manifest.csv \
                              --checkpoint runs/comparison/SimpleViT/best.pt \
                              --mask-dir masks/ --output features_dl.csv

# installation smoke test: no images, no GPU required
python run_candidate_comparison.py --smoke-test --config dl_config.yaml \
                                   --output runs/smoke --models SimpleViT,resnet18
```

The smoke test of the **complete 32-candidate comparison** (synthetic 224 × 224 volumes, 16 cases)
finishes in about 3 minutes on an NVIDIA GeForce RTX 5090; the resulting table is stored in
`data/expected_output/dl_smoke/candidate_comparison.csv`.

## Notes

- `ViT` corresponds to the torchvision ViT-B/16 implementation; `SimpleViT` is the lightweight
  patch-attention baseline.
- `inception_v3` expects 299 × 299 inputs; all other models use 224 × 224.
- No pretrained weights are downloaded unless `--pretrained` is passed.
- Checkpoints are written to the run output directory on the local filesystem; `.gitignore`
  excludes `*.pt`, `*.pth` and similar files, so no weights can be committed by accident.
- Training and inference were performed on an NVIDIA GeForce RTX 5090 (32 GB) with CUDA 12.8
  (PyTorch 2.11.0+cu128); the code also runs on CPU for the smoke test.
