# SAMID

# Unsupervised Hybrid Transformer–Mamba Diffusion for Depth-Dependent Image Denoising in Scanning Acoustic Microscopy

Official code for the paper accepted at **IEEE Transactions on Industrial Informatics (IF = 9.8), 2026**:

> Trong Nhan Nguyen, Vu Hoang Minh Doan, Tan Hung Vo, Dang Khoa Pham, Quoc Dung Nguyen,
> Truong Tien Vo, Jaeyeop Choi, Junghwan Oh, "Unsupervised Hybrid Transformer–Mamba Diffusion
> for Depth-Dependent Image Denoising in Scanning Acoustic Microscopy," *IEEE Transactions on
> Industrial Informatics*, 2026. [DOI: 10.1109/TII.2025.3648838](https://doi.org/10.1109/TII.2025.3648838)

---

## Overview

SAMID is a denoising pipeline for Scanning Acoustic Microscopy (SAM) imagery. It combines a
Transformer + latent-diffusion prior (a fork of [HI-Diff](https://github.com/zhengchen1999/HI-Diff))
with a Mamba-based state-space block for depth-dependent noise, trained on synthetically
degraded SAM images. See [NOTICE](NOTICE) for full attribution of the third-party code this
project builds on.

## Repository structure

```
.
├── denoising/          # Core model package: archs, models (Denoising_S1/S2), data, utils
├── diffusion/           # Latent diffusion components (DDPM schedule, samplers) used by denoising/
├── evaluate/
│   ├── IQA/              # Full-reference metrics (PSNR, SSIM, LPIPS, ...)
│   └── NIQA/              # No-reference IQA (BRISQUE, NIQE, PIQE, RankIQA, MetaIQA)
├── DomainTransfer/       # SAM<->target domain transfer experiments (code only; weights/data gitignored)
├── configs/
│   ├── train/            # Training configs (stage 1: regression, stage 2: + diffusion prior)
│   └── test/              # Testing configs
├── scripts/              # Standalone data-prep / evaluation utility scripts
├── notebooks/             # Analysis and benchmarking notebooks
├── image_degradation.py   # Synthetic SAM noise degradation pipeline
├── train.py / test.py     # Training / testing entry points (BasicSR-based)
└── requirements.txt
```

`data/`, `degradation/`, `results/`, `asset/`, and all model weights (`*.pth`) are not tracked
in git (see `.gitignore`) — they are large binary datasets/outputs regenerated or downloaded
locally, not part of the source tree.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Tested with Python 3.9+ and PyTorch with CUDA support (a GPU is strongly recommended for
training).

## Data preparation

Generate the synthetic degraded (noisy) training/test set from clean SAM images:

```bash
python image_degradation.py --input "data/train" --output "degradation" --dataset "train"
python image_degradation.py --input "data/test"  --output "degradation" --dataset "test"
```

This produces paired clean/degraded images under `degradation/`, following the same
`dataroot_gt` / `dataroot_lq` layout the training configs expect.

## Training

Training follows HI-Diff's two-stage recipe:

1. **Stage 1** — train the regression backbone (Transformer + latent encoder):
   ```bash
   python train.py -opt configs/train/GoPro_S1.yml
   ```
2. **Stage 2** — freeze the stage-1 weights, train the diffusion prior + Mamba denoising block
   (set `pretrain_network_g` / `pretrain_network_le` in the stage-2 config to the stage-1
   checkpoints first):
   ```bash
   python train.py -opt configs/train/GoPro_S2.yml
   ```

Swap in a `RealBlur_*`/SAM-specific config as needed; see `configs/train/` for all available
presets. Update each config's `datasets.train.dataroot_gt` / `dataroot_lq` to point at your
generated data before running.

## Testing

```bash
python test.py -opt configs/test/GoPro.yml
```

Set `path.pretrain_network_g` / `pretrain_network_le_dm` / `pretrain_network_d` in the config to
your trained (or downloaded) checkpoints.

## Evaluation

- Full-reference metrics (PSNR/SSIM/LPIPS): `evaluate/IQA/metric.py`
- No-reference metrics (BRISQUE/NIQE/PIQE/RankIQA/MetaIQA): see `evaluate/NIQA/readme.md` for
  usage and pretrained-checkpoint links.

## Citation

If you use this code, please cite:

```bibtex
@article{nguyen2026samid,
  title={Unsupervised Hybrid Transformer--Mamba Diffusion for Depth-Dependent Image Denoising in Scanning Acoustic Microscopy},
  author={Nguyen, Trong Nhan and Doan, Vu Hoang Minh and Vo, Tan Hung and Pham, Dang Khoa and Nguyen, Quoc Dung and Vo, Truong Tien and Choi, Jaeyeop and Oh, Junghwan},
  journal={IEEE Transactions on Industrial Informatics},
  year={2026},
  doi={10.1109/TII.2025.3648838}
}
```

## Acknowledgements

This project's `denoising/` and `diffusion/` packages are an adapted, renamed fork of
[HI-Diff](https://github.com/zhengchen1999/HI-Diff) (Chen et al., NeurIPS 2023), which itself
builds on [BasicSR](https://github.com/XPixelGroup/BasicSR),
[Restormer](https://github.com/swz30/Restormer), and
[DiffIR](https://github.com/Zj-BinXia/DiffIR). The Mamba block draws on
[MambaIR](https://github.com/csguoh/MambaIR). Full attribution and licenses are in
[NOTICE](NOTICE). This code is released under the [Apache License 2.0](LICENSE).

If you use HI-Diff's underlying method, please also cite:

```bibtex
@inproceedings{chen2023hierarchical,
  title={Hierarchical Integration Diffusion Model for Realistic Image Deblurring},
  author={Chen, Zheng and Zhang, Yulun and Liu, Ding and Xia, Bin and Gu, Jinjin and Kong, Linghe and Yuan, Xin},
  booktitle={NeurIPS},
  year={2023}
}
```
