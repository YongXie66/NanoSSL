# NanoSSL

[Chinese README](./README_cn.md)

**NanoSSL: Attention-based Self-Supervised Learning for Protein Variant Identification with Solid-State Nanopores**

NanoSSL is a self-supervised learning framework for nanopore-based protein signal analysis. It is designed to learn robust representations from noisy single-molecule current traces and improve downstream protein identification when labeled data are limited.

## Paper

This repository accompanies the following paper:

> **NanoSSL: attention mechanism-based self-supervised learning method for protein variant identification using solid-state nanopores**  
> *Bioinformatics*, Volume 42, Issue 1, January 2026

- Paper: https://academic.oup.com/bioinformatics/article/42/1/btaf657/8371887
- DOI: https://doi.org/10.1093/bioinformatics/btaf657
- Code: https://github.com/YongXie66/NanoSSL

## Why NanoSSL?

Protein signals measured by nanopores are often noisy, variable, and expensive to annotate. NanoSSL addresses this challenge with a two-stage learning pipeline that combines self-supervised pretraining and supervised fine-tuning.

### Highlights

- Self-supervised pretraining for low-label settings
- Attention-based sequence modeling with a Transformer encoder
- Designed for nanopore protein current traces, including solid-state nanopore measurements
- Evaluated on public ONT data and amyloid-beta variant identification tasks

## Results at a Glance

According to the paper, NanoSSL achieved strong performance on public nanopore benchmarks:

- `0.951 +- 0.001` accuracy for barcode recognition
- `0.982 +- 0.001` accuracy for binary protein identification

The paper also reports that NanoSSL achieves state-of-the-art performance for amyloid-beta variant identification on solid-state nanopore data, especially on challenging mutant discrimination tasks such as `E22G` and `G37R`.

## Method Overview

NanoSSL follows a two-stage training strategy:

### 1. Self-supervised pretraining

Each nanopore signal is divided into non-overlapping local fragments. A subset of fragments is masked, and the model learns to predict masked representations from visible context.

### 2. Supervised fine-tuning

After pretraining, the encoder is fine-tuned for downstream classification, enabling better performance with limited labeled data.

### Core ideas reflected in the code

- Fragment-based tokenization via `wave_length`
- Transformer encoder for contextual representation learning
- Masked self-supervised objective controlled by `mask_ratio`
- Momentum-style target encoder for stable representation learning
- Classification head for downstream prediction

## Repository Overview

```text
NanoSSL/
├── dataset/
│   └── ont/                  # Example ONT dataset (.pt files)
├── model/
│   ├── NANOSSL.py            # Main NanoSSL model
│   └── layers.py             # Transformer and related layers
├── utils/
│   ├── args.py               # Argument parsing and dataset dispatch
│   ├── dataset.py            # Dataset wrapper
│   ├── datautils.py          # Data loading utilities
│   └── util.py               # Training and evaluation helpers
├── train.py                  # Main training entry
├── trainer.py                # Pretraining and fine-tuning pipeline
├── train.sh                  # Example training script
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YongXie66/NanoSSL.git
cd NanoSSL
```

Create a Python environment and install dependencies:

```bash
conda create -n nanossl python=3.10
conda activate nanossl
pip install -r requirements.txt
```

## Quick Start

### Run the included ONT example

```bash
python train.py \
  --dataset ont \
  --data_path dataset/ont/ \
  --save_path exp/ONT/test \
  --device cuda:0
```

Or use the provided shell script:

```bash
bash train.sh
```

### Run on amyloid-beta variant data

```bash
python train.py \
  --dataset amyloid \
  --data_path data/amyloid/ \
  --mutation E22G \
  --save_path exp/amyloid/E22G \
  --device cuda:0
```

## Data

### Included data

The repository includes an example ONT dataset under:

```text
dataset/ont/train.pt
dataset/ont/val.pt
dataset/ont/test.pt
```

### Amyloid data

For amyloid-beta experiments, the code expects MATLAB files named in the following style:

```text
data/amyloid/
├── Native_BetaAmyloid.mat
├── E22G_BetaAmyloid.mat
├── G37R_BetaAmyloid.mat
└── ...
```

## Training Notes

- `train.py` runs the full pipeline: pretraining first, then fine-tuning.
- The actual data loader is selected by `--data_path`.
- Main outputs are saved to `--save_path`, including configuration, checkpoints, and result summaries.
- The default settings are oriented toward GPU training.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{xie2026nanossl,
  author = {Xie, Yong and Li, Jindong and Zhang, Ziyan and Meng, Bin and Dai, Shuaijian and Zhou, Yuchen and Kennedy, Eamonn and Jiao, Niandong and Chen, Haobin and Dong, Zhuxin},
  title = {NanoSSL: attention mechanism-based self-supervised learning method for protein variant identification using solid-state nanopores},
  journal = {Bioinformatics},
  volume = {42},
  number = {1},
  pages = {btaf657},
  year = {2026},
  doi = {10.1093/bioinformatics/btaf657}
}
```

## Acknowledgement

If NanoSSL is helpful for your work, please consider starring the repository and citing the paper.
