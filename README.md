# IntRo

Official PyTorch implementation of **IntRo** (KDD 2027): *Bridging Molecular Integrity and Relational Drug Roles for Drug–Drug Interaction Prediction*.

## Overview

IntRo combines two complementary components for long-tailed multi-relational DDI prediction:

1. **Integrity-Aware Molecular Pretraining**: Pretrains a bond-aware molecular GNN with Masked Atom Detection and Recovery (MADR) and Molecular Structure Consistency (MSC).
2. **Relational Drug Role Modeling**: Uses a two-layer RGCN for local interaction topology together with an Interaction Pattern Encoder (IPE) that maps degree-normalized relation-participation profiles to role embeddings under a triplet-margin objective.

## Installation

```bash
pip install -r requirements.txt
```

Install the `torch` and `torch-geometric` builds matching your CUDA version if needed.

## Repository Structure

```text
IntRo/
├── config.json         # Pretraining and downstream hyperparameters
├── data.py             # Dataset loading, training graph construction, and relation profiles
├── model.py            # IntRo model (RGCN, IPE, DGI discriminator, prediction MLP)
├── pretrain.py         # Molecular pretraining (MADR + MSC)
├── train.py            # Downstream multi-relational DDI training and evaluation
├── data/
│   ├── Dengsdataset/   # Deng's benchmark splits, SMILES, and pretrained embeddings
│   └── Ryusdataset/    # Ryu's benchmark splits, SMILES, and pretrained embeddings
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### 1. Downstream Training and Evaluation

Pretrained molecular embeddings are included under `data/`, so you can train and evaluate the downstream model directly:

```bash
# Deng's dataset (65 interaction types)
python train.py --dataset Deng --data-root ./data --output-dir ./runs/deng_main

# Ryu's dataset (86 interaction types)
python train.py --dataset Ryu --data-root ./data --output-dir ./runs/ryu_main
```

Outputs written to `--output-dir`:
- `best.pt`: Best checkpoint selected by validation Macro-F1.
- `config.json`: Effective run configuration.
- `metrics.json`: Final test evaluation metrics.
- `predictions.npz`: Test predictions, ground-truth labels, and logits.
- `epochs.jsonl`: Per-epoch training loss and validation metrics.

### 2. Molecular Pretraining from Scratch (Optional)

To pretrain the molecular encoder from SMILES strings:

```bash
python pretrain.py --dataset Deng --data-root ./data --output-dir ./runs/pretrain_deng
python pretrain.py --dataset Ryu --data-root ./data --output-dir ./runs/pretrain_ryu
```

This writes `drug_embeddings.npy`, `drug_ids.npy`, `pretrain_model.pt`, and `epochs.jsonl` to `--output-dir`. To train the downstream model with newly pretrained embeddings:

```bash
python train.py --dataset Deng --data-root ./data \
  --features ./runs/pretrain_deng/drug_embeddings.npy \
  --output-dir ./runs/deng_from_scratch
```

## Datasets

The benchmark splits under `data/` follow the transductive multi-class DDI benchmarks introduced by Deng et al. (2020) and Ryu et al. (2018).

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{qu2027intro,
  title     = {Bridging Molecular Integrity and Relational Drug Roles for Drug--Drug Interaction Prediction},
  author    = {Qu, Yanchen and Zhang, Yijia and Cui, Hai and Duan, Meiyu and Li, Meilin and Wang, Shilong},
  booktitle = {Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '27)},
  year      = {2027}
}
```
