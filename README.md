# IntRo

Official PyTorch implementation of **IntRo** (KDD 2027): *Bridging Molecular Integrity and Relational Drug Roles for Drug-Drug Interaction Prediction*.

## Overview

IntRo addresses the long-tailed multi-relational DDI prediction problem through two complementary mechanisms:
1. **Integrity-Aware Molecular Pretraining**: Pretrains a bond-aware molecular GNN using Masked Atom Detection and Recovery (MADR) to preserve local atom-bond environments and Molecular Structure Consistency (MSC) to stabilize representation identity across perturbed views.
2. **Relational Role Modeling**: Leverages a Relational GCN (RGCN) to capture local interaction topology, complemented by an Interaction Profile Encoder (IPE) that maps degree-normalized relation-participation profiles into a structured relational role space via triplet-margin contrast.

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
IntRo/
├── config.json         # Model and training hyperparameters
├── data.py             # Dataset loader, graph construction, and profile calculation
├── model.py            # IntRo network architecture (RGCN, IPE, Discriminator, MLP)
├── pretrain.py         # Stage 1: Molecular self-supervised pretraining (MADR + MSC)
├── train.py            # Stage 2: Downstream multi-relational DDI training and evaluation
├── requirements.txt    # Package dependencies
└── README.md
```

## Quick Start

### 1. Train and Evaluate IntRo

Pretrained molecular embeddings are provided in `data/`, enabling direct training and evaluation of the downstream model:

```bash
# Train on Deng's dataset (65 interaction types)
python train.py --dataset Deng --data-root ./data --output-dir ./runs/deng_main

# Train on Ryu's dataset (86 interaction types)
python train.py --dataset Ryu --data-root ./data --output-dir ./runs/ryu_main
```

Output files saved to `--output-dir`:
- `best.pt`: Best model checkpoint selected by validation Macro-F1.
- `metrics.json`: Final test evaluation metrics.
- `predictions.npz`: Test predictions, ground-truth labels, and logits.
- `epochs.jsonl`: Training loss and validation progress logs.

### 2. Pretraining from Scratch (Optional)

To pretrain the molecular encoder from raw SMILES strings:

```bash
# Pretrain on Deng molecules
python pretrain.py --dataset Deng --data-root ./data --output-dir ./runs/pretrain_deng

# Pretrain on Ryu molecules
python pretrain.py --dataset Ryu --data-root ./data --output-dir ./runs/pretrain_ryu
```

To use newly pretrained embeddings in downstream training:

```bash
python train.py --dataset Deng --data-root ./data \
  --features ./runs/pretrain_deng/drug_embeddings.npy \
  --output-dir ./runs/deng_from_scratch
```
