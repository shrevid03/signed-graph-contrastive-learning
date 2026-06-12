<div align="center">

# Structural Balance aware Contrastive Learning for Signed Networks

**A polarity-aware graph contrastive learning framework for signed link prediction.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch)
![DGL](https://img.shields.io/badge/DGL-Graph%20Learning-6f42c1?style=for-the-badge)
![GNN](https://img.shields.io/badge/GNN-Signed%20Networks-2ea44f?style=for-the-badge)
![Status](https://img.shields.io/badge/Research-Manuscript%20in%20Progress-orange?style=for-the-badge)

</div>

---

## Overview

Signed networks model relationships that can be either **positive** or **negative** â for example, trust/distrust, support/opposition, friendship/hostility, or agreement/disagreement. Unlike ordinary graphs, signed graphs carry polarity, which makes representation learning more challenging.

This project implements a **Structural Balance aware Contrastive Learning Framework for Signed Networks**. The goal is to learn robust node embeddings for **signed link prediction**, where the model predicts whether a relationship between two nodes is positive or negative.

Instead of applying generic random augmentations to a graph, this framework uses signed-network-specific structural cues inspired by **structural balance theory**. The model learns from complementary graph views while preserving the meaning of positive and negative links.

---

## Motivation

Most graph contrastive learning methods are designed for unsigned graphs. In unsigned graphs, random edge dropping or feature masking can work well as augmentations. However, in signed networks, these operations can easily destroy important polarity patterns.

For example:

- A friend of a friend is often a friend.
- An enemy of an enemy may become a friend.
- A friend of an enemy may indicate a negative relationship.

These patterns are not just graph structure; they carry social meaning. If augmentations ignore edge signs, the model may learn from unrealistic views of the network.

This project focuses on a simple idea:

> Signed graph augmentations should preserve signed-network semantics instead of treating all edges as ordinary connections.

---

## Key Ideas

### 1. Polarity-Aware Encoding

Positive and negative edges are processed through separate GNN channels. This prevents positive and negative signals from being mixed too early and helps preserve the meaning of each relation type.

### 2. Structural-Balance-Aware Augmentation

The framework creates meaningful graph views using signed structural priors rather than purely random perturbations.

Main augmentation views include:

- **Signed random-walk-based view** to capture second-order balance patterns.
- **Centrality-based view** to highlight influential nodes and polarized hubs.

### 3. Joint Learning Objective

The model is trained using both supervised and self-supervised objectives:

- **Supervised link sign prediction loss** for the downstream prediction task.
- **Inter-view contrastive loss** to align representations across augmented graph views.
- **Intra-view contrastive loss** to keep final embeddings closer to positive-channel evidence and farther from negative-channel evidence.

---

## Problem Statement

Given a signed graph:

```text
G = (V, E, S)
```

where:

- `V` is the set of nodes,
- `E` is the set of observed edges,
- `S` represents the sign of an edge: `+1` for positive and `-1` for negative,

predict the sign of an unseen or held-out edge between two nodes.

---

## Architecture

```text
Signed Graph
     â
     âââ Positive Edge View âââº Positive GNN Encoder âââ
     â                                                  â
     âââ Negative Edge View âââº Negative GNN Encoder âââ¼âââº Fusion Head âââº Link Sign Predictor
     â                                                  â
     âââ Augmented Views ââââââº Contrastive Objectives ââ
```

The model learns node representations by combining:

- polarity-separated message passing,
- signed graph augmentations,
- contrastive alignment across graph views,
- and supervised link sign prediction.

---

## Dataset

The main experimental dataset is the **Bitcoin-OTC signed trust network**.

| Dataset | Nodes | Positive Links | Negative Links | Positive Ratio | Sparsity |
|---|---:|---:|---:|---:|---:|
| Bitcoin-OTC | 5,881 | 32,029 | 3,563 | 0.8999 | 0.1029% |

The dataset is challenging because it is:

- highly sparse,
- heavily imbalanced toward positive links,
- noisy due to real-world trust/distrust behavior,
- and long-tailed in node degree distribution.

---

## Results

The framework was evaluated using an 85/15 train-test split on Bitcoin network experiments.

| Metric | Best / Reported Value |
|---|---:|
| Accuracy | ~0.9496 |
| Precision | up to 0.961 |
| Recall | up to 0.9888 |
| Binary F1 | 0.9733 |
| Micro F1 | 0.9493 |

> Results may vary slightly depending on the split, random seed, augmentation configuration, and threshold selection.

The model performs strongly on operating-point metrics such as F1, precision, and recall, which are important for practical use cases like trust assessment, fraud detection, and moderation systems.

---

## Ablation and Robustness

The project includes experiments to understand the contribution of different components:

- baseline signed GNN setup,
- random-walk augmentation,
- centrality-based augmentation,
- dual-view contrastive learning,
- and sparsity-based robustness analysis.

A key focus is evaluating whether the model remains stable when the graph becomes more sparse or noisy.

---

## Tech Stack

- **Python**
- **PyTorch**
- **Deep Graph Library (DGL)**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **NetworkX**

## Why This Matters

Signed link prediction has applications in:

- trust and reputation systems,
- fraud and risk analysis,
- social network mining,
- recommendation systems,
- community analysis,
- and moderation pipelines.

In these settings, negative links are often rare but highly informative. A model that ignores polarity or class imbalance may perform well superficially while missing the most important minority patterns.

---

## Future Work

Planned extensions include:

- evaluating on more signed network datasets such as Slashdot, Epinions, Reddit, and Wiki-RFA,
- extending augmentations to multi-hop signed random walks,
- adding probabilistic thresholds during augmentation,
- improving scalability for larger graphs,
- supporting temporal signed networks,
- and studying weighted or directed signed relationships.

---

## Research Status

This work was developed as an academic research / thesis project at **ABV-IIITM Gwalior** under the supervision of **Dr. Roshni Chakraborty**.

A manuscript based on this work is currently in preparation.

---

## Contributors

- Nilay Srivastava
- Shreya Vidyadhar
- Aradhya Dixit
- Shruti Chauhan

Supervisor: **Dr. Roshni Chakraborty**

---

<div align="center">

**Built to preserve the meaning of signed relationships, not just the structure of a graph.**

</div>
