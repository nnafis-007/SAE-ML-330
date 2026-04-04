# Sparse Autoencoder (SAE) for GPT-2 Interpretability

This project implements Sparse Autoencoders to interpret the internal representations of GPT-2's hidden layers, based on the methodology from "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning".

## Overview

Sparse Autoencoders help us understand what features neural networks learn by:
1. Collecting activations from a specific layer of GPT-2
2. Training a sparse autoencoder to reconstruct these activations
3. Analyzing the learned features to understand what concepts the model represents

## Project Structure

```
SAE-Project/
├── src/
│   ├── data_collection.py      # Collect activations from GPT-2
│   ├── sae_model.py             # Sparse Autoencoder architecture
│   ├── training.py              # Training loop and utilities
│   └── interpretation.py        # Analysis and visualization tools
├── notebooks/
│   └── tutorial.ipynb           # Step-by-step walkthrough
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

See `notebooks/tutorial.ipynb` for a complete walkthrough.

## Key Concepts

### 1. Activation Collection
We extract hidden layer activations from GPT-2 as it processes text data.

### 2. Sparse Autoencoder
The SAE learns to represent dense activations as sparse linear combinations of learned features.

### 3. Feature Interpretation
We analyze which features activate for specific inputs to understand what the model has learned.
