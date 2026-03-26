# Sparse Autoencoder (SAE) Training Framework

This repository provides tools for training Sparse Autoencoders on GPT-2 hidden layer activations, including the novel **Feature Choice SAE** (Ayonrinde, 2024).

## Overview

### Standard SAE
Traditional Sparse Autoencoders use L1 regularization to encourage sparsity. Each token can activate multiple features, but the L1 penalty limits the total activation magnitude.

### Feature Choice SAE (FC-SAE)
Feature Choice SAEs flip the sparsity constraint: instead of limiting features per token, they limit **tokens per feature**. Each feature must activate for exactly `m` tokens in a batch, ensuring:
- No dead features (all features are utilized)
- Adaptive computation per token (some tokens use more features)
- Better reconstruction with the same sparsity budget

**Mathematical Constraint:**
```
∑_j S_{i,j} = m, ∀i
```
Where `S_{i,j}` is a binary selection matrix indicating whether feature `i` is active for token `j`.

---

## Quick Start

### 1. Data Collection

Collect GPT-2 activations from a text dataset:

```bash
# Basic collection (50k samples from layer 8)
python collect_activation.py --layer 8 --samples 50000

# With custom dataset
python collect_activation.py --layer 8 --samples 100000 --dataset wikitext --dataset-config wikitext-103-v1

# Prepare data for training (split + normalize)
python collect_activation.py --layer 8 --samples 50000 --prepare-data --output activations.pt
```

### 2. Train Standard SAE

```bash
# Basic training
python run_sae.py --layer 8 --samples 50000 --epochs 30

# With dead feature resampling
python run_sae.py --layer 8 --samples 50000 --epochs 50 --resample-dead

# Skip collection if activations already exist
python run_sae.py --skip-collection --epochs 30
```

### 3. Train Feature Choice SAE

```bash
# Basic FC-SAE training
python run_FC_sae.py --layer 8 --samples 50000 --epochs 30 --m 32

# With custom parameters
python run_FC_sae.py --layer 8 --samples 100000 --epochs 50 --m 64 --expansion 32

# Use existing activations
python run_FC_sae.py --skip-collection --epochs 30 --m 32
```

---

## Script Reference

### collect_activation.py

Collects GPT-2 hidden layer activations for SAE training.

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | gpt2 | GPT-2 variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl) |
| `--layer` | 8 | Layer index to extract from (0-11 for gpt2) |
| `--samples` | 50000 | Maximum activation samples to collect |
| `--dataset` | openwebtext | HuggingFace dataset name |
| `--batch-size` | 16 | Batch size for GPT-2 forward pass |
| `--max-length` | 128 | Maximum sequence length |
| `--output` | activations.pt | Output file path |
| `--prepare-data` | False | Split and normalize data |
| `--normalize` | standardize | Normalization mode (standardize/center/none) |
| `--save-corpus` | False | Save text corpus for interpretation |

**Examples:**

```bash
# Collect with corpus for interpretation
python collect_activation.py --layer 8 --samples 50000 --save-corpus --corpus-output corpus.txt

# Different model and layer
python collect_activation.py --model gpt2-medium --layer 12 --samples 100000

# Prepare training data directly
python collect_activation.py --layer 8 --samples 50000 --prepare-data --prepared-output data.pt
```

---

### run_sae.py

Trains a standard Sparse Autoencoder with L1 sparsity.

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | 8 | GPT-2 layer to analyze |
| `--samples` | 50000 | Number of activation samples |
| `--expansion` | 16 | Expansion factor (d_hidden = expansion × d_model) |
| `--l1-coeff` | 3e-4 | L1 sparsity coefficient |
| `--epochs` | 30 | Maximum training epochs |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--resample-dead` | False | Enable dead feature resampling |
| `--checkpoint-dir` | checkpoints | Output directory |
| `--skip-collection` | False | Use existing activations.pt |

**Examples:**

```bash
# Standard training
python run_sae.py --layer 8 --samples 50000 --epochs 30

# Higher expansion factor
python run_sae.py --layer 8 --expansion 32 --l1-coeff 1e-4

# With dead feature resampling
python run_sae.py --layer 8 --samples 100000 --epochs 100 --resample-dead --resample-freq-threshold 0.0001
```

---

### run_FC_sae.py

Trains a Feature Choice SAE with per-feature sparsity constraint.

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | 8 | GPT-2 layer to analyze |
| `--samples` | 50000 | Number of activation samples |
| `--expansion` | 16 | Expansion factor |
| `--m` | 32 | **Tokens per feature** (key FC-SAE parameter) |
| `--aux-loss-coeff` | 1e-3 | Auxiliary loss coefficient |
| `--epochs` | 30 | Maximum training epochs |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--checkpoint-dir` | checkpoints_fc | Output directory |
| `--skip-collection` | False | Use existing activations |
| `--activations-path` | activations.pt | Path to activations file |

**Examples:**

```bash
# Basic FC-SAE
python run_FC_sae.py --layer 8 --samples 50000 --m 32

# Higher sparsity (fewer tokens per feature)
python run_FC_sae.py --layer 8 --samples 50000 --m 16 --expansion 32

# Lower sparsity (more tokens per feature)
python run_FC_sae.py --layer 8 --samples 50000 --m 64 --aux-loss-coeff 0
```

---

## Understanding the `m` Parameter (FC-SAE)

The `m` parameter controls how many tokens each feature can activate for in a single batch:

| m | Effect | Use Case |
|---|--------|----------|
| Small (8-16) | Very sparse, each feature highly selective | Memory-efficient, interpretable |
| Medium (32-64) | Balanced sparsity and reconstruction | General purpose |
| Large (128+) | Dense, many features per token | Maximum reconstruction quality |

**Relationship to batch size:**
- Generally, `m` should be less than batch_size
- A good starting point: `m = batch_size // 8`

---

## Architecture Comparison

| Aspect | Standard SAE | Feature Choice SAE |
|--------|--------------|-------------------|
| Sparsity constraint | L1 penalty on activations | Hard top-m selection per feature |
| Dead features | Common problem | Eliminated by design |
| Features per token | Variable (L1-controlled) | Variable (adaptive) |
| Tokens per feature | Variable | Fixed (exactly m) |
| Hyperparameters | l1_coeff | m_tokens_per_feature |

---

## Output Files

After training, you'll find these files in the checkpoint directory:

| File | Description |
|------|-------------|
| `best_model.pt` | Best model checkpoint (lowest validation loss) |
| `checkpoint_epoch_N.pt` | Periodic checkpoints |
| `training_history.png` | Training curves plot |
| `analysis_report.txt` | Detailed analysis (standard SAE only) |

---

## Loading a Trained Model

```python
import torch
import sys
sys.path.append("src")

# Standard SAE
from sae_model import SparseAutoencoder
checkpoint = torch.load("checkpoints/best_model.pt")
sae = SparseAutoencoder(
    d_model=checkpoint["hyperparameters"]["d_model"],
    d_hidden=checkpoint["hyperparameters"]["d_hidden"],
    l1_coeff=checkpoint["hyperparameters"]["l1_coeff"],
)
sae.load_state_dict(checkpoint["model_state_dict"])

# Feature Choice SAE
from fc_sae_model import FeatureChoiceSAE
checkpoint = torch.load("checkpoints_fc/best_model.pt")
fc_sae = FeatureChoiceSAE(
    d_model=checkpoint["hyperparameters"]["d_model"],
    d_hidden=checkpoint["hyperparameters"]["d_hidden"],
    m_tokens_per_feature=checkpoint["hyperparameters"]["m_tokens_per_feature"],
)
fc_sae.load_state_dict(checkpoint["model_state_dict"])
```

---

## Tips for Best Results

### Standard SAE
1. Start with `l1_coeff=3e-4` and adjust based on feature density
2. Enable `--resample-dead` for long training runs
3. Target 1-5% feature density for interpretability

### Feature Choice SAE
1. Set `m` based on batch size (start with `batch_size // 8`)
2. Use `aux_loss_coeff=0` for pure reconstruction
3. Monitor "Mean Features/Token" metric for adaptive computation

### General
1. Use `--normalize-mode standardize` for stable training
2. Larger datasets (100k+ samples) improve feature quality
3. Middle layers (4-8) often contain the most interpretable features

---

## References

- Original SAE: ["Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- Feature Choice SAE: Ayonrinde, K. (2024). "Feature Choice SAEs"
