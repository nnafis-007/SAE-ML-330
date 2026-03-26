# Step-by-Step Guide: Running SAE Training on Kaggle

This guide walks you through training Sparse Autoencoders (both standard and Feature Choice) on Kaggle's free GPU environment.

---

## Prerequisites

1. A Kaggle account (free): https://www.kaggle.com/
2. Phone verification enabled (required for GPU access)

---

## Step 1: Create a New Kaggle Notebook

1. Go to https://www.kaggle.com/
2. Click **"Code"** in the left sidebar
3. Click **"+ New Notebook"** (top right)
4. Your notebook will open

---

## Step 2: Enable GPU Acceleration

**IMPORTANT:** Enable GPU before running any code.

1. Click **"Accelerator"** in the right sidebar (under Settings)
2. Select **"GPU T4 x2"** (free tier)
3. Click **"Save"**

The notebook will restart with GPU enabled.

---

## Step 3: Upload Your Code to Kaggle

### Option A: Upload as Dataset (Recommended)

1. **Compress your code locally:**
   ```bash
   cd /home/nn/buet-classes/330-ML-lab/SAE-OOM-fixed
   zip -r sae_code.zip src/ run_sae.py run_FC_sae.py collect_activation.py
   ```

2. **Create a Kaggle Dataset:**
   - Go to https://www.kaggle.com/datasets
   - Click **"New Dataset"**
   - Upload `sae_code.zip`
   - Set title: "SAE Training Code"
   - Click **"Create"**

3. **Add dataset to your notebook:**
   - In your notebook, click **"+ Add Data"** (right sidebar)
   - Search for your dataset name
   - Click **"Add"**

### Option B: Clone from GitHub (If code is in a repo)

If your code is in a GitHub repository:

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

### Option C: Upload Files Directly

Copy and paste the code files into notebook cells (works for small projects).

---

## Step 4: Set Up Environment

Run these commands in the first cell of your notebook:

```python
# Cell 1: Install dependencies
!pip install -q transformers datasets torch tqdm matplotlib

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## Step 5: Extract Code (if uploaded as zip)

```python
# Cell 2: Extract code
!unzip -q /kaggle/input/sae-training-code/sae_code.zip -d /kaggle/working/
%cd /kaggle/working/

# Verify files
!ls -la
!ls -la src/
```

---

## Step 6: Collect Activations

**Estimated time:** 10-20 minutes for 50k samples

```python
# Cell 3: Collect activations
!python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --dataset openwebtext \
    --batch-size 32 \
    --max-length 128 \
    --output activations.pt \
    --prepare-data \
    --prepared-output prepared_data.pt
```

**What this does:**
- Loads GPT-2 on GPU
- Extracts 50,000 activation vectors from layer 8
- Saves to `activations.pt` (for raw data)
- Prepares train/val splits to `prepared_data.pt`

**Memory tip:** If you get OOM errors, reduce `--batch-size` to 16 or 8.

---

## Step 7: Train Standard SAE

**Estimated time:** 20-40 minutes

```python
# Cell 4: Train standard SAE
!python run_sae.py \
    --skip-collection \
    --layer 8 \
    --epochs 30 \
    --batch-size 256 \
    --expansion 16 \
    --l1-coeff 3e-4 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints_standard \
    --save-every 10
```

**What this does:**
- Trains a standard SAE with L1 sparsity
- Saves checkpoints every 10 epochs
- Creates `checkpoints_standard/best_model.pt`
- Generates training curves plot

**To monitor progress:**
The script prints metrics every epoch. Watch for:
- **Val Loss**: Should decrease (lower = better)
- **Feature Density**: Should be 1-5% for good sparsity

---

## Step 8: Train Feature Choice SAE

**Estimated time:** 20-40 minutes

```python
# Cell 5: Train Feature Choice SAE
!python run_FC_sae.py \
    --skip-collection \
    --layer 8 \
    --epochs 30 \
    --batch-size 256 \
    --expansion 16 \
    --m 32 \
    --aux-loss-coeff 1e-3 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints_fc \
    --save-every 10
```

**What this does:**
- Trains Feature Choice SAE with m=32 tokens per feature
- No dead features by design
- Creates `checkpoints_fc/best_model.pt`

**Parameter tuning:**
- Increase `--m` (e.g., 64) for denser representations
- Decrease `--m` (e.g., 16) for sparser representations
- Set `--aux-loss-coeff 0` to disable auxiliary loss

---

## Step 9: Compare Results

```python
# Cell 6: Load and compare models
import torch
import sys
sys.path.append('/kaggle/working/src')

from sae_model import SparseAutoencoder
from fc_sae_model import FeatureChoiceSAE

# Load standard SAE
std_checkpoint = torch.load('checkpoints_standard/best_model.pt')
std_sae = SparseAutoencoder(
    d_model=std_checkpoint['hyperparameters']['d_model'],
    d_hidden=std_checkpoint['hyperparameters']['d_hidden'],
    l1_coeff=std_checkpoint['hyperparameters']['l1_coeff']
)
std_sae.load_state_dict(std_checkpoint['model_state_dict'])

# Load FC-SAE
fc_checkpoint = torch.load('checkpoints_fc/best_model.pt')
fc_sae = FeatureChoiceSAE(
    d_model=fc_checkpoint['hyperparameters']['d_model'],
    d_hidden=fc_checkpoint['hyperparameters']['d_hidden'],
    m_tokens_per_feature=fc_checkpoint['hyperparameters']['m_tokens_per_feature']
)
fc_sae.load_state_dict(fc_checkpoint['model_state_dict'])

print("=" * 60)
print("TRAINING COMPARISON")
print("=" * 60)

print("\nStandard SAE:")
print(f"  Final Val Loss: {std_checkpoint['history']['val_loss'][-1]:.6f}")
print(f"  Feature Density: {std_checkpoint['history']['feature_density'][-1]:.2%}")

print("\nFeature Choice SAE:")
print(f"  Final Val Loss: {fc_checkpoint['history']['val_loss'][-1]:.6f}")
print(f"  Feature Density: {fc_checkpoint['history']['feature_density'][-1]:.2%}")
print(f"  Mean Features/Token: {fc_checkpoint['history']['mean_features_per_token'][-1]:.1f}")
```

---

## Step 10: Visualize Training Curves

```python
# Cell 7: Plot training history
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Standard SAE
epochs_std = range(1, len(std_checkpoint['history']['train_loss']) + 1)
axes[0].plot(epochs_std, std_checkpoint['history']['train_loss'], label='Train', alpha=0.7)
axes[0].plot(epochs_std, std_checkpoint['history']['val_loss'], label='Val', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Standard SAE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# FC-SAE
epochs_fc = range(1, len(fc_checkpoint['history']['train_loss']) + 1)
axes[1].plot(epochs_fc, fc_checkpoint['history']['train_loss'], label='Train', alpha=0.7)
axes[1].plot(epochs_fc, fc_checkpoint['history']['val_loss'], label='Val', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Feature Choice SAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Step 11: Download Results

### Method 1: Download via Notebook Interface

1. Check the **Output** section in the right sidebar
2. Files will appear after training completes
3. Click the download icon next to each file

### Method 2: Create a Kaggle Dataset

```python
# Cell 8: Prepare files for download
!mkdir -p /kaggle/working/results
!cp checkpoints_standard/best_model.pt /kaggle/working/results/standard_sae.pt
!cp checkpoints_fc/best_model.pt /kaggle/working/results/fc_sae.pt
!cp checkpoints_standard/training_history.png /kaggle/working/results/std_history.png
!cp checkpoints_fc/training_history.png /kaggle/working/results/fc_history.png
!cp comparison.png /kaggle/working/results/
!cp activations.pt /kaggle/working/results/

# Create a single archive
!cd /kaggle/working && tar -czf results.tar.gz results/
print("Download results.tar.gz from the Output section!")
```

### Method 3: Direct Download

```python
# Download individual files
from IPython.display import FileLink

display(FileLink('checkpoints_standard/best_model.pt'))
display(FileLink('checkpoints_fc/best_model.pt'))
display(FileLink('comparison.png'))
```

---

## Complete Kaggle Notebook Template

Here's a complete notebook you can copy-paste:

```python
# ==============================================================================
# CELL 1: Setup
# ==============================================================================
!pip install -q transformers datasets torch tqdm matplotlib

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ==============================================================================
# CELL 2: Upload and extract code
# ==============================================================================
# If uploaded as dataset:
!unzip -q /kaggle/input/YOUR_DATASET/sae_code.zip -d /kaggle/working/
%cd /kaggle/working/

# Or clone from GitHub:
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# %cd YOUR_REPO

# ==============================================================================
# CELL 3: Collect data
# ==============================================================================
!python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --batch-size 32 \
    --prepare-data \
    --output activations.pt

# ==============================================================================
# CELL 4: Train Standard SAE
# ==============================================================================
!python run_sae.py \
    --skip-collection \
    --epochs 30 \
    --batch-size 256 \
    --checkpoint-dir checkpoints_standard

# ==============================================================================
# CELL 5: Train Feature Choice SAE
# ==============================================================================
!python run_FC_sae.py \
    --skip-collection \
    --epochs 30 \
    --batch-size 256 \
    --m 32 \
    --checkpoint-dir checkpoints_fc

# ==============================================================================
# CELL 6: Compare and visualize
# ==============================================================================
import sys
sys.path.append('/kaggle/working/src')
import matplotlib.pyplot as plt

# Load checkpoints
std_ckpt = torch.load('checkpoints_standard/best_model.pt')
fc_ckpt = torch.load('checkpoints_fc/best_model.pt')

# Print comparison
print("\nStandard SAE Val Loss:", std_ckpt['history']['val_loss'][-1])
print("FC-SAE Val Loss:", fc_ckpt['history']['val_loss'][-1])

# Plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(std_ckpt['history']['val_loss'], label='Standard SAE')
ax[1].plot(fc_ckpt['history']['val_loss'], label='FC-SAE')
plt.savefig('results.png')
plt.show()

# ==============================================================================
# CELL 7: Package results
# ==============================================================================
!mkdir results
!cp checkpoints_standard/best_model.pt results/standard_sae.pt
!cp checkpoints_fc/best_model.pt results/fc_sae.pt
!tar -czf results.tar.gz results/
print("✓ Download results.tar.gz from Output panel")
```

---

## Troubleshooting

### Problem: Out of Memory (OOM)

**Solutions:**
```python
# Reduce batch sizes
--batch-size 128          # Instead of 256
--collection-batch-size 8  # Instead of 16

# Reduce samples
--samples 25000           # Instead of 50000

# Reduce expansion factor
--expansion 8             # Instead of 16
```

### Problem: Session Timeout

Kaggle notebooks timeout after **12 hours** or if inactive.

**Solution:** Run smaller experiments first:
```python
# Quick test run (5-10 minutes)
!python run_FC_sae.py --skip-collection --epochs 5 --samples 10000
```

### Problem: Slow Data Loading

**Solution:** Use `--shuffle-buffer-size 0` to disable shuffling:
```python
!python collect_activation.py --shuffle-buffer-size 0 --samples 50000
```

### Problem: Can't find modules

**Solution:** Add src to path:
```python
import sys
sys.path.append('/kaggle/working/')
sys.path.append('/kaggle/working/src/')
```

---

## Expected Runtime

| Task | GPU | CPU |
|------|-----|-----|
| Data Collection (50k) | 15 min | 60 min |
| Standard SAE (30 epochs) | 30 min | 3+ hours |
| FC-SAE (30 epochs) | 30 min | 3+ hours |
| **Total** | **~1.5 hours** | **~5 hours** |

---

## Tips for Success

1. **Enable GPU first** - Don't forget this step!
2. **Start small** - Test with `--epochs 5` first
3. **Monitor memory** - Use `!nvidia-smi` to check GPU usage
4. **Save often** - Use `--save-every 5` to save more checkpoints
5. **Download early** - Download results before session expires

---

## Next Steps

After training:
1. Download the model checkpoints
2. Run interpretation analysis locally
3. Compare feature quality between standard and FC-SAE
4. Try different hyperparameters (m, expansion, etc.)

---

## Additional Resources

- Kaggle GPU Docs: https://www.kaggle.com/docs/notebooks
- Troubleshooting: See `summary.md` in this repo
- Example notebook: [Coming soon - link to public Kaggle notebook]
