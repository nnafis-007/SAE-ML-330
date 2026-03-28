# Step-by-Step Guide: Running SAE Training on Kaggle

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

---

## Step 4: Set Up Environment (Optional)

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

## Step 6: Collect Activations


```python
# Cell 3: Collect activations
!python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --dataset openwebtext \
    --batch-size 32 \
    --max-length 128 \
    --output activations.pt \
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
    --m 8 \
    --aux-loss-coeff 1e-3 \
    --lr 5e-4 \
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