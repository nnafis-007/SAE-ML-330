# Quick Reference: Kaggle Training Commands

## One-Cell Quick Start 🚀

```python
# Complete training pipeline in one cell
!pip install -q transformers datasets torch tqdm matplotlib

# Collect data
!python collect_activation.py --layer 8 --samples 50000 --prepare-data

# Train both models
!python run_sae.py --skip-collection --epochs 30 --checkpoint-dir ckpt_std
!python run_FC_sae.py --skip-collection --epochs 30 --m 32 --checkpoint-dir ckpt_fc

# Package results
!mkdir results && cp ckpt_std/best_model.pt results/std.pt && cp ckpt_fc/best_model.pt results/fc.pt
!tar -czf results.tar.gz results/
```

---

## Essential Commands

### 1. Data Collection
```bash
# Minimal (fastest)
python collect_activation.py --layer 8 --samples 25000

# Standard (balanced)
python collect_activation.py --layer 8 --samples 50000 --prepare-data

# Full (best quality)
python collect_activation.py --layer 8 --samples 100000 --prepare-data --save-corpus
```

### 2. Standard SAE Training
```bash
# Quick test (5-10 min)
python run_sae.py --skip-collection --epochs 5 --batch-size 256

# Standard run (30 min)
python run_sae.py --skip-collection --epochs 30

# With dead feature resampling
python run_sae.py --skip-collection --epochs 50 --resample-dead
```

### 3. Feature Choice SAE Training
```bash
# Quick test
python run_FC_sae.py --skip-collection --epochs 5 --m 32

# Standard run
python run_FC_sae.py --skip-collection --epochs 30 --m 32

# Sparse variant (fewer tokens per feature)
python run_FC_sae.py --skip-collection --epochs 30 --m 16

# Dense variant (more tokens per feature)
python run_FC_sae.py --skip-collection --epochs 30 --m 64
```

---

## Parameter Recommendations

| Scenario | Command |
|----------|---------|
| **Fast test** | `--samples 10000 --epochs 5 --batch-size 128` |
| **Balanced** | `--samples 50000 --epochs 30 --batch-size 256` |
| **High quality** | `--samples 100000 --epochs 50 --batch-size 256` |
| **Low memory** | `--batch-size 64 --collection-batch-size 8` |

---

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| OOM error | `--batch-size 64` or `--expansion 8` |
| Slow collection | `--shuffle-buffer-size 0` |
| Session timeout | Start with `--epochs 5` first |
| Module not found | `sys.path.append('/kaggle/working/src')` |

---

## Monitoring Commands

```python
# Check GPU
!nvidia-smi

# Check disk space
!df -h

# Monitor training (in another cell)
!tail -f /kaggle/working/nohup.out

# Check file sizes
!du -sh checkpoints_* activations.pt
```

---

## Download Results

```python
# Option 1: Single archive
!tar -czf results.tar.gz checkpoints_*/best_model.pt
from IPython.display import FileLink
display(FileLink('results.tar.gz'))

# Option 2: Individual files
display(FileLink('checkpoints_standard/best_model.pt'))
display(FileLink('checkpoints_fc/best_model.pt'))
```

---

## Time Estimates (on Kaggle T4 GPU)

| Task | Time |
|------|------|
| Collect 50k samples | 15 min |
| Train 30 epochs (standard) | 30 min |
| Train 30 epochs (FC-SAE) | 30 min |
| **Total workflow** | **~1.5 hours** |

---

## Memory Usage

| Configuration | GPU Memory | Disk Space |
|--------------|------------|------------|
| samples=50k, batch=256 | ~8 GB | ~2 GB |
| samples=100k, batch=256 | ~10 GB | ~4 GB |
| expansion=32 | +2 GB | +500 MB |
