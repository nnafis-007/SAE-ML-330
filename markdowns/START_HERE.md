# 🎯 GETTING STARTED - Read This First!

Welcome to the Sparse Autoencoder (SAE) project for interpreting GPT-2 hidden layers!

## 📚 What This Project Does

This project implements the methodology from the paper **"Towards Monosemanticity"** to:
1. Extract internal representations from GPT-2
2. Train a Sparse Autoencoder to decompose them into interpretable features
3. Analyze what concepts GPT-2 has learned

## 🚀 Quick Start (3 Options)

### Option 1: Interactive Tutorial (RECOMMENDED for learning)
```bash
jupyter notebook notebooks/tutorial.ipynb
```
**Best for**: Understanding each step in detail with explanations and visualizations.

### Option 2: Command-Line Script (RECOMMENDED for quick results)
```bash
python run_sae.py --layer 8 --samples 50000 --epochs 30
```
**Best for**: Getting results quickly without coding.

### Option 3: Python API (RECOMMENDED for customization)
```python
from src.data_collection import GPT2ActivationCollector
from src.training import train_sae

# Your custom code...
```
**Best for**: Building your own analysis pipeline.

## 📁 Project Files Guide

**Start here:**
- `README.md` - Project overview
- `START_HERE.md` - This file (quick start)
- `notebooks/tutorial.ipynb` - Complete walkthrough ⭐

**Reference:**
- `GUIDE.md` - Detailed documentation
- `requirements.txt` - Python dependencies

**Core implementation:**
- `src/data_collection.py` - Extract GPT-2 activations
- `src/sae_model.py` - Sparse Autoencoder model
- `src/training.py` - Training loop
- `src/interpretation.py` - Analysis tools

**Utilities:**
- `run_sae.py` - Command-line interface
- `visualize_architecture.py` - Generate diagrams

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('Transformers: OK')"
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recommended)
- GPU recommended but not required

## 🎓 Learning Path

### Beginner Path (Understanding the concepts)
1. Read `README.md` to understand what we're doing
2. Open `notebooks/tutorial.ipynb`
3. Run cells one by one, reading explanations
4. Experiment with different parameters

**Time**: 2-3 hours

### Intermediate Path (Getting results)
1. Skim `README.md` and `GUIDE.md`
2. Run: `python run_sae.py --help`
3. Train: `python run_sae.py --layer 8 --samples 50000 --epochs 30`
4. Check `checkpoints/` for results

**Time**: 30 minutes + training time (30-60 min)

### Advanced Path (Custom analysis)
1. Read `GUIDE.md` thoroughly
2. Study `src/` modules
3. Build custom pipeline using the API
4. Extend with your own features

**Time**: Variable

## 🔍 What to Expect

### Training takes time
- Small experiment (20K samples): ~10 minutes
- Medium experiment (50K samples): ~30 minutes
- Large experiment (100K+ samples): ~1-2 hours

### Results location
After training, check `checkpoints/`:
- `best_model.pt` - Trained SAE model
- `training_history.png` - Loss curves
- `analysis_report.txt` - Feature statistics

### Good results look like:
- ✅ Feature density: 1-5%
- ✅ Cosine similarity: >0.90
- ✅ Dead features: <20%
- ✅ Loss decreasing steadily

## 🎯 Step-by-Step First Run

### 1. Set up environment
```bash
# Clone/navigate to project
cd SAE-Project

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate architecture diagrams (optional but helpful)
```bash
python visualize_architecture.py
```
This creates visual diagrams explaining the architecture.

### 3. Run a small test
```bash
python run_sae.py --layer 8 --samples 10000 --epochs 10
```
This runs a quick test (~5-10 minutes) to verify everything works.

### 4. Check results
```bash
ls checkpoints/
cat checkpoints/analysis_report.txt
```

### 5. If successful, run full training
```bash
python run_sae.py --layer 8 --samples 50000 --epochs 30
```

## 🐛 Troubleshooting

### ImportError: No module named 'transformers'
```bash
pip install transformers
```

### CUDA out of memory
Reduce batch size:
```bash
python run_sae.py --batch-size 128  # Instead of default 256
```

### Too slow on CPU
Use fewer samples:
```bash
python run_sae.py --samples 10000  # Instead of 50000
```

### Poor results (high dead features)
Try lower L1 coefficient:
```bash
python run_sae.py --l1-coeff 1e-4  # Instead of default 3e-4
```

## 📊 Understanding the Output

### Terminal output explains:
1. **Data collection**: How many activations were collected
2. **Training progress**: Loss, feature density per epoch
3. **Final metrics**: Overall performance
4. **File locations**: Where outputs are saved

### Key metrics:
- **Validation Loss**: Lower is better (target: <0.1)
- **Feature Density**: Percent of non-zero features (target: 1-5%)
- **Cosine Similarity**: How well we reconstruct (target: >0.90)
- **Dead Features**: Features that never activate (target: <20%)

## 🎨 Visualization

The tutorial notebook includes:
- Training curves (loss over time)
- Feature activation distributions
- Reconstruction quality plots
- Individual feature dashboards
- Feature correlation heatmaps

## 📖 Next Steps After First Run

1. **Analyze features**: Open analysis report, check feature dashboards
2. **Try different layers**: Compare layers 6, 7, 8, 9
3. **Tune hyperparameters**: Experiment with L1 coefficient
4. **Scale up**: Use more data (100K+ samples)
5. **Deep dive**: Use interpretation tools in notebook

## 🤝 Getting Help

### Check these first:
1. `GUIDE.md` - Comprehensive documentation
2. Comments in source code - Detailed explanations
3. Tutorial notebook - Step-by-step examples

### Common questions:
**Q: How long should training take?**
A: 30-60 minutes for 50K samples on GPU, 2-3 hours on CPU.

**Q: What layer should I use?**
A: Layer 8 (middle layer) works well. Try 6-9 for comparison.

**Q: How many samples do I need?**
A: Minimum 20K, recommended 50K-100K, ideal 500K+.

**Q: My features aren't interpretable, why?**
A: Try adjusting L1 coefficient, collecting more data, or using different layer.

## 🎯 Success Criteria

You've successfully completed the project when you can:
- ✅ Extract activations from GPT-2
- ✅ Train a sparse autoencoder
- ✅ Achieve >90% reconstruction quality
- ✅ Get 1-5% feature density
- ✅ Find interpretable features (e.g., geographic entities, numbers, syntax)

## 🚀 Ready to Start?

Choose your path:
```bash
# Interactive learning
jupyter notebook notebooks/tutorial.ipynb

# Quick results
python run_sae.py

# Custom code
python -c "from src import *; help(GPT2ActivationCollector)"
```

Good luck! 🎉
