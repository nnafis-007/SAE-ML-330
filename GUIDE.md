# Sparse Autoencoder Project - Additional Documentation

## Quick Start Guide

### Option 1: Run the complete script
```bash
python run_sae.py --layer 8 --samples 50000 --epochs 30
```

### Option 2: Use the Jupyter notebook
```bash
jupyter notebook notebooks/tutorial.ipynb
```

### Option 3: Use the Python modules directly
```python
from src.data_collection import GPT2ActivationCollector
from src.sae_model import create_sae_for_gpt2
from src.training import SAETrainer

# Your custom code here
```

## Project Structure Explained

```
SAE-Project/
├── src/                          # Core implementation modules
│   ├── data_collection.py        # Extract activations from GPT-2
│   ├── sae_model.py              # Sparse Autoencoder architecture
│   ├── training.py               # Training loop and optimization
│   └── interpretation.py         # Analysis and visualization tools
│
├── notebooks/                    # Interactive tutorials
│   └── tutorial.ipynb            # Complete step-by-step guide
│
├── checkpoints/                  # Saved models and results (created during training)
│   ├── best_model.pt
│   ├── training_history.png
│   └── analysis_report.txt
│
├── run_sae.py                    # Command-line training script
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Understanding the Components

### 1. Data Collection (`data_collection.py`)

**Purpose**: Extract hidden layer activations from GPT-2

**Key classes**:
- `GPT2ActivationCollector`: Main class for collecting activations
  - Loads GPT-2 model
  - Hooks into specified layer
  - Processes text and extracts activations

**Usage**:
```python
collector = GPT2ActivationCollector(layer_index=8)
activations = collector.collect_from_dataset("openwebtext", num_texts=1000)
```

**Why this matters**: We need diverse activation examples to train the SAE effectively.

### 2. SAE Model (`sae_model.py`)

**Purpose**: Define the Sparse Autoencoder architecture

**Key classes**:
- `SparseAutoencoder`: The main model
  - Encoder: d_model → d_hidden (with ReLU)
  - Decoder: d_hidden → d_model
  - Loss: MSE reconstruction + L1 sparsity

**Mathematical formulation**:
```
Encode: f = ReLU(W_enc @ (x - b_pre) + b_enc)
Decode: x̂ = W_dec @ f + b_dec
Loss:   L = ||x - x̂||² + λ||f||₁
```

**Key hyperparameters**:
- `d_hidden`: Number of features (typically 8-64x d_model)
- `l1_coeff`: Sparsity penalty weight (1e-4 to 1e-2)

### 3. Training (`training.py`)

**Purpose**: Train the SAE on collected activations

**Key classes**:
- `SAETrainer`: Handles training loop
  - Adam optimizer
  - Early stopping
  - Checkpointing
  - Metric tracking

**Training process**:
1. Forward pass through SAE
2. Compute loss (MSE + L1)
3. Backward pass (gradients)
4. Update weights
5. **Normalize decoder** (critical!)

### 4. Interpretation (`interpretation.py`)

**Purpose**: Analyze what the SAE learned

**Key classes**:
- `FeatureAnalyzer`: Tools for understanding features
  - Find max activating examples
  - Compute feature statistics
  - Visualize feature behavior

**Key methods**:
- `find_max_activating_examples()`: Find texts that activate a feature
- `create_feature_dashboard()`: Visualize a single feature
- `get_reconstruction_quality()`: Measure SAE performance

## Hyperparameter Guide

### L1 Coefficient (`l1_coeff`)

Controls the sparsity-reconstruction tradeoff:

- **Too low (< 1e-5)**: Dense features, less interpretable
- **Sweet spot (1e-4 to 5e-4)**: Good balance
- **Too high (> 1e-2)**: Dead features, poor reconstruction

**How to tune**: Start with 3e-4, then:
- If feature density > 10%: Increase L1
- If too many dead features: Decrease L1

### Expansion Factor

Controls model capacity:

- **Small (4-8x)**: Fewer features, faster training
- **Medium (16-32x)**: Good default
- **Large (64-128x)**: More features, slower training

**Rule of thumb**: Start with 16x, increase if you need more capacity.

### Learning Rate

- **Standard**: 1e-3 (works well with Adam)
- **Faster**: 3e-3 (may be less stable)
- **Slower**: 3e-4 (more stable, takes longer)

### Batch Size

- **Small (64-128)**: More noisy gradients, faster per-epoch
- **Medium (256-512)**: Good default
- **Large (1024+)**: Smoother gradients, needs more memory

## Common Issues and Solutions

### Issue: Too many dead features (>30%)

**Causes**:
- L1 coefficient too high
- Not enough training data
- Poor initialization

**Solutions**:
1. Reduce `l1_coeff` by 2-3x
2. Collect more diverse activations
3. Train for more epochs

### Issue: Poor reconstruction (cosine sim < 0.85)

**Causes**:
- L1 coefficient too high
- Model too small
- Training not converged

**Solutions**:
1. Reduce `l1_coeff`
2. Increase `expansion_factor`
3. Train longer

### Issue: Features not interpretable

**Causes**:
- Not sparse enough
- Wrong layer
- Insufficient training data

**Solutions**:
1. Increase `l1_coeff` slightly
2. Try middle layers (6-9)
3. Collect more diverse texts

## Advanced Usage

### Custom Dataset

```python
# Use your own texts
my_texts = ["text 1", "text 2", ...]
activations = collector.collect_activations(my_texts)
```

### Different Layers

```python
# Compare multiple layers
for layer in [6, 7, 8, 9]:
    collector = GPT2ActivationCollector(layer_index=layer)
    # ... train SAE for each layer
```

### Feature Steering

```python
# Modify activations to test feature effects
with torch.no_grad():
    original = sae.encode(activation)
    modified = original.clone()
    modified[:, feature_idx] *= 2  # Double feature activation
    reconstructed = sae.decode(modified)
    # See how GPT-2 behaves with modified activation
```

## Performance Tips

### For faster training:
- Use smaller `expansion_factor` (8x instead of 16x)
- Reduce `num_samples` during development
- Use larger `batch_size` if you have GPU memory

### For better interpretability:
- Collect more diverse activations (100K+)
- Use middle layers (6-9)
- Fine-tune `l1_coeff` carefully
- Try different random seeds

## Citation

If you use this code for research, please cite the original paper:

```
@article{cunningham2023sparse,
  title={Towards Monosemanticity: Decomposing Language Models With Dictionary Learning},
  author={Cunningham, Hoagy and Ewart, Aidan and Riggs, Logan and Huben, Robert and Sharkey, Lee},
  journal={Transformer Circuits Thread},
  year={2023}
}
```

## Further Reading

- **Original paper**: https://transformer-circuits.pub/2023/monosemantic-features/
- **Sparse coding**: https://en.wikipedia.org/wiki/Sparse_coding
- **Mechanistic interpretability**: https://distill.pub/2020/circuits/

## Contributing

Feel free to extend this codebase:
- Add support for other models (GPT-J, LLaMA, etc.)
- Implement advanced features (top-k activation, etc.)
- Improve visualization tools
- Add automated feature labeling

## License

This code is for educational purposes. Please see original paper for attribution.
