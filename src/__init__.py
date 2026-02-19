"""
Sparse Autoencoder (SAE) for GPT-2 Interpretability

This package implements Sparse Autoencoders to interpret hidden layer
representations in GPT-2, based on the methodology from "Towards 
Monosemanticity: Decomposing Language Models With Dictionary Learning".

Main components:
- data_collection: Extract activations from GPT-2
- sae_model: Sparse Autoencoder architecture
- training: Training loop and optimization
- interpretation: Analysis and visualization tools

Example usage:
    >>> from src.data_collection import GPT2ActivationCollector
    >>> from src.sae_model import create_sae_for_gpt2
    >>> from src.training import SAETrainer
    >>> 
    >>> # Collect activations
    >>> collector = GPT2ActivationCollector(layer_index=8)
    >>> activations = collector.collect_from_dataset("openwebtext", num_texts=1000)
    >>> 
    >>> # Create and train SAE
    >>> sae = create_sae_for_gpt2(expansion_factor=16, l1_coeff=3e-4)
    >>> trainer = SAETrainer(sae, train_data, val_data)
    >>> history = trainer.train(num_epochs=30)
"""

__version__ = "1.0.0"
__author__ = "SAE Interpretability Project"

from .data_collection import GPT2ActivationCollector, prepare_training_data
from .sae_model import SparseAutoencoder, create_sae_for_gpt2
from .training import SAETrainer, train_sae
from .interpretation import FeatureAnalyzer

__all__ = [
    "GPT2ActivationCollector",
    "prepare_training_data",
    "SparseAutoencoder",
    "create_sae_for_gpt2",
    "SAETrainer",
    "train_sae",
    "FeatureAnalyzer",
]
