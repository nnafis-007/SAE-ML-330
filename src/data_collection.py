"""
Data Collection Module for Sparse Autoencoder Training

This module handles:
1. Loading GPT-2 from HuggingFace
2. Extracting activations from a specific hidden layer
3. Processing text datasets to collect activation data

Key Concepts:
- We need to collect the intermediate activations (hidden states) from GPT-2
- These activations represent what the model "thinks" at that layer
- We'll use these activations to train our Sparse Autoencoder
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from typing import Optional, List, Tuple, Dict
from tqdm.auto import tqdm
import numpy as np


class GPT2ActivationCollector:
    """
    Collects activations from a specific layer of GPT-2.
    
    The key insight: GPT-2 has multiple transformer layers, and each layer
    produces hidden states (activations) for each token. We want to capture
    these activations to understand what features the model uses internally.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt2",
        layer_index: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the activation collector.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "gpt2-medium")
            layer_index: Which transformer layer to extract from (0-11 for gpt2)
                        Middle layers (6-9) often contain the most interpretable features
            device: Computing device (cuda/cpu)
        
        Why layer_index matters:
        - Early layers (0-3): Basic syntax and word features
        - Middle layers (4-8): Complex semantic features, relationships
        - Late layers (9-11): Task-specific features, predictions
        """
        self.device = device
        self.layer_index = layer_index
        
        print(f"Loading GPT-2 model: {model_name}")
        print(f"Target layer: {layer_index}")
        print(f"Device: {device}")
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 doesn't have a padding token by default, so we set it to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            output_hidden_states=True  # Critical: enables access to intermediate layers
        ).to(device)
        
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        # Get model configuration
        self.hidden_size = self.model.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")
        
    def collect_activations(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 128,
        max_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Collect activations from GPT-2 for a list of texts.
        
        Args:
            texts: List of input texts to process
            batch_size: Number of texts to process at once (depends on GPU memory)
            max_length: Maximum token length per text
            max_samples: Maximum number of activation vectors to collect (None = all)
        
        Returns:
            Tensor of shape (num_samples, hidden_size) containing activations
            
        How it works:
        1. Tokenize text into token IDs
        2. Feed through GPT-2
        3. Extract hidden states from the specified layer
        4. Flatten and concatenate all token activations
        
        Why we collect ALL token activations:
        - Each token position has its own activation vector
        - More data = better SAE training
        - We want diverse examples of what the layer computes
        """
        all_activations = []
        num_collected = 0
        
        with torch.no_grad():  # Disable gradient computation (we're not training GPT-2)
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                # Handle max_samples limit
                if max_samples and num_collected >= max_samples:
                    break
                    
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize the batch
                # padding=True: Makes all sequences the same length
                # truncation=True: Cuts sequences that are too long
                # return_tensors="pt": Returns PyTorch tensors
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Forward pass through GPT-2
                outputs = self.model(**inputs)
                
                # Extract hidden states from the target layer
                # hidden_states is a tuple: (embedding_layer, layer_0, layer_1, ..., layer_n)
                # We add 1 because index 0 is the embedding layer
                hidden_states = outputs.hidden_states[self.layer_index + 1]
                # Shape: (batch_size, sequence_length, hidden_size)
                
                # Get the attention mask to filter out padding tokens
                attention_mask = inputs["attention_mask"]
                # Shape: (batch_size, sequence_length)
                
                # Extract only real (non-padded) token activations
                for j in range(hidden_states.shape[0]):
                    # Get positions where mask is 1 (real tokens)
                    mask = attention_mask[j].bool()
                    # Extract activations for real tokens only
                    token_activations = hidden_states[j][mask]
                    # Shape: (num_real_tokens, hidden_size)
                    
                    all_activations.append(token_activations.cpu())
                    num_collected += token_activations.shape[0]
                    
                    if max_samples and num_collected >= max_samples:
                        break
        
        # Concatenate all activations into a single tensor
        activations = torch.cat(all_activations, dim=0)
        
        # Apply max_samples limit if needed
        if max_samples:
            activations = activations[:max_samples]
        
        print(f"\nCollected {activations.shape[0]} activation vectors")
        print(f"Shape: {activations.shape}")
        print(f"Mean activation: {activations.mean().item():.4f}")
        print(f"Std activation: {activations.std().item():.4f}")
        
        return activations
    
    def collect_from_dataset(
        self,
        dataset_name: str = "openwebtext",
        split: str = "train",
        num_texts: int = 10000,
        **kwargs
    ) -> torch.Tensor:
        """
        Convenience method to collect activations from a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
                         "openwebtext": Web text (similar to GPT-2 training data)
                         "openwebtext": Wikipedia articles
                         "bookcorpus": Book text
            split: Dataset split to use
            num_texts: Number of texts to sample from dataset
            **kwargs: Additional arguments passed to collect_activations
        
        Returns:
            Tensor of collected activations
            
        Why dataset choice matters:
        - Use data similar to what GPT-2 was trained on for best results
        - OpenWebText is a good default (mimics GPT-2's training data)
        - More diverse data = more diverse features learned
        """
        print(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            # Sample texts from the dataset
            texts = []
            # add tqdm for progress indication
            for i, example in enumerate(tqdm(dataset, total=num_texts, desc="Loading texts")):
                if i >= num_texts:
                    break
                # Different datasets have different text field names
                text = example.get("text", example.get("content", ""))
                if text.strip():  # Only add non-empty texts
                    texts.append(text)
            
            print(f"Loaded {len(texts)} texts from {dataset_name}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to sample texts...")
            # Fallback: use some sample texts
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing enables computers to understand human language.",
            ] * (num_texts // 3)
        
        # Collect activations from these texts
        return self.collect_activations(texts, **kwargs)


def prepare_training_data(
    activations: torch.Tensor,
    train_ratio: float = 0.9,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Prepare activation data for SAE training.
    
    Args:
        activations: Raw activation tensor
        train_ratio: Fraction of data to use for training (rest for validation)
        normalize: Whether to normalize activations (recommended)
    
    Returns:
        train_data: Training activations
        val_data: Validation activations
        stats: Dictionary with normalization statistics
        
    Why normalization matters:
    - Activations can have different scales across dimensions
    - Normalization helps the autoencoder learn balanced features
    - We typically center (subtract mean) and sometimes scale
    """
    print("\nPreparing training data...")
    
    # Calculate statistics on the training portion only
    # (to avoid data leakage from validation set)
    split_idx = int(len(activations) * train_ratio)
    train_data = activations[:split_idx]
    val_data = activations[split_idx:]
    
    stats = {}
    
    if normalize:
        # Calculate mean and std on training data
        mean = train_data.mean(dim=0, keepdim=True)
        std = train_data.std(dim=0, keepdim=True)
        
        # Store statistics
        stats["mean"] = mean
        stats["std"] = std
        
        # Normalize both train and validation using training statistics
        # This is important: validation data should be normalized using
        # the same parameters as training data
        train_data = (train_data - mean) / (std + 1e-8)
        val_data = (val_data - mean) / (std + 1e-8)
        
        print(f"Normalized data (mean=0, std=1)")
    
    print(f"Training samples: {train_data.shape[0]}")
    print(f"Validation samples: {val_data.shape[0]}")
    
    return train_data, val_data, stats


if __name__ == "__main__":
    """
    Example usage: Collect activations from GPT-2 layer 8
    """
    # Initialize collector
    collector = GPT2ActivationCollector(
        model_name="gpt2",
        layer_index=8  # Middle layer - often most interpretable
    )
    
    # Collect from a sample dataset
    activations = collector.collect_from_dataset(
        dataset_name="openwebtext",
        num_texts=1000,
        max_samples=50000,
        batch_size=16
    )
    
    # Prepare for training
    train_data, val_data, stats = prepare_training_data(activations)
    
    # Save the data
    torch.save({
        "train": train_data,
        "val": val_data,
        "stats": stats,
        "hidden_size": collector.hidden_size,
        "layer_index": collector.layer_index
    }, "gpt2_activations.pt")
    
    print("\nData saved to gpt2_activations.pt")
