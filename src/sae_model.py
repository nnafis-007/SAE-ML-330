"""
Sparse Autoencoder (SAE) Model Architecture

This module implements the core SAE model based on the paper
"Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"

Key Concepts:
1. The SAE learns a sparse, overcomplete representation of activations
2. "Sparse" means most features are zero for any given input
3. "Overcomplete" means we have MORE features than input dimensions
4. This helps discover interpretable, monosemantic features

Architecture:
    Input (d_model) -> Encoder -> Latent (d_hidden) -> Decoder -> Output (d_model)
    
    where d_hidden >> d_model (typically 8-64x larger)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for neural network interpretability.
    
    The goal: Decompose dense neural network activations into sparse,
    interpretable features that represent individual concepts.
    
    Mathematical formulation:
        Given activation x ∈ ℝ^d:
        1. Encode: f = ReLU(W_enc @ (x - b_pre) + b_enc)
        2. Decode: x̂ = W_dec @ f + b_dec
        3. Loss: ||x - x̂||² + λ||f||₁
    
    Key innovations from the paper:
    - Pre-bias (b_pre) to center activations before encoding
    - L1 sparsity penalty to encourage sparse features
    - Tied or untied decoder weights (we use untied by default)
    - Normalization of decoder weights to prevent scale collapse
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        l1_coeff: float = 1e-3,
        use_tied_weights: bool = False,
        use_pre_bias: bool = True,
        normalize_decoder: bool = True
    ):
        """
        Initialize the Sparse Autoencoder.
        
        Args:
            d_model: Input/output dimension (e.g., 768 for GPT-2)
            d_hidden: Hidden layer dimension (typically 8-64x d_model)
                     Larger = more capacity to learn diverse features
                     Smaller = faster training, less memory
            l1_coeff: L1 sparsity coefficient (λ in the loss function)
                     Larger = sparser features (more zeros)
                     Smaller = denser features (better reconstruction)
                     Typical range: 1e-4 to 1e-2
            use_tied_weights: If True, decoder = encoder^T (saves memory)
                            Usually False for better performance
            use_pre_bias: Use pre-encoding bias (recommended by paper)
            normalize_decoder: Normalize decoder weights (prevents feature collapse)
        
        Why d_hidden >> d_model?
        - Neural networks use "superposition" - multiple features per neuron
        - Overcomplete representation separates these mixed features
        - Example: 768 -> 16384 gives ~21x expansion
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.use_tied_weights = use_tied_weights
        self.use_pre_bias = use_pre_bias
        self.normalize_decoder = normalize_decoder
        
        # Pre-encoding bias: centers the input activations
        # This is learned during training to find the "typical" activation
        if use_pre_bias:
            self.b_pre = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('b_pre', None)
        
        # Encoder: maps input to overcomplete hidden representation
        # We use Xavier initialization for stable training
        self.W_enc = nn.Parameter(torch.randn(d_hidden, d_model) / math.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        
        # Decoder: maps sparse hidden representation back to input space
        if use_tied_weights:
            # Tied weights: decoder is just encoder transpose
            # Pro: Fewer parameters, faster training
            # Con: Less flexible, potentially worse reconstruction
            self.W_dec = None
        else:
            # Untied weights: decoder is independent
            # Pro: More flexible, better reconstruction
            # Con: More parameters, slower training
            self.W_dec = nn.Parameter(torch.randn(d_model, d_hidden) / math.sqrt(d_hidden))
        
        # Post-decoding bias: allows decoder to output mean activation
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Initialize decoder weight norms to 1
        # This prevents features from growing arbitrarily large
        if normalize_decoder and not use_tied_weights:
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse hidden representation.
        
        Args:
            x: Input activations, shape (batch_size, d_model)
        
        Returns:
            f: Sparse hidden activations, shape (batch_size, d_hidden)
        
        Process:
        1. Center input by subtracting pre-bias (if used)
        2. Linear transform with encoder weights
        3. Add encoder bias
        4. Apply ReLU activation (enforces non-negativity = sparsity)
        
        Why ReLU?
        - Forces features to be non-negative (f ≥ 0)
        - Natural sparsity: many features will be exactly 0
        - Interpretable: feature strength is proportional to activation
        """
        if self.use_pre_bias:
            x = x - self.b_pre
        
        # Linear transformation
        f = F.linear(x, self.W_enc, self.b_enc)
        
        # ReLU activation for sparsity
        f = F.relu(f)
        
        return f
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse hidden representation back to input space.
        
        Args:
            f: Sparse hidden activations, shape (batch_size, d_hidden)
        
        Returns:
            x_reconstructed: Reconstructed input, shape (batch_size, d_model)
        
        Process:
        1. Linear transform with decoder weights (or encoder transpose)
        2. Add decoder bias
        
        Each column of W_dec is a "feature direction" in activation space.
        The reconstruction is a weighted sum of these directions.
        """
        if self.use_tied_weights:
            # Use encoder weights transposed
            x_reconstructed = F.linear(f, self.W_enc.t(), self.b_dec)
        else:
            # Use separate decoder weights
            x_reconstructed = F.linear(f, self.W_dec, self.b_dec)
        
        return x_reconstructed
    
    def forward(
        self, 
        x: torch.Tensor,
        return_loss_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input activations, shape (batch_size, d_model)
            return_loss_components: If True, return detailed loss breakdown
        
        Returns:
            x_reconstructed: Reconstructed activations
            loss: Total loss (MSE + L1 sparsity)
            loss_dict: Optional dictionary with loss components
        
        Loss function:
            L = ||x - x̂||² + λ||f||₁
        
        Where:
        - ||x - x̂||²: Reconstruction error (MSE)
          Measures how well we can reconstruct the original activation
        
        - λ||f||₁: L1 sparsity penalty
          Encourages most features to be zero
          L1 norm = sum of absolute values
        
        The balance:
        - High λ: Very sparse features, but poor reconstruction
        - Low λ: Good reconstruction, but dense (uninterpretable) features
        """
        # Encode to sparse representation
        f = self.encode(x)
        
        # Decode back to input space
        x_reconstructed = self.decode(f)
        
        # Calculate reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_reconstructed, x)
        
        # Calculate L1 sparsity loss
        # We take the mean over batch and sum over features
        l1_loss = f.abs().mean()
        
        # Total loss
        total_loss = mse_loss + self.l1_coeff * l1_loss
        
        if return_loss_components:
            loss_dict = {
                "loss": total_loss.item(),
                "mse_loss": mse_loss.item(),
                "l1_loss": l1_loss.item(),
                "l1_scaled": (self.l1_coeff * l1_loss).item(),
                "mean_activation": f.mean().item(),
                "frac_active": (f > 0).float().mean().item(),  # Fraction of non-zero features
                "max_activation": f.max().item()
            }
            return x_reconstructed, total_loss, loss_dict
        
        return x_reconstructed, total_loss, None
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        """
        Normalize decoder weight columns to unit norm.
        
        Why this matters:
        - Without normalization, features can grow large to reduce L1 penalty
        - Normalization keeps feature magnitudes comparable
        - Run this periodically during training (e.g., after each gradient step)
        
        This is a key technique from the paper for stable training.
        """
        if not self.use_tied_weights and self.normalize_decoder:
            # Normalize each column (feature direction) to unit length
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
    
    @torch.no_grad()
    def get_feature_statistics(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Compute statistics about learned features.
        
        Args:
            dataloader: DataLoader with activation data
        
        Returns:
            Dictionary with feature statistics:
            - feature_frequencies: How often each feature activates
            - feature_magnitudes: Average magnitude when active
            - dead_features: Features that never activate
        
        Why this is useful:
        - Identify "dead" features that aren't learning anything
        - Understand feature usage distribution
        - Detect if features are too sparse or too dense
        """
        feature_counts = torch.zeros(self.d_hidden)
        feature_sums = torch.zeros(self.d_hidden)
        total_samples = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(next(self.parameters()).device)
            f = self.encode(batch)
            
            # Count how many times each feature is active
            feature_counts += (f > 0).sum(dim=0).cpu()
            # Sum feature activations
            feature_sums += f.sum(dim=0).cpu()
            total_samples += batch.shape[0]
        
        # Calculate statistics
        feature_frequencies = feature_counts / total_samples
        feature_magnitudes = feature_sums / (feature_counts + 1e-8)
        dead_features = (feature_counts == 0).sum().item()
        
        stats = {
            "feature_frequencies": feature_frequencies,
            "feature_magnitudes": feature_magnitudes,
            "dead_features": dead_features,
            "dead_fraction": dead_features / self.d_hidden,
            "mean_frequency": feature_frequencies.mean().item(),
            "median_frequency": feature_frequencies.median().item(),
        }
        
        return stats
    
    def get_feature_density(self, x: torch.Tensor) -> float:
        """
        Calculate L0 "density" (fraction of active features).
        
        Args:
            x: Input activations
        
        Returns:
            Fraction of features that are non-zero (active)
        
        Target: typically 1-5% for good interpretability
        """
        f = self.encode(x)
        return (f > 0).float().mean().item()


def create_sae_for_gpt2(
    model_name: str = "gpt2",
    expansion_factor: int = 16,
    l1_coeff: float = 3e-4,
    **kwargs
) -> SparseAutoencoder:
    """
    Factory function to create an SAE for a specific GPT-2 model.
    
    Args:
        model_name: GPT-2 variant ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        expansion_factor: Hidden dimension multiplier (d_hidden = expansion_factor * d_model)
        l1_coeff: L1 sparsity coefficient
        **kwargs: Additional arguments for SparseAutoencoder
    
    Returns:
        Initialized SparseAutoencoder
    
    Model dimensions:
    - gpt2: 768
    - gpt2-medium: 1024
    - gpt2-large: 1280
    - gpt2-xl: 1600
    """
    model_dims = {
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600
    }
    
    d_model = model_dims.get(model_name, 768)
    d_hidden = d_model * expansion_factor
    
    print(f"Creating SAE for {model_name}")
    print(f"d_model: {d_model}")
    print(f"d_hidden: {d_hidden}")
    print(f"Expansion factor: {expansion_factor}x")
    print(f"L1 coefficient: {l1_coeff}")
    
    return SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        l1_coeff=l1_coeff,
        **kwargs
    )


if __name__ == "__main__":
    """
    Example usage and testing
    """
    # Create SAE for GPT-2
    sae = create_sae_for_gpt2(
        model_name="gpt2",
        expansion_factor=16,
        l1_coeff=3e-4
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in sae.parameters()):,}")
    
    # Test with random data
    batch_size = 32
    d_model = 768
    
    x = torch.randn(batch_size, d_model)
    
    # Forward pass
    x_reconstructed, loss, loss_dict = sae(x, return_loss_components=True)
    
    print("\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_reconstructed.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss components: {loss_dict}")
    
    # Test encoding
    f = sae.encode(x)
    print(f"\nEncoded shape: {f.shape}")
    print(f"Feature density: {sae.get_feature_density(x):.2%}")
