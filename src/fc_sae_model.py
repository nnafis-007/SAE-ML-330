"""
Feature Choice Sparse Autoencoder (FC-SAE) Model

This module implements the Feature Choice SAE from the paper
"Feature Choice SAEs" (Ayonrinde, 2024).

Key Innovation:
Traditional SAEs impose sparsity per token (limit active features per token).
Feature Choice SAEs impose sparsity per feature (limit tokens each feature can
be active for). This ensures ALL features are utilized, preventing dead features
and improving reconstruction quality.

Mathematical formulation:
    Given batch of activations X ∈ ℝ^{N×d}:
    1. Pre-activation: Z = XW_enc^T + b_enc  (shape: N×F)
    2. For each feature i, select top-m tokens: S_i,j = 1 if j in top-m(Z[:,i])
    3. Apply selection: F = ReLU(Z) ⊙ S  (element-wise mask)
    4. Decode: X̂ = FW_dec + b_dec

    Constraint: ∑_j S_{i,j} = m for all features i
    Total sparsity budget: M = m × (number of features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class FeatureChoiceSAE(nn.Module):
    """
    Feature Choice Sparse Autoencoder.

    Instead of limiting features per token, this SAE limits tokens per feature.
    Each feature is forced to activate for exactly m tokens in each batch,
    ensuring uniform feature utilization.

    Key advantages:
    - No dead features (all features are forced to activate)
    - Adaptive computation per token (some tokens use more features)
    - Better reconstruction with same total sparsity budget
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        m_tokens_per_feature: int = 8,
        use_pre_bias: bool = True,
        normalize_decoder: bool = True,
        aux_loss_coeff: float = 1e-3,
    ):
        """
        Initialize Feature Choice SAE.

        Args:
            d_model: Input/output dimension (e.g., 768 for GPT-2)
            d_hidden: Hidden layer dimension (number of features)
            m_tokens_per_feature: Number of tokens each feature can activate for
                                  in a batch. Higher = denser representation.
                                  Typically set based on batch size (e.g., batch_size // 8)
            use_pre_bias: Use pre-encoding bias (centers input)
            normalize_decoder: Normalize decoder weights to unit norm
            aux_loss_coeff: Coefficient for auxiliary reconstruction loss on
                           non-selected features (helps feature learning)
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.m = m_tokens_per_feature
        self.use_pre_bias = use_pre_bias
        self.normalize_decoder = normalize_decoder
        self.aux_loss_coeff = aux_loss_coeff

        # Pre-encoding bias (centers input)
        if use_pre_bias:
            self.b_pre = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('b_pre', None)

        # Encoder weights
        self.W_enc = nn.Parameter(
            torch.randn(d_hidden, d_model) / math.sqrt(d_model)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        # Decoder weights
        self.W_dec = nn.Parameter(
            torch.randn(d_model, d_hidden) / math.sqrt(d_hidden)
        )
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize decoder to unit norm
        if normalize_decoder:
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def compute_selection_mask(
        self,
        pre_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the feature selection mask S.

        For each feature i, select the top-m tokens with highest pre-activation.
        This ensures each feature activates for exactly m tokens.

        Args:
            pre_activations: Pre-ReLU activations, shape (batch_size, d_hidden)

        Returns:
            Binary mask S of shape (batch_size, d_hidden)
            where S[j,i] = 1 if feature i is selected for token j
        """
        batch_size = pre_activations.shape[0]
        device = pre_activations.device

        # Clamp m to batch size (can't select more tokens than exist)
        m = min(self.m, batch_size)

        # For each feature (column), find top-m tokens
        # pre_activations: (batch_size, d_hidden)
        # We need to work column-wise (per feature)

        # Get indices of top-m pre-activations for each feature
        _, top_indices = torch.topk(pre_activations, k=m, dim=0)
        # top_indices: (m, d_hidden) - indices of top m tokens for each feature

        # Create selection mask
        mask = torch.zeros_like(pre_activations)

        # Scatter 1s at the selected positions
        # For each feature (column), scatter 1s at the top_indices rows
        feature_indices = torch.arange(self.d_hidden, device=device).unsqueeze(0)
        feature_indices = feature_indices.expand(m, -1)  # (m, d_hidden)

        mask.scatter_(0, top_indices, 1.0)

        return mask

    def encode(
        self,
        x: torch.Tensor,
        return_pre_activations: bool = False,
    ) -> torch.Tensor:
        """
        Encode input to sparse feature representation with feature-choice constraint.

        Args:
            x: Input activations, shape (batch_size, d_model)
            return_pre_activations: If True, also return pre-activations

        Returns:
            f: Sparse features after selection, shape (batch_size, d_hidden)
            pre_act: (optional) Pre-activations before selection
        """
        # Center input
        if self.use_pre_bias:
            x = x - self.b_pre

        # Linear transform to get pre-activations
        pre_act = F.linear(x, self.W_enc, self.b_enc)
        # Shape: (batch_size, d_hidden)

        # Compute selection mask (per-feature top-m selection)
        selection_mask = self.compute_selection_mask(pre_act)

        # Apply ReLU and mask
        f = F.relu(pre_act) * selection_mask

        if return_pre_activations:
            return f, pre_act
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to input space.

        Args:
            f: Sparse features, shape (batch_size, d_hidden)

        Returns:
            Reconstructed input, shape (batch_size, d_model)
        """
        return F.linear(f, self.W_dec, self.b_dec)

    def forward(
        self,
        x: torch.Tensor,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
        """
        Forward pass with feature choice constraint.

        Args:
            x: Input activations, shape (batch_size, d_model)
            return_loss_components: If True, return detailed loss breakdown

        Returns:
            x_reconstructed: Reconstructed activations
            loss: Total loss (reconstruction + auxiliary)
            loss_dict: Optional dictionary with loss components
        """
        # Encode with selection
        f, pre_act = self.encode(x, return_pre_activations=True)

        # Decode
        x_reconstructed = self.decode(f)

        # Reconstruction loss (primary objective)
        mse_loss = F.mse_loss(x_reconstructed, x)

        # Auxiliary loss: encourage features to have meaningful pre-activations
        # This helps features learn to activate on meaningful patterns
        # even when not selected
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.aux_loss_coeff > 0:
            # Penalize large negative pre-activations (features should want to fire)
            # This encourages features to compete for selection
            relu_pre = F.relu(pre_act)
            aux_loss = relu_pre.sum(dim=1).mean() * self.aux_loss_coeff

        # Total loss
        total_loss = mse_loss + aux_loss

        if return_loss_components:
            # Compute additional metrics
            selection_mask = self.compute_selection_mask(pre_act)
            active_per_token = selection_mask.sum(dim=1)

            loss_dict = {
                "loss": total_loss.item(),
                "mse_loss": mse_loss.item(),
                "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                "mean_activation": f.mean().item(),
                "frac_active": (f > 0).float().mean().item(),
                "max_activation": f.max().item(),
                "mean_features_per_token": active_per_token.float().mean().item(),
                "min_features_per_token": active_per_token.min().item(),
                "max_features_per_token": active_per_token.max().item(),
            }
            return x_reconstructed, total_loss, loss_dict

        return x_reconstructed, total_loss, None

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm."""
        if self.normalize_decoder:
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def get_feature_statistics(self, dataloader) -> Dict:
        """
        Compute feature utilization statistics.

        Unlike traditional SAEs, FC-SAE should have high and uniform
        feature utilization by design.
        """
        feature_counts = torch.zeros(self.d_hidden)
        feature_sums = torch.zeros(self.d_hidden)
        total_samples = 0

        device = next(self.parameters()).device

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            f = self.encode(batch)

            feature_counts += (f > 0).sum(dim=0).cpu()
            feature_sums += f.sum(dim=0).cpu()
            total_samples += batch.shape[0]

        feature_frequencies = feature_counts / total_samples
        feature_magnitudes = feature_sums / (feature_counts + 1e-8)
        dead_features = (feature_counts == 0).sum().item()

        return {
            "feature_frequencies": feature_frequencies,
            "feature_magnitudes": feature_magnitudes,
            "dead_features": dead_features,
            "dead_fraction": dead_features / self.d_hidden,
            "mean_frequency": feature_frequencies.mean().item(),
            "median_frequency": feature_frequencies.median().item(),
            "min_frequency": feature_frequencies.min().item(),
            "max_frequency": feature_frequencies.max().item(),
        }

    def get_feature_density(self, x: torch.Tensor) -> float:
        """Calculate fraction of active features."""
        f = self.encode(x)
        return (f > 0).float().mean().item()


class FeatureChoiceTopKSAE(nn.Module):
    """
    Hybrid Feature Choice SAE with Top-K per-token constraint.

    This variant combines both constraints:
    1. Feature choice: each feature can only activate for m tokens
    2. Top-K: each token uses at most k features

    This provides even more controlled sparsity.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        m_tokens_per_feature: int = 8,
        k_features_per_token: int = 32,
        use_pre_bias: bool = True,
        normalize_decoder: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.m = m_tokens_per_feature
        self.k = k_features_per_token
        self.use_pre_bias = use_pre_bias
        self.normalize_decoder = normalize_decoder

        if use_pre_bias:
            self.b_pre = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('b_pre', None)

        self.W_enc = nn.Parameter(
            torch.randn(d_hidden, d_model) / math.sqrt(d_model)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(
            torch.randn(d_model, d_hidden) / math.sqrt(d_hidden)
        )
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if normalize_decoder:
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with both feature-choice and top-k constraints."""
        if self.use_pre_bias:
            x = x - self.b_pre

        pre_act = F.linear(x, self.W_enc, self.b_enc)
        batch_size = pre_act.shape[0]

        # Feature choice mask (per feature, select top-m tokens)
        m = min(self.m, batch_size)
        _, feature_top_indices = torch.topk(pre_act, k=m, dim=0)
        feature_mask = torch.zeros_like(pre_act)
        feature_mask.scatter_(0, feature_top_indices, 1.0)

        # Top-K mask (per token, select top-k features)
        k = min(self.k, self.d_hidden)
        _, token_top_indices = torch.topk(pre_act, k=k, dim=1)
        token_mask = torch.zeros_like(pre_act)
        token_mask.scatter_(1, token_top_indices, 1.0)

        # Combine masks (intersection: both constraints must be satisfied)
        combined_mask = feature_mask * token_mask

        # Apply ReLU and mask
        f = F.relu(pre_act) * combined_mask

        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return F.linear(f, self.W_dec, self.b_dec)

    def forward(
        self,
        x: torch.Tensor,
        return_loss_components: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        f = self.encode(x)
        x_reconstructed = self.decode(f)

        mse_loss = F.mse_loss(x_reconstructed, x)

        if return_loss_components:
            loss_dict = {
                "loss": mse_loss.item(),
                "mse_loss": mse_loss.item(),
                "mean_activation": f.mean().item(),
                "frac_active": (f > 0).float().mean().item(),
                "max_activation": f.max().item(),
            }
            return x_reconstructed, mse_loss, loss_dict

        return x_reconstructed, mse_loss, None

    @torch.no_grad()
    def normalize_decoder_weights(self):
        if self.normalize_decoder:
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


def create_fc_sae_for_gpt2(
    model_name: str = "gpt2",
    expansion_factor: int = 16,
    m_tokens_per_feature: int = 8,
    aux_loss_coeff: float = 1e-3,
    **kwargs
) -> FeatureChoiceSAE:
    """
    Factory function to create a Feature Choice SAE for GPT-2.

    Args:
        model_name: GPT-2 variant
        expansion_factor: Hidden dimension multiplier
        m_tokens_per_feature: Tokens each feature can activate for
        aux_loss_coeff: Auxiliary loss coefficient
        **kwargs: Additional arguments for FeatureChoiceSAE

    Returns:
        Initialized FeatureChoiceSAE
    """
    model_dims = {
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600
    }

    d_model = model_dims.get(model_name, 768)
    d_hidden = d_model * expansion_factor

    print(f"Creating Feature Choice SAE for {model_name}")
    print(f"d_model: {d_model}")
    print(f"d_hidden: {d_hidden}")
    print(f"Expansion factor: {expansion_factor}x")
    print(f"Tokens per feature (m): {m_tokens_per_feature}")
    print(f"Aux loss coefficient: {aux_loss_coeff}")

    return FeatureChoiceSAE(
        d_model=d_model,
        d_hidden=d_hidden,
        m_tokens_per_feature=m_tokens_per_feature,
        aux_loss_coeff=aux_loss_coeff,
        **kwargs
    )


if __name__ == "__main__":
    """Test Feature Choice SAE."""
    print("Testing Feature Choice SAE")
    print("=" * 50)

    # Create model
    sae = create_fc_sae_for_gpt2(
        model_name="gpt2",
        expansion_factor=16,
        m_tokens_per_feature=8,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Test with random data
    batch_size = 256
    d_model = 768

    x = torch.randn(batch_size, d_model)

    # Forward pass
    x_reconstructed, loss, loss_dict = sae(x, return_loss_components=True)

    print("\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_reconstructed.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss components: {loss_dict}")

    # Verify feature choice constraint
    f = sae.encode(x)
    feature_activations = (f > 0).sum(dim=0)
    print(f"\nFeature choice verification:")
    print(f"Expected activations per feature: {sae.m}")
    print(f"Actual min: {feature_activations.min().item()}")
    print(f"Actual max: {feature_activations.max().item()}")
    print(f"Actual mean: {feature_activations.float().mean().item():.1f}")
