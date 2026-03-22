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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        normalize_decoder: bool = True,
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
            self.register_parameter("b_pre", None)

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
            self.W_dec = nn.Parameter(
                torch.randn(d_model, d_hidden) / math.sqrt(d_hidden)
            )

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
        self, x: torch.Tensor, return_loss_components: bool = False
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
        # We take the mean over batch and sum over features (per-sample L1 norm).
        # This keeps the meaning of l1_coeff stable as d_hidden changes.
        l1_loss = f.abs().sum(dim=1).mean()

        # Total loss
        total_loss = mse_loss + self.l1_coeff * l1_loss

        if return_loss_components:
            loss_dict = {
                "loss": total_loss.item(),
                "mse_loss": mse_loss.item(),
                "l1_loss": l1_loss.item(),
                "l1_scaled": (self.l1_coeff * l1_loss).item(),
                "mean_activation": f.mean().item(),
                "frac_active": (f > 0)
                .float()
                .mean()
                .item(),  # Fraction of non-zero features
                "max_activation": f.max().item(),
            }
            return x_reconstructed, total_loss, loss_dict

        return x_reconstructed, total_loss, None

    @torch.no_grad()
    def pruned_copy(self, keep_indices: torch.Tensor) -> "SparseAutoencoder":
        """Return a new SAE containing only the specified hidden features.

        This is useful when the model is heavily overcomplete and many features are
        "dead" (never/rarely activate). Pruning keeps behavior for the retained
        features while reducing parameter count.

        Args:
            keep_indices: 1D tensor of feature indices to keep.

        Returns:
            A new SparseAutoencoder with d_hidden = len(keep_indices).
        """
        keep_indices = keep_indices.to(self.W_enc.device).long().flatten()
        if keep_indices.numel() == 0:
            raise ValueError("keep_indices must be non-empty")

        if self.use_tied_weights:
            raise NotImplementedError("pruned_copy is not implemented for tied weights")

        pruned = SparseAutoencoder(
            d_model=self.d_model,
            d_hidden=int(keep_indices.numel()),
            l1_coeff=self.l1_coeff,
            use_tied_weights=False,
            use_pre_bias=self.use_pre_bias,
            normalize_decoder=self.normalize_decoder,
        ).to(self.W_enc.device)

        # Copy biases
        pruned.b_dec.data.copy_(self.b_dec.data)
        if self.use_pre_bias and self.b_pre is not None and pruned.b_pre is not None:
            pruned.b_pre.data.copy_(self.b_pre.data)

        # Copy selected features
        pruned.W_enc.data.copy_(self.W_enc.data[keep_indices])
        pruned.b_enc.data.copy_(self.b_enc.data[keep_indices])
        pruned.W_dec.data.copy_(self.W_dec.data[:, keep_indices])

        if pruned.normalize_decoder:
            pruned.W_dec.data = F.normalize(pruned.W_dec.data, dim=0)

        return pruned

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
    **kwargs,
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
    model_dims = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}

    d_model = model_dims.get(model_name, 768)
    d_hidden = d_model * expansion_factor

    print(f"Creating SAE for {model_name}")
    print(f"d_model: {d_model}")
    print(f"d_hidden: {d_hidden}")
    print(f"Expansion factor: {expansion_factor}x")
    print(f"L1 coefficient: {l1_coeff}")

    return SparseAutoencoder(
        d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff, **kwargs
    )


# ===========================================================================
# Top-K Sparse Autoencoder
# ===========================================================================


class TopKSparseAutoencoder(nn.Module):
    """
    Top-K Sparse Autoencoder variant.

    Replaces the ReLU + L1-penalty sparsity mechanism with a hard Top-K
    selection.  After computing linear pre-activations, only the K largest
    values are kept; all others are set to zero.  This gives exact, per-sample
    control over feature density without requiring a manually-tuned sparsity
    coefficient.

    Mathematical formulation:
        Given activation x ∈ ℝ^d:
        1. (Optional) center:   x̃ = x − b_pre
        2. Pre-activations:     z  = x̃ W_enc^T + b_enc
        3. Top-K selection:     f  = TopK(z, k)  — keep k largest, zero the rest
        4. Non-negativity:      f  = ReLU(f)     — zeros out any negative top-k values
        5. Decode:              x̂  = f W_dec^T + b_dec
        6. Loss:                L  = ‖x − x̂‖²   (MSE only — no L1 needed)

    Gradients flow naturally through the k selected units via standard
    autograd (torch.topk is differentiable w.r.t. its input values);
    no straight-through estimator is required.

    The l1_coeff attribute is kept at 0.0 purely for SAETrainer compatibility
    (the trainer reads model.l1_coeff when saving checkpoints).

    Reference:
        Gao et al., "Scaling and evaluating sparse autoencoders", OpenAI 2024.
        https://arxiv.org/abs/2406.04093
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        k: int,
        use_tied_weights: bool = False,
        use_pre_bias: bool = True,
        normalize_decoder: bool = True,
    ):
        """
        Initialize the Top-K Sparse Autoencoder.

        Args:
            d_model:           Input/output dimension (e.g. 768 for GPT-2).
            d_hidden:          Hidden/dictionary dimension (typically 8–64× d_model).
            k:                 Number of active features per input token.
                               Directly controls sparsity — smaller k = sparser.
                               Use suggest_k() for a sensible starting point.
            use_tied_weights:  If True, decoder = W_enc^T (saves parameters).
                               Usually False for Top-K; untied weights give better
                               reconstruction at the cost of more parameters.
            use_pre_bias:      Subtract a learned bias before encoding (recommended).
            normalize_decoder: Normalize each decoder column to unit norm after
                               every gradient step (prevents scale / shrinkage collapse).

        Why untied weights as default for Top-K?
        - The L1 SAE paper used tied weights in early experiments but found that
          untied weights improve reconstruction quality.
        - With Top-K, the structural sparsity guarantee is independent of weight
          norms, so there is even less reason to tie them.
        """
        super().__init__()

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > d_hidden:
            raise ValueError(f"k ({k}) cannot exceed d_hidden ({d_hidden})")

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.use_tied_weights = use_tied_weights
        self.use_pre_bias = use_pre_bias
        self.normalize_decoder = normalize_decoder

        # Kept at 0.0 for SAETrainer / checkpoint compatibility.
        # Top-K SAEs do not use an L1 penalty — sparsity is structural.
        self.l1_coeff: float = 0.0

        # Pre-encoding bias: centers activations before projection
        if use_pre_bias:
            self.b_pre = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("b_pre", None)

        # Encoder: maps d_model → d_hidden
        self.W_enc = nn.Parameter(torch.randn(d_hidden, d_model) / math.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))

        # Decoder: maps d_hidden → d_model
        if use_tied_weights:
            self.W_dec = None  # will use W_enc.t() at runtime
        else:
            self.W_dec = nn.Parameter(
                torch.randn(d_model, d_hidden) / math.sqrt(d_hidden)
            )

        # Post-decoding bias: allows decoder to output mean activation
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize decoder columns to unit norm
        if normalize_decoder and not use_tied_weights:
            assert self.W_dec is not None
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to a sparse Top-K hidden representation.

        Args:
            x: Input activations, shape (batch_size, d_model).

        Returns:
            f: Sparse hidden activations, shape (batch_size, d_hidden).
               At most k values per sample are non-zero (may be fewer if
               some of the k largest pre-activations were negative and get
               zeroed by the trailing ReLU).

        Steps:
            1. Optionally subtract b_pre to center the input.
            2. Linear projection:  z = x W_enc^T + b_enc
            3. Top-K selection:    keep the k largest values, zero the rest.
            4. ReLU:               enforce non-negativity.

        Gradient flow:
            torch.topk is differentiable w.r.t. its value inputs.
            Gradients propagate directly through the k selected positions
            via standard autograd — no straight-through estimator needed.
        """
        if self.use_pre_bias:
            x = x - self.b_pre

        # Linear pre-activations, shape (batch, d_hidden)
        z = F.linear(x, self.W_enc, self.b_enc)

        # Top-K selection ------------------------------------------------
        # Clamp k defensively in case d_hidden is unexpectedly smaller at
        # inference time (shouldn't happen after __init__ validation).
        k = min(self.k, z.shape[-1])

        # topk_values keeps its gradient connection to z.
        topk_values, topk_indices = z.topk(k=k, dim=-1)  # (batch, k)

        # Scatter top-k values into a zero tensor of full width.
        # All non-top-k positions remain exactly 0.
        f = torch.zeros_like(z)
        f.scatter_(dim=-1, index=topk_indices, src=topk_values)

        # Enforce non-negativity — consistent with standard SAE semantics
        # and ensures feature magnitudes are interpretable.
        f = F.relu(f)

        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse hidden representation back to input space.

        Args:
            f: Sparse hidden activations, shape (batch_size, d_hidden).

        Returns:
            x_reconstructed: Reconstructed input, shape (batch_size, d_model).
        """
        if self.use_tied_weights:
            return F.linear(f, self.W_enc.t(), self.b_dec)
        assert self.W_dec is not None
        return F.linear(f, self.W_dec, self.b_dec)

    def forward(
        self,
        x: torch.Tensor,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Full forward pass: encode → decode → loss.

        Args:
            x:                      Input activations, shape (batch_size, d_model).
            return_loss_components: If True, return a detailed per-component dict.

        Returns:
            x_reconstructed: Reconstructed activations, shape (batch_size, d_model).
            loss:            Scalar MSE reconstruction loss.
            loss_dict:       Optional metrics dict (same keys as SparseAutoencoder
                             for drop-in compatibility with SAETrainer / analysis code).

        Loss:
            L = ‖x − x̂‖²    (MSE only — Top-K enforces sparsity structurally)

        Compatibility note:
            loss_dict always contains "l1_loss" and "l1_scaled" keys (both 0.0)
            so that any downstream code written for SparseAutoencoder continues
            to work without modification.
        """
        f = self.encode(x)
        x_reconstructed = self.decode(f)

        # Pure reconstruction loss — sparsity is guaranteed by Top-K selection,
        # not by penalizing it in the objective.
        mse_loss = F.mse_loss(x_reconstructed, x)
        total_loss = mse_loss

        if return_loss_components:
            active_per_sample = (f > 0).float().sum(dim=-1)  # (batch,)
            loss_dict = {
                "loss": total_loss.item(),
                "mse_loss": mse_loss.item(),
                # Always 0.0 — kept for SAETrainer / logging compatibility
                "l1_loss": 0.0,
                "l1_scaled": 0.0,
                # Standard SAE diagnostics
                "mean_activation": f.mean().item(),
                "frac_active": (f > 0).float().mean().item(),
                "max_activation": f.max().item(),
                # Top-K specific: how many features actually fired (≤ k due to ReLU)
                "mean_active_per_sample": active_per_sample.mean().item(),
            }
            return x_reconstructed, total_loss, loss_dict

        return x_reconstructed, total_loss, None

    # ------------------------------------------------------------------
    # Weight maintenance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """
        Normalize decoder weight columns to unit norm.

        Should be called after every optimizer step during training to prevent
        the model from trivially reducing loss by scaling up decoder columns.
        This is especially important for Top-K because the absence of an L1
        penalty means there is no other force keeping norms bounded.

        No-op when use_tied_weights=True or normalize_decoder=False.
        """
        if not self.use_tied_weights and self.normalize_decoder:
            assert self.W_dec is not None
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def pruned_copy(self, keep_indices: torch.Tensor) -> "TopKSparseAutoencoder":
        """Return a new TopKSparseAutoencoder containing only the specified features.

        Dead features (those that rarely appear in the top-k) can be identified
        via get_feature_statistics() and removed with this method to reduce
        parameter count without changing the behavior for retained features.

        Args:
            keep_indices: 1-D tensor of feature indices to retain.

        Returns:
            A new TopKSparseAutoencoder with d_hidden = len(keep_indices).
            k is clamped to the new d_hidden if necessary.
        """
        keep_indices = keep_indices.to(self.W_enc.device).long().flatten()
        if keep_indices.numel() == 0:
            raise ValueError("keep_indices must be non-empty")
        if self.use_tied_weights:
            raise NotImplementedError("pruned_copy is not implemented for tied weights")

        new_d_hidden = int(keep_indices.numel())
        new_k = min(self.k, new_d_hidden)

        pruned = TopKSparseAutoencoder(
            d_model=self.d_model,
            d_hidden=new_d_hidden,
            k=new_k,
            use_tied_weights=False,
            use_pre_bias=self.use_pre_bias,
            normalize_decoder=self.normalize_decoder,
        ).to(self.W_enc.device)

        pruned.b_dec.data.copy_(self.b_dec.data)
        if self.use_pre_bias and self.b_pre is not None and pruned.b_pre is not None:
            pruned.b_pre.data.copy_(self.b_pre.data)

        assert self.W_dec is not None, (
            "W_dec is None; pruned_copy already guards against tied weights"
        )
        assert pruned.W_dec is not None
        pruned.W_enc.data.copy_(self.W_enc.data[keep_indices])
        pruned.b_enc.data.copy_(self.b_enc.data[keep_indices])
        pruned.W_dec.data.copy_(self.W_dec.data[:, keep_indices])

        if pruned.normalize_decoder:
            pruned.W_dec.data = F.normalize(pruned.W_dec.data, dim=0)

        return pruned

    # ------------------------------------------------------------------
    # Analysis helpers (mirror the SparseAutoencoder public API)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_feature_statistics(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Compute per-feature activation statistics over an entire dataloader.

        Mirrors SparseAutoencoder.get_feature_statistics() so that analysis
        code works with both model types without modification.

        Args:
            dataloader: DataLoader yielding activation batches.

        Returns:
            dict with keys:
                feature_frequencies  — fraction of samples each feature fired in.
                feature_magnitudes   — mean activation magnitude when active.
                dead_features        — count of features that never activated.
                dead_fraction        — dead_features / d_hidden.
                mean_frequency       — mean(feature_frequencies).
                median_frequency     — median(feature_frequencies).
                mean_k_active        — empirical mean of active features per sample
                                       (ideally ≈ k but may be slightly less due to ReLU).
        """
        feature_counts = torch.zeros(self.d_hidden)
        feature_sums = torch.zeros(self.d_hidden)
        total_samples = 0
        total_active = 0.0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(next(self.parameters()).device)
            f = self.encode(batch)

            feature_counts += (f > 0).sum(dim=0).cpu()
            feature_sums += f.sum(dim=0).cpu()
            total_active += (f > 0).float().sum().item()
            total_samples += batch.shape[0]

        feature_frequencies = feature_counts / max(total_samples, 1)
        feature_magnitudes = feature_sums / (feature_counts + 1e-8)
        dead_features = int((feature_counts == 0).sum().item())

        return {
            "feature_frequencies": feature_frequencies,
            "feature_magnitudes": feature_magnitudes,
            "dead_features": dead_features,
            "dead_fraction": dead_features / self.d_hidden,
            "mean_frequency": feature_frequencies.mean().item(),
            "median_frequency": feature_frequencies.median().item(),
            "mean_k_active": total_active / max(total_samples, 1),
        }

    def get_feature_density(self, x: torch.Tensor) -> float:
        """
        Fraction of (sample, feature) pairs that are non-zero.

        For a well-trained Top-K SAE this is approximately k / d_hidden,
        but may be slightly lower if some top-k pre-activations were negative
        and got zeroed by the trailing ReLU.

        Args:
            x: Input activations, shape (batch_size, d_model).

        Returns:
            Float in [0, 1].
        """
        f = self.encode(x)
        return (f > 0).float().mean().item()


# ===========================================================================
# Top-K hyperparameter utilities
# ===========================================================================


def suggest_k(
    d_model: int,
    expansion_factor: int = 1,
    target_density: float = 0.04,
) -> int:
    """
    Suggest a value for k (number of active features per input token).

    Rules of thumb from the literature:
    - k is usually chosen relative to *d_model*, not d_hidden, so that the
      fraction of input variance explained per token stays roughly constant
      as the dictionary size grows.
    - A common target is ~1–5 % of d_model active per sample.
    - For GPT-2 (d_model = 768) this gives k ≈ 8–38; k = 32 is a popular choice.
    - For larger dictionaries (higher expansion_factor) a modest upward
      adjustment helps because features are more specialised and more of them
      may legitimately fire on a single token.

    Formula:
        base_k = round(d_model × target_density)
        scale  = 1 + 0.1 × log(expansion_factor)
        k      = max(1, round(base_k × scale))

    Args:
        d_model:          Input/output dimension.
        expansion_factor: d_hidden / d_model ratio (used to scale k slightly).
        target_density:   Desired fraction of d_model features active per sample
                          (default 4 %).

    Returns:
        Suggested integer k.

    Examples:
        >>> suggest_k(512,  expansion_factor=1)   # ~20
        >>> suggest_k(768,  expansion_factor=16)  # ~35
        >>> suggest_k(1024, expansion_factor=32)  # ~52
    """
    base_k = max(1, round(d_model * target_density))
    scale = 1.0 + 0.1 * math.log(max(1, expansion_factor))
    return max(1, round(base_k * scale))


def create_topk_sae_for_gpt2(
    model_name: str = "gpt2",
    expansion_factor: int = 16,
    k: Optional[int] = None,
    use_tied_weights: bool = False,
    **kwargs,
) -> TopKSparseAutoencoder:
    """
    Factory function to create a Top-K SAE for a specific GPT-2 variant.

    Args:
        model_name:       GPT-2 variant ("gpt2", "gpt2-medium", "gpt2-large",
                          "gpt2-xl").
        expansion_factor: Hidden-dimension multiplier
                          (d_hidden = expansion_factor × d_model).
        k:                Number of active features per token.
                          If None, suggest_k() is called automatically using
                          the model's d_model and expansion_factor.
        use_tied_weights: Use W_enc^T as decoder (default False for Top-K).
        **kwargs:         Additional keyword arguments forwarded to
                          TopKSparseAutoencoder (e.g. use_pre_bias,
                          normalize_decoder).

    Returns:
        Initialized TopKSparseAutoencoder.

    Model dimensions:
        gpt2:         768
        gpt2-medium: 1024
        gpt2-large:  1280
        gpt2-xl:     1600
    """
    model_dims = {
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600,
    }

    d_model = model_dims.get(model_name, 768)
    d_hidden = d_model * expansion_factor

    if k is None:
        k = suggest_k(d_model, expansion_factor=expansion_factor)

    print(f"Creating Top-K SAE for {model_name}")
    print(f"  d_model:          {d_model}")
    print(f"  d_hidden:         {d_hidden}  (expansion ×{expansion_factor})")
    print(f"  k:                {k}  (~{k / d_model:.1%} of d_model active per token)")
    print(f"  use_tied_weights: {use_tied_weights}")

    return TopKSparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        k=k,
        use_tied_weights=use_tied_weights,
        **kwargs,
    )


# ===========================================================================
# Entry-point tests
# ===========================================================================

if __name__ == "__main__":
    """
    Example usage and testing — exercises both SAE variants.
    """
    # ---- Standard SAE -------------------------------------------------------
    print("=" * 60)
    print("Standard Sparse Autoencoder")
    print("=" * 60)

    # Create SAE for GPT-2
    sae = create_sae_for_gpt2(model_name="gpt2", expansion_factor=16, l1_coeff=3e-4)

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

    # ---- Top-K SAE ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Top-K Sparse Autoencoder")
    print("=" * 60)

    topk_sae = create_topk_sae_for_gpt2(
        model_name="gpt2",
        expansion_factor=16,
        k=32,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in topk_sae.parameters()):,}")

    # Suggest k for reference
    k_suggested = suggest_k(d_model=768, expansion_factor=16)
    print(f"suggest_k(768, expansion_factor=16) → {k_suggested}")

    # Forward pass
    x_rec_topk, loss_topk, dict_topk = topk_sae(x, return_loss_components=True)

    print("\nTest forward pass:")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {x_rec_topk.shape}")
    print(f"Loss (MSE only): {loss_topk.item():.6f}")
    print(f"Loss components: {dict_topk}")

    # Verify exact sparsity
    f_topk = topk_sae.encode(x)
    active_counts = (f_topk > 0).sum(dim=-1).float()
    print(f"\nEncoded shape:        {f_topk.shape}")
    print(f"Mean active per sample: {active_counts.mean().item():.1f}  (target k=32)")
    print(f"Feature density:      {topk_sae.get_feature_density(x):.4%}")
