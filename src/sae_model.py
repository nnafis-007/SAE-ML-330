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
        aux_k: Optional[int] = None,
        aux_loss_coeff: float = 1 / 32,
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
            aux_k:             Number of dead features activated in the auxiliary
                               reconstruction pass. Defaults to a scaled value:
                               - expansion ≤ 8x:  d_hidden // 2
                               - expansion ≤ 16x: d_hidden // 4
                               - expansion > 16x: max(512, d_hidden // 8)
                               Set to 0 to disable. For large expansions, smaller
                               aux_k gives more focused gradients to fewer features.
            aux_loss_coeff:    Weight applied to the auxiliary loss term.
                               The OpenAI paper uses 1/32 (~0.03125). Increase if
                               dead features persist; decrease if reconstruction
                               quality degrades. Default: 1/32.

        Why aux_k matters with large expansion factors:
            With d_hidden=18432 and k=32, a feature has only a 0.17% chance of
            landing in the top-k per sample. Random initialisation breaks symmetry
            immediately — features that win early get stronger, win more often, and
            dominate permanently. The remaining ~70% receive zero gradient and die.
            The aux loss gives every dead feature a residual-based gradient signal
            every batch, preventing this rich-get-richer collapse.

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

        # aux_k defaults to a value that scales with expansion factor.
        # Gao et al. 2024 used d_hidden // 2 for smaller expansions (4-8x),
        # but for large expansions (16x+), this gives too many dead features
        # a gradient signal simultaneously, diluting the learning.
        # Rule of thumb: use ~min(512, d_hidden // 4) to ~min(2048, d_hidden // 2)
        # depending on expansion factor.
        if aux_k is None:
            # Scale aux_k inversely with expansion factor
            expansion = d_hidden / d_model
            if expansion <= 8:
                self.aux_k = d_hidden // 2
            elif expansion <= 16:
                self.aux_k = d_hidden // 4
            else:  # expansion > 16
                # For very large expansions (24x), use even smaller aux_k
                # to give focused gradients to fewer dead features at a time
                self.aux_k = max(512, d_hidden // 8)
        else:
            self.aux_k = int(aux_k)
        self.aux_loss_coeff: float = float(aux_loss_coeff)

        # Kept at 0.0 for SAETrainer / checkpoint compatibility.
        # Top-K SAEs do not use an L1 penalty — sparsity is structural.
        self.l1_coeff: float = 0.0

        # EMA of per-feature activation frequency, updated every training
        # forward pass. Used to identify dead features for the aux loss without
        # requiring a separate data pass.
        #   decay = 0.99  →  a feature must fire consistently over ~100 recent
        #   batches to be considered alive.
        # Registered as a buffer: saved/loaded with state_dict and moved to the
        # correct device automatically, but receives NO gradient.
        self.register_buffer("ema_activations", torch.zeros(d_hidden))
        self.ema_decay: float = 0.99
        self.dead_threshold: float = 1e-3  # ema value below which a feature is "dead"

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Center input (optionally) and return linear pre-activations z.

        Kept separate so forward() can reuse z for both the main top-k pass
        and the auxiliary dead-feature pass without recomputing it.

        Args:
            x: Input activations, shape (batch_size, d_model).

        Returns:
            z: Pre-activations, shape (batch_size, d_hidden).
        """
        if self.use_pre_bias:
            x = x - self.b_pre
        return F.linear(x, self.W_enc, self.b_enc)

    @staticmethod
    def _topk_scatter(z: torch.Tensor, k: int) -> torch.Tensor:
        """Apply Top-K selection + ReLU to pre-activations.

        Keeps the k largest values per row, zeros the rest, then applies ReLU.
        Differentiable w.r.t. z via standard autograd.

        Args:
            z: Pre-activations, shape (batch_size, n_features).
            k: Number of features to keep active.

        Returns:
            f: Sparse non-negative activations, shape (batch_size, n_features).
        """
        k = min(k, z.shape[-1])
        topk_values, topk_indices = z.topk(k=k, dim=-1)
        f = torch.zeros_like(z)
        f.scatter_(dim=-1, index=topk_indices, src=topk_values)
        return F.relu(f)

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to a sparse Top-K hidden representation.

        The EMA is NOT updated here so that analysis / eval calls to encode()
        do not corrupt the dead-feature statistics tracked by forward().

        Args:
            x: Input activations, shape (batch_size, d_model).

        Returns:
            f: Sparse hidden activations, shape (batch_size, d_hidden).
               At most k values per sample are non-zero.
        """
        z = self._pre_activations(x)
        return self._topk_scatter(z, self.k)

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
        Full forward pass: encode → decode → loss (main + optional auxiliary).

        Main loss:
            L_main = MSE(x, x_hat)

        Auxiliary loss (training only, when aux_k > 0):
            1. Identify dead features: those whose EMA activation frequency
               has dropped below dead_threshold (~1e-3).
            2. Run a separate top-aux_k selection over ONLY those dead features.
            3. Use them to reconstruct the residual (x − x_hat).
            4. L_aux = MSE(residual, x_hat_aux)

            Total: L = L_main + aux_loss_coeff * L_aux

        The EMA of per-feature activation frequency is updated every training
        forward pass, so dead features are tracked online at zero extra cost.

        Compatibility:
            loss_dict always contains "l1_loss" / "l1_scaled" keys (both 0.0)
            so downstream code written for SparseAutoencoder still works.
        """
        # ---- Compute pre-activations once; reuse for aux pass --------------
        z = self._pre_activations(x)

        # ---- Main top-k pass -----------------------------------------------
        f = self._topk_scatter(z, self.k)
        x_reconstructed = self.decode(f)
        mse_loss = F.mse_loss(x_reconstructed, x)

        # ---- EMA update (training only) ------------------------------------
        # Track per-feature activation frequency as an exponential moving
        # average so we can identify dead features cheaply every batch.
        if self.training:
            with torch.no_grad():
                batch_freq = (f > 0).float().mean(dim=0)  # (d_hidden,)
                self.ema_activations.mul_(self.ema_decay).add_(
                    batch_freq * (1.0 - self.ema_decay)
                )

        # ---- Auxiliary dead-feature loss (training only) -------------------
        # Provides gradient signal to features that never win the top-k
        # competition, preventing the rich-get-richer collapse seen with large
        # expansion factors (e.g. 24x with k=32).
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.training and self.aux_k > 0 and self.aux_loss_coeff > 0.0:
            with torch.no_grad():
                dead_mask = self.ema_activations < self.dead_threshold  # (d_hidden,)
                n_dead = int(dead_mask.sum().item())

            if n_dead > 0:
                k_aux = min(self.aux_k, n_dead)

                # Pre-activations for dead features only: (batch, n_dead)
                z_dead = z[:, dead_mask]

                # Top-k_aux selection among dead features only.
                f_aux = self._topk_scatter(z_dead, k_aux)  # (batch, n_dead)

                # Decode using only the dead-feature decoder columns.
                # No b_dec offset: we are targeting the residual, not x itself.
                assert self.W_dec is not None
                W_dec_dead = self.W_dec[:, dead_mask]  # (d_model, n_dead)
                x_hat_aux = F.linear(f_aux, W_dec_dead)  # (batch, d_model)

                # Target: residual after main reconstruction (detached so
                # gradients do NOT flow back through the main reconstruction path).
                residual = (x - x_reconstructed).detach()
                aux_loss = F.mse_loss(x_hat_aux, residual)

        total_loss = mse_loss + self.aux_loss_coeff * aux_loss

        if return_loss_components:
            active_per_sample = (f > 0).float().sum(dim=-1)  # (batch,)
            with torch.no_grad():
                n_ema_dead = int(
                    (self.ema_activations < self.dead_threshold).sum().item()
                )
            loss_dict = {
                "loss": total_loss.item(),
                "mse_loss": mse_loss.item(),
                # Always 0.0 — kept for SAETrainer / logging compatibility
                "l1_loss": 0.0,
                "l1_scaled": 0.0,
                # Auxiliary loss diagnostics
                "aux_loss": aux_loss.item(),
                "aux_loss_scaled": (self.aux_loss_coeff * aux_loss).item(),
                "ema_dead_features": n_ema_dead,
                # Standard SAE diagnostics
                "mean_activation": f.mean().item(),
                "frac_active": (f > 0).float().mean().item(),
                "max_activation": f.max().item(),
                # Top-K specific
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
            aux_k=min(self.aux_k, new_d_hidden // 2),
            aux_loss_coeff=self.aux_loss_coeff,
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
        # Carry over the EMA so dead-feature tracking survives pruning.
        pruned.ema_activations.copy_(self.ema_activations[keep_indices])

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

        ema_dead = int((self.ema_activations < self.dead_threshold).sum().item())

        return {
            "feature_frequencies": feature_frequencies,
            "feature_magnitudes": feature_magnitudes,
            "dead_features": dead_features,
            "dead_fraction": dead_features / self.d_hidden,
            "mean_frequency": feature_frequencies.mean().item(),
            "median_frequency": feature_frequencies.median().item(),
            "mean_k_active": total_active / max(total_samples, 1),
            # EMA-based dead count: reflects online training history rather
            # than a single eval pass, useful for monitoring convergence.
            "ema_dead_features": ema_dead,
            "ema_dead_fraction": ema_dead / self.d_hidden,
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
    aux_k: Optional[int] = None,
    aux_loss_coeff: float = 1 / 32,
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
        aux_k:            Dead-feature aux-loss budget. Defaults to d_hidden//2.
                          Set to 0 to disable the auxiliary loss.
        aux_loss_coeff:   Weight of the auxiliary loss. Default: 1/32.
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

    # Calculate default aux_k with the same logic as __init__
    if aux_k is None:
        expansion = expansion_factor
        if expansion <= 8:
            _aux_k_display = d_hidden // 2
        elif expansion <= 16:
            _aux_k_display = d_hidden // 4
        else:
            _aux_k_display = max(512, d_hidden // 8)
    else:
        _aux_k_display = aux_k

    print(f"Creating Top-K SAE for {model_name}")
    print(f"  d_model:          {d_model}")
    print(f"  d_hidden:         {d_hidden}  (expansion x{expansion_factor})")
    print(f"  k:                {k}  (~{k / d_model:.1%} of d_model active per token)")
    print(
        f"  aux_k:            {_aux_k_display}  (aux loss {'disabled' if _aux_k_display == 0 else 'enabled'})"
    )
    print(f"  aux_loss_coeff:   {aux_loss_coeff:.4f}")
    print(f"  use_tied_weights: {use_tied_weights}")

    return TopKSparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        k=k,
        use_tied_weights=use_tied_weights,
        aux_k=aux_k,
        aux_loss_coeff=aux_loss_coeff,
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
