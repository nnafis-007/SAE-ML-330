"""
Training Module for Sparse Autoencoder

This module handles:
1. Training loop with proper optimization
2. Validation and early stopping
3. Learning rate scheduling
4. Checkpoint saving
5. Logging and monitoring

Key training considerations from the paper:
- Adam optimizer works well
- Learning rate warmup helps stability
- Periodic decoder normalization is critical
- Monitor both reconstruction quality and sparsity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Callable, Dict, List
from pathlib import Path
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from sae_model import SparseAutoencoder


class SAETrainer:
    """
    Trainer class for Sparse Autoencoder.
    
    Handles all aspects of training including:
    - Optimization and learning rate scheduling
    - Validation and checkpointing
    - Metrics tracking and visualization
    - Early stopping
    """
    
    def __init__(
        self,
        model: SparseAutoencoder,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        lr: float = 1e-3,
        batch_size: int = 256,
        grad_clip_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SparseAutoencoder to train
            train_data: Training activations
            val_data: Validation activations
            lr: Learning rate (Adam default of 1e-3 works well)
            batch_size: Batch size (larger = more stable gradients, but needs more memory)
            device: Computing device
            checkpoint_dir: Directory to save checkpoints
            
        Why these hyperparameters?
        - Learning rate 1e-3: Standard for Adam, works well for SAEs
        - Batch size 256-1024: Balances gradient quality and memory
        - Adam: Handles the different scales of encoder/decoder well
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create data loaders
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for better training
            num_workers=0,  # Increase if CPU preprocessing is bottleneck
            pin_memory=True if device == "cuda" else False  # Faster GPU transfer
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation
            num_workers=0,
            pin_memory=True if device == "cuda" else False
        )
        
        # Initialize optimizer
        # Adam is preferred because:
        # - Handles different learning rates for different parameters
        # - Robust to learning rate choice
        # - Works well with sparse gradients
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_l1": [],
            "val_l1": [],
            "feature_density": [],
            "learning_rate": []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with average metrics for the epoch
            
        Training loop:
        1. Forward pass through SAE
        2. Compute loss (MSE + L1)
        3. Backward pass (compute gradients)
        4. Optimizer step (update weights)
        5. Normalize decoder weights (critical!)
        """
        self.model.train()
        
        epoch_metrics = {
            "loss": 0.0,
            "mse_loss": 0.0,
            "l1_loss": 0.0,
            "frac_active": 0.0
        }
        
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Get batch data
            x = batch[0].to(self.device)
            
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Forward pass
            x_reconstructed, loss, loss_dict = self.model(
                x, 
                return_loss_components=True
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            # Not always necessary but can help stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # CRITICAL: Normalize decoder weights
            # This prevents features from growing large to cheat the L1 penalty
            self.model.normalize_decoder_weights()
            
            # Accumulate metrics
            epoch_metrics["loss"] += loss_dict["loss"]
            epoch_metrics["mse_loss"] += loss_dict["mse_loss"]
            epoch_metrics["l1_loss"] += loss_dict["l1_loss"]
            epoch_metrics["frac_active"] += loss_dict["frac_active"]
            num_batches += 1
        
        # Average over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
            
        Validation is done without gradients (torch.no_grad) for efficiency.
        We use the same metrics as training to track generalization.
        """
        self.model.eval()
        
        val_metrics = {
            "loss": 0.0,
            "mse_loss": 0.0,
            "l1_loss": 0.0,
            "frac_active": 0.0
        }
        
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            x = batch[0].to(self.device)
            
            # Forward pass only
            x_reconstructed, loss, loss_dict = self.model(
                x,
                return_loss_components=True
            )
            
            # Accumulate metrics
            val_metrics["loss"] += loss_dict["loss"]
            val_metrics["mse_loss"] += loss_dict["mse_loss"]
            val_metrics["l1_loss"] += loss_dict["l1_loss"]
            val_metrics["frac_active"] += loss_dict["frac_active"]
            num_batches += 1
        
        # Average over batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        disable_early_stopping: bool = False,
        save_every: int = 5,
        log_every: int = 1,
        resample_dead_features: bool = False,
        resample_every: int = 1,
        resample_freq_threshold: float = 0.0,
        resample_eval_samples: int = 8192,
        resample_max_features: int = 2048,
        resample_start_epoch: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if validation loss doesn't improve for this many epochs
            disable_early_stopping: If True, never early-stop
            save_every: Save checkpoint every N epochs
            log_every: Print metrics every N epochs
            resample_dead_features: If True, periodically re-initialize rarely-active features
            resample_every: Resample cadence in epochs (e.g. 1 = every epoch)
            resample_freq_threshold: Features with activation frequency below this threshold are resampled
            resample_eval_samples: Number of training samples used to estimate activation frequency
            resample_max_features: Maximum number of features to resample per step
            resample_start_epoch: First epoch (1-indexed) at which resampling is allowed
        
        Returns:
            Training history dictionary
            
        Training strategy:
        - Train until validation loss stops improving (early stopping)
        - Save best model based on validation loss
        - Track metrics for analysis and debugging
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        if disable_early_stopping:
            early_stopping_patience = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_mse"].append(train_metrics["mse_loss"])
            self.history["val_mse"].append(val_metrics["mse_loss"])
            self.history["train_l1"].append(train_metrics["l1_loss"])
            self.history["val_l1"].append(val_metrics["l1_loss"])
            self.history["feature_density"].append(train_metrics["frac_active"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            
            # Log metrics
            if (epoch + 1) % log_every == 0:
                print(f"  Train Loss: {train_metrics['loss']:.6f} "
                      f"(MSE: {train_metrics['mse_loss']:.6f}, L1: {train_metrics['l1_loss']:.6f})")
                print(f"  Val Loss:   {val_metrics['loss']:.6f} "
                      f"(MSE: {val_metrics['mse_loss']:.6f}, L1: {val_metrics['l1_loss']:.6f})")
                print(f"  Feature Density: {train_metrics['frac_active']:.2%}")
                print()
            
            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                print(f"  ✓ New best model! Val loss: {val_metrics['loss']:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", epoch, val_metrics)

            # Optional: resample dead/rare features to improve utilization in overcomplete SAEs.
            # This can help when training collapses to a small subset of always-on features.
            if (
                resample_dead_features
                and resample_freq_threshold > 0.0
                and resample_every > 0
                and (epoch + 1) >= int(resample_start_epoch)
                and ((epoch + 1) % resample_every == 0)
            ):
                freqs = self._estimate_feature_frequencies(max_samples=resample_eval_samples)
                dead_mask = freqs < float(resample_freq_threshold)
                num_resampled = self._resample_features(dead_mask, max_features=resample_max_features)
                if num_resampled > 0:
                    print(
                        f"  ↻ Resampled {num_resampled} features "
                        f"(freq < {resample_freq_threshold:g} over ~{resample_eval_samples} samples)"
                    )
            
            # Early stopping
            if early_stopping_patience and early_stopping_patience > 0 and self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return self.history

    @torch.no_grad()
    def _estimate_feature_frequencies(self, max_samples: int = 8192) -> torch.Tensor:
        """Estimate per-feature activation frequency on a subset of training data."""
        self.model.eval()

        device = next(self.model.parameters()).device
        total = 0
        counts = torch.zeros(self.model.d_hidden, device=device)

        for batch in self.train_loader:
            x = batch[0].to(device)
            f = self.model.encode(x)
            counts += (f > 0).sum(dim=0)
            total += x.shape[0]
            if total >= max_samples:
                break

        if total == 0:
            return torch.zeros(self.model.d_hidden, device=device)

        return counts / float(total)

    @torch.no_grad()
    def _resample_features(self, dead_mask: torch.Tensor, max_features: int = 2048) -> int:
        """Re-initialize encoder/decoder params for a subset of dead features.

        Note: For large, overcomplete SAEs (e.g. 32-64x), purely-random reinit
        often produces features that remain dead. We instead seed new decoder
        directions from real residuals on a minibatch, then align the encoder
        row with that direction and set the bias so the feature activates on a
        small fraction of that minibatch.
        """
        if dead_mask is None:
            return 0

        dead_mask = dead_mask.to(next(self.model.parameters()).device).bool().flatten()
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return 0

        if self.model.use_tied_weights:
            # Keeping behavior simple: resampling requires independent decoder columns.
            return 0

        if max_features and dead_indices.numel() > max_features:
            perm = torch.randperm(dead_indices.numel(), device=dead_indices.device)
            dead_indices = dead_indices[perm[: int(max_features)]]

        device = next(self.model.parameters()).device
        d_model = self.model.d_model

        # Pick a minibatch and compute residuals in activation space.
        # This makes resampled decoder columns start on-manifold.
        try:
            batch = next(iter(self.train_loader))[0].to(device)
        except Exception:
            batch = None

        if batch is None or batch.numel() == 0:
            # Fallback: if we cannot access data, do a random re-init.
            self.model.W_enc.data[dead_indices] = (
                torch.randn(dead_indices.numel(), d_model, device=self.model.W_enc.device)
                / (d_model ** 0.5)
            )
            self.model.b_enc.data[dead_indices] = 0.0
            new_cols = torch.randn(d_model, dead_indices.numel(), device=self.model.W_dec.device)
            new_cols = torch.nn.functional.normalize(new_cols, dim=0)
            self.model.W_dec.data[:, dead_indices] = new_cols
        else:
            x = batch
            f = self.model.encode(x)
            x_hat = self.model.decode(f)
            residual = (x - x_hat).detach()

            # Sample residual vectors to seed decoder columns.
            k = int(dead_indices.numel())
            n = int(residual.shape[0])
            if n == 0:
                return 0

            # Choose residual examples with probability proportional to residual norm.
            norms = residual.norm(dim=1)
            probs = norms / (norms.sum() + 1e-12)
            sample_idx = torch.multinomial(probs, num_samples=k, replacement=(k > n))
            new_cols = residual[sample_idx].T.contiguous()  # (d_model, k)
            new_cols = torch.nn.functional.normalize(new_cols, dim=0)

            # Set decoder columns for these features.
            self.model.W_dec.data[:, dead_indices] = new_cols

            # Align encoder rows with decoder directions.
            # This makes the feature's dot-product correlate with its decode direction.
            self.model.W_enc.data[dead_indices] = new_cols.T

            # Bias calibration: target a small activation rate on this minibatch
            # so features don't stay completely off.
            target_active = 0.01
            if hasattr(self.model, "l1_coeff") and self.model.l1_coeff is not None:
                # If L1 is very small, allow slightly higher initial activity.
                if float(self.model.l1_coeff) <= 1e-4:
                    target_active = 0.02

            # Compute pre-activations (without bias) on the minibatch.
            if self.model.use_pre_bias and getattr(self.model, "b_pre", None) is not None:
                x_centered = x - self.model.b_pre
            else:
                x_centered = x
            preact = torch.matmul(x_centered, self.model.W_enc.data[dead_indices].T)  # (batch, k)

            # Set b_enc so that ~target_active fraction is positive.
            # Use kthvalue for speed/compatibility.
            q = max(0.0, min(1.0, 1.0 - float(target_active)))
            kth = max(1, min(preact.shape[0], int(q * preact.shape[0])))
            thresh = preact.kthvalue(kth, dim=0).values
            self.model.b_enc.data[dead_indices] = (-thresh).to(self.model.b_enc.device)

        if self.model.normalize_decoder:
            self.model.normalize_decoder_weights()

        # Best-effort: reset Adam moments for the resampled slices.
        try:
            for param in (self.model.W_enc, self.model.b_enc, self.model.W_dec):
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                if "exp_avg" in state:
                    exp_avg = state["exp_avg"]
                    if param is self.model.W_enc:
                        exp_avg[dead_indices] = 0
                    elif param is self.model.b_enc:
                        exp_avg[dead_indices] = 0
                    elif param is self.model.W_dec:
                        exp_avg[:, dead_indices] = 0
                if "exp_avg_sq" in state:
                    exp_avg_sq = state["exp_avg_sq"]
                    if param is self.model.W_enc:
                        exp_avg_sq[dead_indices] = 0
                    elif param is self.model.b_enc:
                        exp_avg_sq[dead_indices] = 0
                    elif param is self.model.W_dec:
                        exp_avg_sq[:, dead_indices] = 0
        except Exception:
            # Optimizer state reset is optional; ignore failures.
            pass

        return int(dead_indices.numel())
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
            epoch: Current epoch number
            metrics: Current metrics
            
        Checkpoint includes:
        - Model state (weights and biases)
        - Optimizer state (for resuming training)
        - Training history
        - Hyperparameters
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "metrics": metrics,
            "hyperparameters": {
                "d_model": self.model.d_model,
                "d_hidden": self.model.d_hidden,
                "l1_coeff": self.model.l1_coeff,
                "lr": self.lr,
                "batch_size": self.batch_size
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Args:
            filename: Name of checkpoint file to load
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
            
        Creates a multi-panel plot showing:
        1. Loss curves (train vs val)
        2. MSE and L1 components
        3. Feature density over time
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Plot 1: Total Loss
        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train", alpha=0.7)
        axes[0, 0].plot(epochs, self.history["val_loss"], label="Validation", alpha=0.7)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("Total Loss (MSE + L1)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: MSE Loss
        axes[0, 1].plot(epochs, self.history["train_mse"], label="Train", alpha=0.7)
        axes[0, 1].plot(epochs, self.history["val_mse"], label="Validation", alpha=0.7)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE Loss")
        axes[0, 1].set_title("Reconstruction Error (MSE)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: L1 Loss
        axes[1, 0].plot(epochs, self.history["train_l1"], label="Train", alpha=0.7)
        axes[1, 0].plot(epochs, self.history["val_l1"], label="Validation", alpha=0.7)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("L1 Loss")
        axes[1, 0].set_title("Sparsity Penalty (L1)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature Density
        axes[1, 1].plot(epochs, [d * 100 for d in self.history["feature_density"]], alpha=0.7)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Active Features (%)")
        axes[1, 1].set_title("Feature Density")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def train_sae(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    d_model: int = 768,
    expansion_factor: int = 16,
    l1_coeff: float = 3e-4,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_epochs: int = 100,
    early_stopping_patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "checkpoints"
) -> tuple[SparseAutoencoder, Dict[str, List[float]]]:
    """
    Convenience function to create and train an SAE.
    
    Args:
        train_data: Training activations
        val_data: Validation activations
        d_model: Input dimension
        expansion_factor: Hidden dimension multiplier
        l1_coeff: L1 sparsity coefficient
        lr: Learning rate
        batch_size: Batch size
        num_epochs: Maximum epochs
        early_stopping_patience: Early stopping patience
        device: Computing device
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Trained model and training history
        
    This function wraps the entire training process for convenience.
    """
    from sae_model import SparseAutoencoder
    
    # Create model
    model = SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_model * expansion_factor,
        l1_coeff=l1_coeff
    )
    
    # Create trainer
    trainer = SAETrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        lr=lr,
        batch_size=batch_size,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience
    )
    
    # Plot results
    trainer.plot_training_history(
        save_path=f"{checkpoint_dir}/training_history.png"
    )
    
    # Load best model
    trainer.load_checkpoint("best_model.pt")
    
    return model, history


if __name__ == "__main__":
    """
    Example: Train an SAE on sample data
    """
    # Generate sample data (replace with real GPT-2 activations)
    d_model = 768
    n_train = 10000
    n_val = 2000
    
    train_data = torch.randn(n_train, d_model)
    val_data = torch.randn(n_val, d_model)
    
    # Train
    model, history = train_sae(
        train_data=train_data,
        val_data=val_data,
        d_model=d_model,
        expansion_factor=16,
        l1_coeff=3e-4,
        num_epochs=50,
        batch_size=256
    )
    
    print("\nTraining complete!")
    print(f"Final feature density: {history['feature_density'][-1]:.2%}")
