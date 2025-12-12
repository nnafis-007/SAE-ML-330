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
        save_every: int = 5,
        log_every: int = 1
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if validation loss doesn't improve for this many epochs
            save_every: Save checkpoint every N epochs
            log_every: Print metrics every N epochs
        
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
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return self.history
    
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
