#!/usr/bin/env python3
"""
Training script for Feature Choice Sparse Autoencoder (FC-SAE)

Feature Choice SAEs (Ayonrinde, 2024) enforce sparsity per feature instead of
per token. Each feature is constrained to activate for exactly m tokens,
ensuring uniform feature utilization and preventing dead features.

Key Difference from Standard SAE:
- Standard SAE: L1 penalty limits features per token
- Feature Choice SAE: Hard constraint limits tokens per feature

Usage:
    python run_FC_sae.py --help
    python run_FC_sae.py --layer 8 --samples 50000 --epochs 30 --m 32
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from data_collection import GPT2ActivationCollector, prepare_training_data
from fc_sae_model import FeatureChoiceSAE, create_fc_sae_for_gpt2
from interpretation import FeatureAnalyzer
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional


class FCSAETrainer:
    """
    Trainer for Feature Choice SAE.

    Similar to SAETrainer but optimized for FC-SAE which doesn't use L1 penalty.
    """

    def __init__(
        self,
        model: FeatureChoiceSAE,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        lr: float = 1e-3,
        batch_size: int = 64,
        grad_clip_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == "cuda" else False,
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "feature_density": [],
            "mean_features_per_token": [],
            "learning_rate": [],
        }

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_metrics = {
            "loss": 0.0,
            "mse_loss": 0.0,
            "aux_loss": 0.0,
            "frac_active": 0.0,
            "mean_features_per_token": 0.0,
        }

        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            x = batch[0].to(self.device)

            self.optimizer.zero_grad()

            x_reconstructed, loss, loss_dict = self.model(
                x, return_loss_components=True
            )

            loss.backward()

            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )

            self.optimizer.step()

            # Normalize decoder weights
            self.model.normalize_decoder_weights()

            epoch_metrics["loss"] += loss_dict["loss"]
            epoch_metrics["mse_loss"] += loss_dict["mse_loss"]
            epoch_metrics["aux_loss"] += loss_dict.get("aux_loss", 0.0)
            epoch_metrics["frac_active"] += loss_dict["frac_active"]
            epoch_metrics["mean_features_per_token"] += loss_dict.get(
                "mean_features_per_token", 0.0
            )
            num_batches += 1

        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        val_metrics = {
            "loss": 0.0,
            "mse_loss": 0.0,
            "aux_loss": 0.0,
            "frac_active": 0.0,
            "mean_features_per_token": 0.0,
        }

        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            x = batch[0].to(self.device)

            x_reconstructed, loss, loss_dict = self.model(
                x, return_loss_components=True
            )

            val_metrics["loss"] += loss_dict["loss"]
            val_metrics["mse_loss"] += loss_dict["mse_loss"]
            val_metrics["aux_loss"] += loss_dict.get("aux_loss", 0.0)
            val_metrics["frac_active"] += loss_dict["frac_active"]
            val_metrics["mean_features_per_token"] += loss_dict.get(
                "mean_features_per_token", 0.0
            )
            num_batches += 1

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
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        print(f"Starting FC-SAE training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Tokens per feature (m): {self.model.m}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        if disable_early_stopping:
            early_stopping_patience = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_mse"].append(train_metrics["mse_loss"])
            self.history["val_mse"].append(val_metrics["mse_loss"])
            self.history["feature_density"].append(train_metrics["frac_active"])
            self.history["mean_features_per_token"].append(
                train_metrics["mean_features_per_token"]
            )
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]['lr']
            )

            if (epoch + 1) % log_every == 0:
                print(
                    f"  Train Loss: {train_metrics['loss']:.6f} "
                    f"(MSE: {train_metrics['mse_loss']:.6f}, Aux: {train_metrics['aux_loss']:.6f})"
                )
                print(
                    f"  Val Loss:   {val_metrics['loss']:.6f} "
                    f"(MSE: {val_metrics['mse_loss']:.6f})"
                )
                print(f"  Feature Density: {train_metrics['frac_active']:.2%}")
                print(
                    f"  Mean Features/Token: {train_metrics['mean_features_per_token']:.1f}"
                )
                print()

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                print(f"  New best model! Val loss: {val_metrics['loss']:.6f}")
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pt", epoch, val_metrics
                )

            if (
                early_stopping_patience
                and early_stopping_patience > 0
                and self.epochs_without_improvement >= early_stopping_patience
            ):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

        return self.history

    def save_checkpoint(
        self, filename: str, epoch: int, metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
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
                "m_tokens_per_feature": self.model.m,
                "aux_loss_coeff": self.model.aux_loss_coeff,
                "lr": self.lr,
                "batch_size": self.batch_size,
            },
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Total Loss
        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train", alpha=0.7)
        axes[0, 0].plot(epochs, self.history["val_loss"], label="Validation", alpha=0.7)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("Total Loss (MSE + Aux)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE Loss
        axes[0, 1].plot(epochs, self.history["train_mse"], label="Train", alpha=0.7)
        axes[0, 1].plot(epochs, self.history["val_mse"], label="Validation", alpha=0.7)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE Loss")
        axes[0, 1].set_title("Reconstruction Error (MSE)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Feature Density
        axes[1, 0].plot(
            epochs, [d * 100 for d in self.history["feature_density"]], alpha=0.7
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Active Features (%)")
        axes[1, 0].set_title("Feature Density")
        axes[1, 0].grid(True, alpha=0.3)

        # Mean Features per Token
        axes[1, 1].plot(epochs, self.history["mean_features_per_token"], alpha=0.7)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Features")
        axes[1, 1].set_title("Mean Features per Token")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train a Feature Choice SAE on GPT-2 hidden layers"
    )

    # Data collection arguments
    parser.add_argument(
        "--layer", type=int, default=8,
        help="GPT-2 layer to analyze (0-11). Default: 8"
    )
    parser.add_argument(
        "--samples", type=int, default=50000,
        help="Number of activation samples to collect. Default: 50000"
    )
    parser.add_argument(
        "--dataset", type=str, default="openwebtext",
        help="HuggingFace dataset to use. Default: openwebtext"
    )
    parser.add_argument(
        "--dataset-config", type=str, default=None,
        help="Dataset configuration name (e.g., 'wikitext-103-v1' for wikitext)"
    )
    parser.add_argument(
        "--num-texts", type=int, default=None,
        help="Number of texts to load from the dataset."
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Max token length per text. Default: 128"
    )
    parser.add_argument(
        "--collection-batch-size", type=int, default=16,
        help="Batch size for activation collection. Default: 16"
    )
    parser.add_argument(
        "--shuffle-buffer-size", type=int, default=10_000,
        help="Shuffle buffer size for streaming datasets. Default: 10000"
    )
    parser.add_argument(
        "--dataset-seed", type=int, default=0,
        help="Shuffle seed for streaming dataset. Default: 0"
    )

    # Model arguments
    parser.add_argument(
        "--expansion", type=int, default=16,
        help="Expansion factor (d_hidden / d_model). Default: 16"
    )
    parser.add_argument(
        "--m", type=int, default=32,
        help="Tokens per feature constraint. Default: 32"
    )
    parser.add_argument(
        "--aux-loss-coeff", type=float, default=1e-3,
        help="Auxiliary loss coefficient. Default: 1e-3"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Maximum training epochs. Default: 30"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size. Default: 256"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate. Default: 1e-3"
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0,
        help="Max grad norm for clipping. Default: 1.0"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Early stopping patience in epochs. Default: 10"
    )
    parser.add_argument(
        "--disable-early-stopping", action="store_true",
        help="Disable early stopping."
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs. Default: 10"
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="Print metrics every N epochs. Default: 1"
    )

    # Other arguments
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (cuda/cpu/auto). Default: auto"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints_fc",
        help="Directory to save checkpoints. Default: checkpoints_fc"
    )
    parser.add_argument(
        "--skip-collection", action="store_true",
        help="Skip data collection (use existing activations.pt)"
    )
    parser.add_argument(
        "--activations-path", type=str, default="activations.pt",
        help="Path to activations file. Default: activations.pt"
    )

    # Data normalization
    parser.add_argument(
        "--normalize-mode", type=str, default="standardize",
        choices=["standardize", "center", "none"],
        help="How to normalize activations. Default: standardize"
    )
    parser.add_argument(
        "--std-floor", type=float, default=1e-3,
        help="Std floor for normalization. Default: 1e-3"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("FEATURE CHOICE SAE TRAINING FOR GPT-2")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target layer: {args.layer}")
    print(f"  Samples: {args.samples:,}")
    print(f"  Expansion factor: {args.expansion}x")
    print(f"  Tokens per feature (m): {args.m}")
    print(f"  Aux loss coefficient: {args.aux_loss_coeff}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print()

    # Step 1: Collect activations
    if not args.skip_collection:
        print("Step 1: Collecting activations...")
        print("-" * 70)

        collector = GPT2ActivationCollector(
            model_name="gpt2",
            layer_index=args.layer,
            device=device
        )

        activations = collector.collect_from_dataset(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            num_texts=(
                args.num_texts if args.num_texts is not None
                else args.samples // 10
            ),
            max_samples=args.samples,
            batch_size=args.collection_batch_size,
            max_length=args.max_length,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.dataset_seed,
        )

        torch.save(activations, args.activations_path)
        print(f"\nActivations saved to {args.activations_path}")
    else:
        print("Step 1: Loading existing activations...")
        print("-" * 70)
        activations = torch.load(args.activations_path)
        print(f"Loaded {activations.shape[0]} activations")

    # Step 2: Prepare data
    print("\nStep 2: Preparing training data...")
    print("-" * 70)

    train_data, val_data, stats = prepare_training_data(
        activations,
        train_ratio=0.9,
        normalize=True,
        normalize_mode=args.normalize_mode,
        std_floor=args.std_floor,
    )

    # Step 3: Create model
    print("\nStep 3: Creating Feature Choice SAE...")
    print("-" * 70)

    sae = create_fc_sae_for_gpt2(
        model_name="gpt2",
        expansion_factor=args.expansion,
        m_tokens_per_feature=args.m,
        aux_loss_coeff=args.aux_loss_coeff,
    )

    # Step 4: Train
    print("\nStep 4: Training...")
    print("-" * 70)

    trainer = FCSAETrainer(
        model=sae,
        train_data=train_data,
        val_data=val_data,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip_norm=args.grad_clip_norm,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        disable_early_stopping=args.disable_early_stopping,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    # Plot results
    trainer.plot_training_history(
        save_path=f"{args.checkpoint_dir}/training_history.png"
    )

    # Load best model for analysis
    trainer.load_checkpoint("best_model.pt")

    # Step 5: Analysis
    print("\nStep 5: Analyzing results...")
    print("-" * 70)

    # Get feature statistics
    stats = sae.get_feature_statistics(trainer.val_loader)
    print(f"Feature utilization statistics:")
    print(f"  Dead features: {stats['dead_features']} ({stats['dead_fraction']:.2%})")
    print(f"  Mean frequency: {stats['mean_frequency']:.4f}")
    print(f"  Min frequency: {stats['min_frequency']:.4f}")
    print(f"  Max frequency: {stats['max_frequency']:.4f}")

    # Reconstruction quality
    with torch.no_grad():
        sae.eval()
        val_sample = val_data[:1000].to(device)
        f = sae.encode(val_sample)
        x_hat = sae.decode(f)

        mse = ((val_sample - x_hat) ** 2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            val_sample, x_hat, dim=1
        ).mean().item()
        explained_var = 1 - mse / val_sample.var().item()

    print(f"\nReconstruction quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Explained variance: {explained_var:.6f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Metrics:")
    print(f"  Validation Loss: {history['val_loss'][-1]:.6f}")
    print(f"  Feature Density: {history['feature_density'][-1]:.2%}")
    print(f"  Mean Features/Token: {history['mean_features_per_token'][-1]:.1f}")

    print(f"\nOutputs saved to: {args.checkpoint_dir}/")
    print(f"  best_model.pt - Best model checkpoint")
    print(f"  training_history.png - Training curves")


if __name__ == "__main__":
    main()
