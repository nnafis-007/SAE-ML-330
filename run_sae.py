#!/usr/bin/env python3
"""
Quick start script for SAE training on GPT-2

This script provides a simple command-line interface to:
1. Collect activations from GPT-2
2. Train a Sparse Autoencoder
3. Analyze the results

Usage:
    python run_sae.py --help
    python run_sae.py --layer 8 --samples 50000 --epochs 30
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from transformers import GPT2Tokenizer

from data_collection import GPT2ActivationCollector, prepare_training_data
from interpretation import FeatureAnalyzer
from sae_model import create_sae_for_gpt2, create_topk_sae_for_gpt2, suggest_k
from training import SAETrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on GPT-2 hidden layers"
    )

    # Data collection arguments
    parser.add_argument(
        "--layer", type=int, default=8, help="GPT-2 layer to analyze (0-11). Default: 8"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of activation samples to collect. Default: 50000",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="HuggingFace dataset to use. Default: openwebtext",
    )

    parser.add_argument(
        "--num-texts",
        type=int,
        default=None,
        help="Number of texts to load from the dataset (overrides samples//10).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length per text when collecting activations. Default: 128",
    )
    parser.add_argument(
        "--collection-batch-size",
        type=int,
        default=16,
        help="Batch size for activation collection (GPT-2 forward). Default: 16",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10_000,
        help="Shuffle buffer size for streaming datasets (0 disables shuffling). Default: 10000",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=0,
        help="Shuffle seed for streaming dataset. Default: 0",
    )

    # Model arguments
    parser.add_argument(
        "--expansion",
        type=int,
        default=16,
        help="Expansion factor (d_hidden / d_model). Default: 16",
    )
    parser.add_argument(
        "--l1-coeff",
        type=float,
        default=3e-4,
        help="L1 sparsity coefficient. Default: 3e-4",
    )

    # SAE variant selection
    parser.add_argument(
        "--sae-type",
        type=str,
        default="standard",
        choices=["standard", "topk"],
        help=(
            "SAE architecture variant. "
            "'standard' uses ReLU + L1 penalty (original paper). "
            "'topk' uses hard Top-K selection with MSE-only loss. "
            "Default: standard"
        ),
    )
    parser.add_argument(
        "--topk-k",
        type=int,
        default=None,
        help=(
            "Number of active features per token for the Top-K SAE. "
            "Only used when --sae-type=topk. "
            "If omitted, suggest_k(d_model, expansion_factor) is called automatically "
            "(targets ~4%% of d_model active per token, e.g. k~35 for gpt2 + expansion=16)."
        ),
    )
    parser.add_argument(
        "--aux-k",
        type=int,
        default=None,
        help=(
            "Number of dead features used in the auxiliary reconstruction pass. "
            "Only used when --sae-type=topk. "
            "Defaults to a scaled value based on expansion factor: "
            "expansion≤8x: d_hidden//2, expansion≤16x: d_hidden//4, expansion>16x: max(512, d_hidden//8). "
            "For large expansions, smaller aux_k gives focused gradients. "
            "Set to 0 to disable the auxiliary loss entirely."
        ),
    )
    parser.add_argument(
        "--aux-loss-coeff",
        type=float,
        default=1 / 32,
        help=(
            "Weight applied to the auxiliary loss term. "
            "Only used when --sae-type=topk. "
            "OpenAI paper uses 1/32 (~0.03125). "
            "Increase if dead features persist; decrease if reconstruction degrades. "
            "Default: 0.03125 (1/32)."
        ),
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=30, help="Maximum training epochs. Default: 30"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size. Default: 256"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate. Default: 1e-3"
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max grad norm for clipping (0 disables). Default: 1.0",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (0 disables). Default: 10",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping and always run all epochs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs. Default: 10",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print metrics every N epochs. Default: 1",
    )

    # Optional: dead-feature resampling
    parser.add_argument(
        "--resample-dead",
        action="store_true",
        help="Periodically re-initialize rarely-active features during training.",
    )
    parser.add_argument(
        "--resample-every",
        type=int,
        default=1,
        help="Resample cadence in epochs. Default: 1",
    )
    parser.add_argument(
        "--resample-freq-threshold",
        type=float,
        default=0.0002,
        help="Resample features with activation frequency below this threshold. Default: 0.0002",
    )
    parser.add_argument(
        "--resample-eval-samples",
        type=int,
        default=8192,
        help="Training samples used to estimate feature frequency. Default: 8192",
    )
    parser.add_argument(
        "--resample-max-features",
        type=int,
        default=2048,
        help="Max features to resample per step. Default: 2048",
    )
    parser.add_argument(
        "--resample-start-epoch",
        type=int,
        default=5,
        help="First epoch (1-indexed) at which resampling is allowed. Default: 5",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto). Default: auto",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints. Default: checkpoints",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection (use existing activations.pt)",
    )

    # Data normalization
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="standardize",
        choices=["standardize", "center", "none"],
        help="How to normalize activations before training. Default: standardize",
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-3,
        help="Clamp per-dimension std to this floor when standardizing (prevents blowups). Default: 1e-3",
    )

    # Optional: prune dead features after training (Don't do it)
    parser.add_argument(
        "--save-pruned",
        action="store_true",
        help="Save a pruned SAE with dead features removed (default thresholds).",
    )
    parser.add_argument(
        "--dead-freq-threshold",
        type=float,
        default=0.001,
        help="Dead feature frequency threshold used for pruning/reporting. Default: 0.001",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("SPARSE AUTOENCODER TRAINING FOR GPT-2")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target layer: {args.layer}")
    print(f"  Samples: {args.samples:,}")
    print(f"  Expansion factor: {args.expansion}x")
    print(f"  SAE type: {args.sae_type}")
    if args.sae_type == "standard":
        print(f"  L1 coefficient: {args.l1_coeff}")
    else:
        # Resolve k early so we can print the concrete value.
        _k_display = (
            args.topk_k if args.topk_k is not None else suggest_k(768, args.expansion)
        )
        _d_hidden_display = 768 * args.expansion
        # Calculate default aux_k with same logic as TopKSparseAutoencoder.__init__
        if args.aux_k is None:
            expansion = args.expansion
            if expansion <= 8:
                _aux_k_display = _d_hidden_display // 2
            elif expansion <= 16:
                _aux_k_display = _d_hidden_display // 4
            else:
                _aux_k_display = max(512, _d_hidden_display // 8)
        else:
            _aux_k_display = args.aux_k

        print(f"  Top-K k:          {_k_display} (auto={args.topk_k is None})")
        print(
            f"  aux_k:            {_aux_k_display} ({'disabled' if _aux_k_display == 0 else 'enabled'})"
        )
        print(f"  aux_loss_coeff:   {args.aux_loss_coeff:.4f}")
    print(f"  Epochs: {args.epochs}")
    print(
        f"  Early stopping patience: {args.early_stopping_patience}{' (disabled)' if args.disable_early_stopping or args.early_stopping_patience == 0 else ''}"
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print()

    # Step 1: Collect activations
    if not args.skip_collection:
        print("Step 1: Collecting activations...")
        print("-" * 70)

        collector = GPT2ActivationCollector(
            model_name="gpt2", layer_index=args.layer, device=device
        )

        activations = collector.collect_from_dataset(
            dataset_name=args.dataset,
            num_texts=(
                args.num_texts if args.num_texts is not None else args.samples // 10
            ),  # Approximate
            max_samples=args.samples,
            batch_size=args.collection_batch_size,
            max_length=args.max_length,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.dataset_seed,
        )

        # Save activations
        torch.save(activations, "activations.pt")
        print(f"\n✓ Activations saved to activations.pt")
    else:
        print("Step 1: Loading existing activations...")
        print("-" * 70)
        activations = torch.load("activations.pt")
        print(f"✓ Loaded {activations.shape[0]} activations")

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
    print("\nStep 3: Creating Sparse Autoencoder...")
    print("-" * 70)

    if args.sae_type == "topk":
        # Top-K SAE: sparsity is enforced by hard Top-K selection.
        # No L1 penalty is used; the loss is purely MSE + aux_loss.
        # If --topk-k is not specified, suggest_k() picks a sensible default
        # based on the model dimension and expansion factor.
        # The auxiliary loss (aux_k, aux_loss_coeff) prevents the dead-feature
        # collapse that occurs with large expansion factors and small k.
        sae = create_topk_sae_for_gpt2(
            model_name="gpt2",
            expansion_factor=args.expansion,
            k=args.topk_k,  # None → auto-computed inside factory
            aux_k=args.aux_k,  # None → d_hidden // 2
            aux_loss_coeff=args.aux_loss_coeff,
        )
    else:
        # Standard SAE: ReLU activation + L1 sparsity penalty.
        sae = create_sae_for_gpt2(
            model_name="gpt2",
            expansion_factor=args.expansion,
            l1_coeff=args.l1_coeff,
        )

    # Step 4: Train
    print("\nStep 4: Training...")
    print("-" * 70)

    trainer = SAETrainer(
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
        resample_dead_features=args.resample_dead,
        resample_every=args.resample_every,
        resample_freq_threshold=args.resample_freq_threshold,
        resample_eval_samples=args.resample_eval_samples,
        resample_max_features=args.resample_max_features,
        resample_start_epoch=args.resample_start_epoch,
    )

    # Plot results
    trainer.plot_training_history(
        save_path=f"{args.checkpoint_dir}/training_history.png"
    )

    # Ensure we analyze/prune the best-performing checkpoint, not the last epoch.
    trainer.load_checkpoint("best_model.pt")

    # Step 5: Analysis
    print("\nStep 5: Analyzing results...")
    print("-" * 70)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    analyzer = FeatureAnalyzer(sae, tokenizer, device)

    # Generate report
    analyzer.create_summary_report(
        activations=val_data,
        texts=[f"Sample {i}" for i in range(len(val_data))],
        save_path=f"{args.checkpoint_dir}/analysis_report.txt",
        freq_threshold=args.dead_freq_threshold,
    )

    if args.save_pruned:
        analyzer.prune_dead_features_and_save(
            activations=val_data,
            save_path=f"{args.checkpoint_dir}/pruned_model.pt",
            freq_threshold=args.dead_freq_threshold,
        )

    # Reconstruction quality
    metrics = analyzer.get_reconstruction_quality(val_data)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Metrics:")
    print(f"  SAE type: {args.sae_type}")
    if args.sae_type == "topk":
        print(f"  k (active features/token): {sae.k}")
        print(f"  aux_k:                     {sae.aux_k}")
        print(f"  aux_loss_coeff:            {sae.aux_loss_coeff:.4f}")
    else:
        print(f"  L1 coefficient: {args.l1_coeff}")
    print(f"  Validation Loss: {history['val_loss'][-1]:.6f}")
    print(f"  Feature Density: {history['feature_density'][-1]:.2%}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    print(f"  Explained Variance: {metrics['explained_variance']:.6f}")

    print(f"\nOutputs saved to: {args.checkpoint_dir}/")
    print(f"  • best_model.pt - Best model checkpoint")
    print(f"  • training_history.png - Training curves")
    print(f"  • analysis_report.txt - Detailed analysis")

    print("\n💡 Next: Open notebooks/tutorial.ipynb for detailed analysis!")


if __name__ == "__main__":
    main()
