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
from data_collection import GPT2ActivationCollector, prepare_training_data
from sae_model import create_sae_for_gpt2
from training import SAETrainer
from interpretation import FeatureAnalyzer
from transformers import GPT2Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on GPT-2 hidden layers"
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
    
    # Model arguments
    parser.add_argument(
        "--expansion", type=int, default=16,
        help="Expansion factor (d_hidden / d_model). Default: 16"
    )
    parser.add_argument(
        "--l1-coeff", type=float, default=3e-4,
        help="L1 sparsity coefficient. Default: 3e-4"
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
    
    # Other arguments
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (cuda/cpu/auto). Default: auto"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory to save checkpoints. Default: checkpoints"
    )
    parser.add_argument(
        "--skip-collection", action="store_true",
        help="Skip data collection (use existing activations.pt)"
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
    print(f"  L1 coefficient: {args.l1_coeff}")
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
            num_texts=args.samples // 10,  # Approximate
            max_samples=args.samples,
            batch_size=16
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
        normalize=True
    )
    
    # Step 3: Create model
    print("\nStep 3: Creating Sparse Autoencoder...")
    print("-" * 70)
    
    sae = create_sae_for_gpt2(
        model_name="gpt2",
        expansion_factor=args.expansion,
        l1_coeff=args.l1_coeff
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
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=5,
        save_every=10
    )
    
    # Plot results
    trainer.plot_training_history(
        save_path=f"{args.checkpoint_dir}/training_history.png"
    )
    
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
        save_path=f"{args.checkpoint_dir}/analysis_report.txt"
    )
    
    # Reconstruction quality
    metrics = analyzer.get_reconstruction_quality(val_data)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Metrics:")
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
