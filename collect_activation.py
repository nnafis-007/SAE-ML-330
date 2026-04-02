#!/usr/bin/env python3
"""
Data Collection Script for SAE Training

This script collects activations from GPT-2 hidden layers for training
Sparse Autoencoders (SAE or FC-SAE).

Features:
- Supports multiple GPT-2 model sizes (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Collects from any HuggingFace text dataset
- Saves activations and optional text corpus for interpretation
- Configurable normalization and preprocessing

Usage Examples:
    # Basic usage - collect 50k samples from layer 8
    python collect_activation.py --layer 8 --samples 50000

    # Collect from a specific dataset with more texts
    python collect_activation.py --layer 6 --samples 100000 --dataset wikitext --dataset-config wikitext-103-v1

    # Save both activations and corpus for interpretation
    python collect_activation.py --layer 8 --samples 50000 --save-corpus --corpus-output corpus.txt

    # Use a different GPT-2 model
    python collect_activation.py --model gpt2-medium --layer 12 --samples 50000
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from data_collection import GPT2ActivationCollector, prepare_training_data


def main():
    parser = argparse.ArgumentParser(
        description="Collect GPT-2 hidden layer activations for SAE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic collection
  python collect_activation.py --layer 8 --samples 50000

  # With specific output path
  python collect_activation.py --layer 8 --samples 100000 --output my_activations.pt

  # Collect and prepare training data
  python collect_activation.py --layer 8 --samples 50000 --prepare-data --normalize standardize

  # Save corpus for interpretation
  python collect_activation.py --layer 8 --samples 50000 --save-corpus --corpus-output corpus.txt
        """
    )

    # Model arguments
    parser.add_argument(
        "--model", type=str, default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model variant. Default: gpt2"
    )
    parser.add_argument(
        "--layer", type=int, default=8,
        help="Layer index to extract activations from (0-indexed). "
             "GPT-2: 0-11, GPT-2-medium: 0-23, GPT-2-large: 0-35, GPT-2-xl: 0-47. "
             "Default: 8"
    )

    # Data collection arguments
    parser.add_argument(
        "--samples", type=int, default=50000,
        help="Maximum number of activation samples to collect. Default: 50000"
    )
    parser.add_argument(
        "--num-texts", type=int, default=None,
        help="Number of texts to load from dataset. If not specified, "
             "defaults to samples // 10."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for GPT-2 forward pass. Reduce if OOM. Default: 16"
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Maximum sequence length per text. Default: 128"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="openwebtext",
        help="HuggingFace dataset name. Default: openwebtext"
    )
    parser.add_argument(
        "--dataset-config", type=str, default=None,
        help="Dataset configuration name (e.g., 'wikitext-103-v1' for wikitext)"
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train",
        help="Dataset split to use. Default: train"
    )
    parser.add_argument(
        "--text-field", type=str, default=None,
        help="Name of the text field in dataset. Auto-detected if not specified."
    )
    parser.add_argument(
        "--shuffle-buffer-size", type=int, default=10000,
        help="Shuffle buffer size for streaming datasets. 0 disables shuffling. Default: 10000"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset shuffling. Default: 42"
    )
    parser.add_argument(
        "--hf-cache-dir", type=str, default=None,
        help="Optional Hugging Face datasets cache directory (falls back to HF_DATASETS_CACHE env var)."
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o", type=str, default="activations.pt",
        help="Output path for activations tensor. Default: activations.pt"
    )
    parser.add_argument(
        "--save-corpus", action="store_true",
        help="Also save the text corpus used for collection."
    )
    parser.add_argument(
        "--corpus-output", type=str, default="corpus.txt",
        help="Output path for text corpus. Default: corpus.txt"
    )

    # Preprocessing arguments
    parser.add_argument(
        "--prepare-data", action="store_true",
        help="Prepare data for training (split and normalize)."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9,
        help="Ratio of data for training (rest for validation). Default: 0.9"
    )
    parser.add_argument(
        "--normalize", type=str, default="standardize",
        choices=["standardize", "center", "none"],
        help="Normalization mode. Default: standardize"
    )
    parser.add_argument(
        "--std-floor", type=float, default=1e-3,
        help="Minimum std for standardization. Default: 1e-3"
    )
    parser.add_argument(
        "--prepared-output", type=str, default="prepared_data.pt",
        help="Output path for prepared data. Default: prepared_data.pt"
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use. Default: auto (cuda if available)"
    )

    # Verbosity
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce output verbosity."
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if not args.quiet:
        print("=" * 70)
        print("GPT-2 ACTIVATION COLLECTION")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Model: {args.model}")
        print(f"  Layer: {args.layer}")
        print(f"  Max samples: {args.samples:,}")
        print(f"  Dataset: {args.dataset}")
        if args.dataset_config:
            print(f"  Dataset config: {args.dataset_config}")
        print(f"  Device: {device}")
        print(f"  Output: {args.output}")
        print()

    # Initialize collector
    collector = GPT2ActivationCollector(
        model_name=args.model,
        layer_index=args.layer,
        device=device
    )

    # Determine number of texts
    num_texts = args.num_texts if args.num_texts is not None else args.samples // 10

    # Collect activations
    if args.save_corpus:
        if not args.quiet:
            print(f"\nCollecting activations with corpus...")

        activations, texts = collector.collect_from_dataset_with_texts(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.dataset_split,
            num_texts=num_texts,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            text_field=args.text_field,
            cache_dir=args.hf_cache_dir,
            corpus_output=args.corpus_output,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_samples=args.samples,
        )

        if not args.quiet:
            print(f"Corpus saved to: {args.corpus_output}")
    else:
        if not args.quiet:
            print(f"\nCollecting activations...")

        activations = collector.collect_from_dataset(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.dataset_split,
            num_texts=num_texts,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            text_field=args.text_field,
            cache_dir=args.hf_cache_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_samples=args.samples,
        )

    # Save raw activations
    torch.save(activations, args.output)
    if not args.quiet:
        print(f"\nActivations saved to: {args.output}")
        print(f"Shape: {activations.shape}")
        print(f"Size: {activations.numel() * 4 / 1024 / 1024:.2f} MB")

    # Optionally prepare training data
    if args.prepare_data:
        if not args.quiet:
            print(f"\nPreparing training data...")

        train_data, val_data, stats = prepare_training_data(
            activations,
            train_ratio=args.train_ratio,
            normalize=args.normalize != "none",
            normalize_mode=args.normalize,
            std_floor=args.std_floor,
        )

        prepared = {
            "train": train_data,
            "val": val_data,
            "stats": stats,
            "model": args.model,
            "layer": args.layer,
            "hidden_size": collector.hidden_size,
        }

        torch.save(prepared, args.prepared_output)

        if not args.quiet:
            print(f"Prepared data saved to: {args.prepared_output}")
            print(f"  Train samples: {train_data.shape[0]:,}")
            print(f"  Val samples: {val_data.shape[0]:,}")

    if not args.quiet:
        print("\nCollection complete!")


if __name__ == "__main__":
    main()
