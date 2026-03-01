#!/usr/bin/env python3
"""
Feature Interpretation Runner
==============================

Runs the full statistical + semantic interpretation pipeline on
SAE features, producing human-readable reports.

This script:
  1. Loads a trained SAE and GPT-2
  2. Collects activations from a text corpus
  3. Runs PMI + Chi-squared statistical analysis
  4. Runs logit-lens decoder-weight projection
  5. Runs POS-tag linguistic analysis
  6. Produces a combined report per feature

Usage:
    # Interpret top-10 most active features using sample corpus
    python run_interpretation.py

    # Specify features manually
    python run_interpretation.py --features 7949 6629 1938

    # Use a larger corpus from HuggingFace
    python run_interpretation.py --dataset openwebtext --num-texts 500

    # Skip POS tagging (faster, no nltk dependency)
    python run_interpretation.py --no-pos

    # Use pre-existing activations.pt (must match the texts!)
    python run_interpretation.py --use-saved-activations --texts-file corpus.txt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from sae_model import SparseAutoencoder
from feature_interpretation import (
    FeatureInterpreter,
    InterpretationConfig,
    build_flat_token_ids,
)


# -- Diverse sample corpus for feature interpretation --
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Einstein's field equations G_uv + Lambda*g_uv = 8*pi*G*T_uv.",
    "Heavy rain and strong winds are expected for the coastal regions.",
    "Voltage is equal to current multiplied by resistance (Ohm's law).",
    "1, 1, 2, 3, 5, 8, 13, 21 — the Fibonacci sequence.",
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort(smaller) + [pivot] + quicksort(larger)",
    "import torch; x = torch.randn(10, 768); y = torch.relu(x)",
    "print('Hello World'); for i in range(10): print(i)",
    "React.useEffect(() => { console.log('mounted'); }, []);",
    "She whispered, 'I have a secret for you,' as the clock struck midnight.",
    "Why does the sun rise in the east and set in the west?",
    "Blueberry pancakes with maple syrup are a classic breakfast choice.",
    "Under the bridge Downtown, is where I drew some blood.",
    "Wait! Before you go, make sure you have your passport.",
    "To be, or not to be, that is the question.",
    "What is the airspeed velocity of an unladen swallow?",
    "JSON output format: { 'status': 'success', 'data': [] }",
    "0.003, 1.45, -0.72, 3.14159 are floating point numbers.",
    "The fundamental theorem of calculus relates antiderivatives to integrals.",
    "Blue whales are the largest animals ever known to have lived on Earth.",
    "Machine learning models can classify images with superhuman accuracy.",
    "The cat sat on the mat and stared at the bird outside the window.",
    "Parliament voted to approve the new climate change legislation yesterday.",
    "Neurons in the brain communicate through electrical and chemical signals.",
    "The stock market crashed after the unexpected interest rate hike.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen.",
    "Python is a high-level, interpreted programming language with dynamic typing.",
    "The President signed the executive order in the Oval Office on Monday.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The new iPhone features a titanium frame and improved battery life.",
    "Mozart composed his first symphony at the age of eight.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "SELECT * FROM users WHERE age > 18 AND country = 'US' ORDER BY name;",
    "Climate scientists warn that global temperatures could rise 2°C by 2050.",
    "The patient was admitted to the hospital with chest pain and shortness of breath.",
    "Ferrari's new hypercar produces 1,000 horsepower from a V12 engine.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Quantum entanglement allows particles to be correlated across vast distances.",
    "The jury deliberated for three days before reaching a unanimous verdict.",
    "Apple reported quarterly revenue of $89.5 billion, up 8% year over year.",
    "The hikers followed the trail through dense forest to the mountain summit.",
    "TCP/IP protocol suite forms the backbone of modern internet communication.",
    "The orchestra performed Beethoven's Ninth Symphony to a standing ovation.",
    "Researchers discovered a new species of deep-sea fish near the Mariana Trench.",
    "The goalkeeper made a spectacular diving save in the 89th minute.",
    "Abstract algebra studies algebraic structures such as groups, rings, and fields.",
    "The CEO announced plans to lay off 10,000 employees amid declining profits.",
    "Dark matter makes up approximately 27% of the universe's total mass-energy.",
    "The children laughed and played in the park while their parents watched.",
]


def main():
    parser = argparse.ArgumentParser(
        description="Interpret SAE features using statistical + semantic analysis"
    )

    parser.add_argument(
        "--features", type=int, nargs="*", default=None,
        help="Feature indices to interpret. Default: auto-select top-N by mean activation."
    )
    parser.add_argument(
        "--num-features", type=int, default=5,
        help="Number of top features to auto-select if --features is not given. Default: 5"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to SAE checkpoint. Default: checkpoints/best_model.pt"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-K activations to analyze per feature. Default: 50"
    )
    parser.add_argument(
        "--significance", type=float, default=0.01,
        help="p-value threshold for statistical significance. Default: 0.01"
    )
    parser.add_argument(
        "--no-pos", action="store_true",
        help="Skip POS-tag analysis (faster, no nltk needed)."
    )
    parser.add_argument(
        "--logit-lens-k", type=int, default=15,
        help="Top-K tokens to report from logit lens. Default: 15"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="HuggingFace dataset for corpus (e.g., 'openwebtext'). Default: use built-in samples."
    )
    parser.add_argument(
        "--num-texts", type=int, default=200,
        help="Number of texts to load from dataset. Default: 200"
    )
    parser.add_argument(
        "--output", type=str, default="interpretation_reports.json",
        help="Output JSON path. Default: interpretation_reports.json"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (cuda/cpu/auto). Default: auto"
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="GPT-2 layer (overrides checkpoint metadata). Default: from checkpoint."
    )
    parser.add_argument(
        "--corpus-output", type=str, default="corpus_texts.txt",
        help="File to write the collected HuggingFace corpus texts to (one per entry). "
             "Only used when --dataset is set. Default: corpus_texts.txt"
    )
    parser.add_argument(
        "--min-activation-rate", type=float, default=0.001,
        help="Exclude features that fire on fewer than this fraction of tokens (dead/rare). "
             "Default: 0.001 (0.1%%)"
    )
    parser.add_argument(
        "--max-activation-rate", type=float, default=0.20,
        help="Exclude features that fire on more than this fraction of tokens "
             "(background/bias features with no discriminative signal). "
             "Default: 0.20 (20%%). Features above this threshold produce PMI≈0 "
             "and uninterpretable logit-lens results."
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ================================================================== #
    # 1.  Load SAE
    # ================================================================== #
    print("=" * 70)
    print("SAE FEATURE INTERPRETATION PIPELINE")
    print("=" * 70)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Train an SAE first with: python run_sae.py")
        sys.exit(1)

    print(f"\n[1/5] Loading SAE from {ckpt_path}...")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = payload.get("hyperparameters", {})
    state = payload["model_state_dict"]
    d_model = hp.get("d_model", state["W_enc"].shape[1])
    d_hidden = hp.get("d_hidden", state["W_enc"].shape[0])
    l1_coeff = hp.get("l1_coeff", 3e-4)
    layer_index = args.layer or hp.get("layer_index", 8)

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff)
    sae.load_state_dict(state)
    sae.eval()
    print(f"  d_model={d_model}, d_hidden={d_hidden}, layer={layer_index}")

    # ================================================================== #
    # 2.  Load GPT-2 (for logit lens + activation collection)
    # ================================================================== #
    print(f"\n[2/5] Loading GPT-2 model and tokenizer...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(
        "gpt2", output_hidden_states=True
    ).to(device)
    gpt2_model.eval()
    print(f"  GPT-2 loaded on {device}")

    # ================================================================== #
    # 3.  Prepare corpus + collect activations
    # ================================================================== #
    print(f"\n[3/5] Preparing corpus and collecting activations...")

    from data_collection import GPT2ActivationCollector

    collector = GPT2ActivationCollector(
        model_name="gpt2", layer_index=layer_index, device=device
    )

    if args.dataset:
        # collect_from_dataset_with_texts returns BOTH the activations AND
        # the raw text strings, so the same corpus drives:
        #   • feature-selection statistics (step 4)
        #   • token-level interpretation (PMI, logit-lens, POS) (step 5)
        # The texts are also written to --corpus-output for offline inspection.
        print(f"  Loading {args.num_texts} texts from '{args.dataset}'...")
        activations, texts = collector.collect_from_dataset_with_texts(
            dataset_name=args.dataset,
            num_texts=args.num_texts,
            max_samples=args.num_texts * 50,
            batch_size=8,
            max_length=128,
            corpus_output=args.corpus_output,
        )
    else:
        texts = SAMPLE_TEXTS
        activations = collector.collect_activations(
            texts=texts, batch_size=8, max_length=128
        )

    # dataset_activations == activations in both branches (unified corpus)
    dataset_activations = activations

    # Build flat token ID mapping
    token_ids_flat, token_doc_ids, all_token_ids = build_flat_token_ids(
        texts, tokenizer, max_length=128
    )

    # Verify alignment
    if len(token_ids_flat) != activations.shape[0]:
        print(f"  WARNING: Token map ({len(token_ids_flat)}) != activations "
              f"({activations.shape[0]}). Truncating to min.")
        min_len = min(len(token_ids_flat), activations.shape[0])
        token_ids_flat = token_ids_flat[:min_len]
        token_doc_ids = token_doc_ids[:min_len]
        activations = activations[:min_len]

    corpus_label = f"'{args.dataset}' ({len(texts)} texts)" if args.dataset else f"built-in sample ({len(texts)} texts)"
    print(f"  Corpus: {corpus_label}, {activations.shape[0]} token activations")
    if args.dataset:
        print(f"  Corpus saved to: {args.corpus_output}")

    # ================================================================== #
    # 4.  Select features to interpret
    # ================================================================== #
    print(f"\n[4/5] Selecting features...")

    if args.features:
        feature_indices = args.features
        # Still warn if manually chosen features violate the rate bounds
        if args.min_activation_rate > 0.0 or args.max_activation_rate < 1.0:
            print(f"  Note: --min/max-activation-rate filters are applied only "
                  f"during auto-selection, not for manually specified --features.")
    else:
        # Compute per-feature mean activation AND activation rate.
        # Use dataset_activations (larger corpus when --dataset is given,
        # same as activations otherwise) so feature selection benefits from
        # the full openwebtext corpus.
        mean_acts = torch.zeros(sae.d_hidden)
        act_counts = torch.zeros(sae.d_hidden)   # number of positions where feature > 0
        N_total = dataset_activations.shape[0]
        with torch.no_grad():
            for start in range(0, N_total, 512):
                end = min(start + 512, N_total)
                batch = dataset_activations[start:end].to(device)
                enc = sae.encode(batch)
                mean_acts += enc.sum(dim=0).cpu()
                act_counts += (enc > 0).float().sum(dim=0).cpu()
        mean_acts /= N_total
        act_rates = act_counts / N_total  # fraction of tokens where each feature fires

        # Filter by activation rate window
        lo, hi = args.min_activation_rate, args.max_activation_rate
        valid_mask = (act_rates >= lo) & (act_rates <= hi)
        n_valid = int(valid_mask.sum().item())
        print(f"  Activation rate filter: [{lo:.1%}, {hi:.1%}]  "
              f"→ {n_valid} / {sae.d_hidden} features pass")
        if n_valid == 0:
            print("  WARNING: No features pass the activation rate filter. "
                  "Widening the range or using --features manually.")
            valid_mask = torch.ones(sae.d_hidden, dtype=torch.bool)

        # Among valid features, pick top-N by mean activation
        mean_acts_filtered = mean_acts.clone()
        mean_acts_filtered[~valid_mask] = -1.0
        feature_indices = mean_acts_filtered.argsort(descending=True)[
            :args.num_features
        ].tolist()
        # Report the rate for chosen features
        for fi in feature_indices:
            print(f"    feature {fi}: activation_rate={act_rates[fi].item():.2%}, "
                  f"mean_act={mean_acts[fi].item():.4f}")

    print(f"  Features to interpret: {feature_indices}")

    # ================================================================== #
    # 5.  Run interpretation pipeline
    # ================================================================== #
    print(f"\n[5/5] Running interpretation pipeline...")

    cfg = InterpretationConfig(
        top_k=args.top_k,
        significance_level=args.significance,
        logit_lens_top_k=args.logit_lens_k,
        pos_tag=not args.no_pos,
    )

    interp = FeatureInterpreter(
        sae=sae,
        gpt2_model=gpt2_model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
    )

    reports = interp.interpret_features(
        feature_indices=feature_indices,
        activations=activations,
        token_ids_flat=token_ids_flat,
        token_doc_ids=token_doc_ids,
        all_token_ids=all_token_ids,
        save_path=args.output,
    )

    # ================================================================== #
    # Print results
    # ================================================================== #
    print("\n" + "=" * 70)
    print("INTERPRETATION RESULTS")
    print("=" * 70)

    for feat_idx in sorted(reports.keys()):
        FeatureInterpreter.print_report(reports[feat_idx])

    FeatureInterpreter.print_summary(reports)

    print(f"\nFull reports saved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
