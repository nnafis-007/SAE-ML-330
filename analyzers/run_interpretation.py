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
import re
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from sae_lens import SAE
from pmi_feature_interpretation import (
    FeatureInterpreter,
    InterpretationConfig,
    build_flat_token_ids,
)
from datasets import load_dataset

DEFAULT_MODEL_ID = "gpt2-small-res-jb:blocks.8.hook_resid_pre"


def _parse_model_id(model_id: str) -> tuple[str, str]:
    if not re.match(r"^[^:]+:[^:]+$", model_id):
        raise ValueError(
            "model-id must be in 'release:sae_id' format, "
            "e.g. gpt2-small-res-jb:blocks.8.hook_resid_pre"
        )
    return tuple(model_id.split(":", 1))  # type: ignore[return-value]


def _layer_from_hook(hook_name: str) -> int:
    match = re.search(r"blocks\.(\d+)\.", hook_name)
    return int(match.group(1)) if match else 0


def _collect_texts_from_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    num_texts: int,
) -> List[str]:
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    texts: List[str] = []
    fields = ["text", "sentence", "transcript", "content", "article", "document"]
    for ex in ds:
        value = ""
        for field in fields:
            raw = ex.get(field)
            if isinstance(raw, str) and raw.strip():
                value = raw.strip()
                break
        if value:
            texts.append(value)
        if len(texts) >= num_texts:
            break
    return texts


def _collect_gpt2_hidden_activations(
    texts: List[str],
    tokenizer,
    gpt2_model,
    layer_index: int,
    device: str,
    max_length: int = 128,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.numel() == 0:
                continue
            out = gpt2_model(input_ids=input_ids)
            hidden = out.hidden_states[layer_index + 1][0].cpu()
            rows.append(hidden)

    if not rows:
        raise ValueError("No activations were collected from the selected corpus")
    return torch.cat(rows, dim=0)


# -- Diverse sample corpus for feature interpretation --
# SAMPLE_TEXTS = [
#     "The quick brown fox jumps over the lazy dog.",
#     "Einstein's field equations G_uv + Lambda*g_uv = 8*pi*G*T_uv.",
#     "Heavy rain and strong winds are expected for the coastal regions.",
#     "Voltage is equal to current multiplied by resistance (Ohm's law).",
#     "1, 1, 2, 3, 5, 8, 13, 21 — the Fibonacci sequence.",
#     "def quicksort(arr): return arr if len(arr) <= 1 else quicksort(smaller) + [pivot] + quicksort(larger)",
#     "import torch; x = torch.randn(10, 768); y = torch.relu(x)",
#     "print('Hello World'); for i in range(10): print(i)",
#     "React.useEffect(() => { console.log('mounted'); }, []);",
#     "She whispered, 'I have a secret for you,' as the clock struck midnight.",
#     "Why does the sun rise in the east and set in the west?",
#     "Blueberry pancakes with maple syrup are a classic breakfast choice.",
#     "Under the bridge Downtown, is where I drew some blood.",
#     "Wait! Before you go, make sure you have your passport.",
#     "To be, or not to be, that is the question.",
#     "What is the airspeed velocity of an unladen swallow?",
#     "JSON output format: { 'status': 'success', 'data': [] }",
#     "0.003, 1.45, -0.72, 3.14159 are floating point numbers.",
#     "The fundamental theorem of calculus relates antiderivatives to integrals.",
#     "Blue whales are the largest animals ever known to have lived on Earth.",
#     "Machine learning models can classify images with superhuman accuracy.",
#     "The cat sat on the mat and stared at the bird outside the window.",
#     "Parliament voted to approve the new climate change legislation yesterday.",
#     "Neurons in the brain communicate through electrical and chemical signals.",
#     "The stock market crashed after the unexpected interest rate hike.",
#     "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
#     "The Amazon rainforest produces approximately 20% of the world's oxygen.",
#     "Python is a high-level, interpreted programming language with dynamic typing.",
#     "The President signed the executive order in the Oval Office on Monday.",
#     "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
#     "The new iPhone features a titanium frame and improved battery life.",
#     "Mozart composed his first symphony at the age of eight.",
#     "The Great Wall of China stretches over 13,000 miles across northern China.",
#     "SELECT * FROM users WHERE age > 18 AND country = 'US' ORDER BY name;",
#     "Climate scientists warn that global temperatures could rise 2°C by 2050.",
#     "The patient was admitted to the hospital with chest pain and shortness of breath.",
#     "Ferrari's new hypercar produces 1,000 horsepower from a V12 engine.",
#     "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
#     "Quantum entanglement allows particles to be correlated across vast distances.",
#     "The jury deliberated for three days before reaching a unanimous verdict.",
#     "Apple reported quarterly revenue of $89.5 billion, up 8% year over year.",
#     "The hikers followed the trail through dense forest to the mountain summit.",
#     "TCP/IP protocol suite forms the backbone of modern internet communication.",
#     "The orchestra performed Beethoven's Ninth Symphony to a standing ovation.",
#     "Researchers discovered a new species of deep-sea fish near the Mariana Trench.",
#     "The goalkeeper made a spectacular diving save in the 89th minute.",
#     "Abstract algebra studies algebraic structures such as groups, rings, and fields.",
#     "The CEO announced plans to lay off 10,000 employees amid declining profits.",
#     "Dark matter makes up approximately 27% of the universe's total mass-energy.",
#     "The children laughed and played in the park while their parents watched.",
# ]


SAMPLE_TEXT = [
    "The court split those up into separate sentences and dealt with them in isolation. Even dealing with those in isolation, the first one he took on was that her secret sauce achieves the highest payouts in the industry.",
    "Urban areas continue to be eligible for preparedness funding. To ensure a clear path, we have established a new urban area.",
    "We identify athletes to go to special developmental programs. One of those is the commander leader enrichment seminar, which is an advanced leadership development opportunity.",
    "The question would be whether those are still considered traditional bases for general jurisdiction. They are still discussed even in cases where the courts have said otherwise.",
    "If a certified facility isn't in compliance with these requirements, its authority is to impose remedies under both the Medicare and Medicaid acts. Regulations provide appeals procedures for reviewing determinations.",
    "He was acquitted of six counts; however, he was convicted of count one in the indictment, which was a heroin conspiracy. The other six charges related to a murder.",
    "Begin when you are ready.",
    "Willfully becoming a member of that conspiracy, the defendant being within the jurisdiction of the United States at the time he conspired, and at least one overt act committed by at least one conspirator.",
    "A detention system that is neither unsafe nor degrading for detainees begins with two critical reforms: an examination of whom we are detaining and improved alternatives to detention.",
    "Five hundred miles long from south to north and five hundred miles wide. For hundreds of thousands of years, it was a piece of rock, and then each year what would happen is...",
    "Language with respect to the institution, because shareholders, officers, and directors have certain powers, including voting powers.",
    "I thank my colleagues for their patience. I hope we get on to the business of passing the State Children's Health Insurance bill and have a speedy confirmation for Casey as the next Attorney General.",
    "I'm going to make that same standard for detainees. Being in the military is hard; you don't always get three meals a day, but this is different.",
    "I respectfully disagree with that analysis. To the extent that Terry would come into play as to whether officer safety was an issue, I believe this was not a Terry pat-down.",
    "Do you agree that it has to be non-speculative? There has to be a basis for concluding that Mr. John actually did purchase the cheese.",
    "In August 2009, GSA requested expressions of interest from landowners and agents. We also contacted local economic development and planning groups regarding our search.",
    "No part of the applicable First Amendment analysis requires a comparison to society's evolving standards of indecency, however one might characterize it.",
    "This woman was clearly unhappy with the relationship. She had just given him eight hundred dollars, yet the testimony says she never indicated her unhappiness.",
    "Preserving our natural resources by allocating four billion dollars in new budget authority for the conservation title is extraordinarily important to the future of farming.",
    "That premium is based on a two-prong analysis: risk and result. A fifty million dollar recovery here for the class was twenty percent of the estimated recoverable damages.",
    "The statute says there is no claim for medical monitoring; it is not a remedy. This is simply interpretive of the statute that existed previously.",
    "If we were just looking at the summary judgment question on probable cause, all of those factors you suggested weigh in favor. Why is that?",
    "The Fourth Amendment is meant to protect our privacy from intrusion by officers. To prove consent, the government has to meet their burden of proof.",
    "Motion was properly denied. Thank you.",
    "An invoice for that contingent possibility takes that squarely out of the essential purpose. If we substantially performed, we are entitled to performance regardless of the time."
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
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help=(
            "Pretrained SAE id in release:sae_id format. "
            f"Default: {DEFAULT_MODEL_ID}"
        ),
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
        "--dataset-config", type=str, default="train",
        help="Optional HuggingFace dataset config/split alias. Default: train"
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

    release, sae_id = _parse_model_id(args.model_id)
    print(f"\n[1/5] Loading pretrained SAE from {args.model_id}...")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval().to(device)

    d_model = int(getattr(sae.cfg, "d_in", getattr(sae.cfg, "d_model", sae.W_enc.shape[1])))
    d_hidden = int(getattr(sae.cfg, "d_sae", getattr(sae.cfg, "d_hidden", sae.W_enc.shape[0])))
    setattr(sae, "d_hidden", d_hidden)
    hook_name = str(getattr(getattr(sae.cfg, "metadata", object()), "hook_name", sae_id))
    layer_index = _layer_from_hook(hook_name)
    print(f"  d_model={d_model}, d_hidden={d_hidden}, layer={layer_index}, hook={hook_name}")

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

    if args.dataset:
        print(f"  Loading {args.num_texts} texts from '{args.dataset}'...")
        texts = _collect_texts_from_dataset(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split="train",
            num_texts=args.num_texts,
        )
        activations = _collect_gpt2_hidden_activations(
            texts=texts,
            tokenizer=tokenizer,
            gpt2_model=gpt2_model,
            layer_index=layer_index,
            device=device,
            max_length=128,
        )

        with open(args.corpus_output, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(texts))
    else:
        texts = SAMPLE_TEXT
        activations = _collect_gpt2_hidden_activations(
            texts=texts,
            tokenizer=tokenizer,
            gpt2_model=gpt2_model,
            layer_index=layer_index,
            device=device,
            max_length=128,
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
