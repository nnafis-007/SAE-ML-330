#!/usr/bin/env python3
"""
Capitalisation Invariance Test
================================

Tests whether SAE features encode *meaning* independent of surface-form
capitalisation by checking how much feature overlap exists between the same
word written in different cases (e.g. "cat", "Cat", "CAT", "cAt", "caT").

Hypothesis
----------
If an SAE feature encodes a semantic concept rather than a surface-form
pattern, "cat" and "CAT" should activate heavily overlapping feature sets.
Low overlap would imply the SAE is partly responding to case/typography.

What the script does
--------------------
1.  Loads a trained SAE + GPT-2 (same setup as run_synonym_test.py).
2.  For each test word, generates a set of capitalisation variants
    (all-lower, ALL-UPPER, Title, two alternating patterns).
3.  For each variant, creates sentences in which ONLY that variant appears
    at the target position, then extracts SAE activations exactly there.
4.  Averages activations across sentences, computes top-K active features,
    and calculates pairwise Jaccard + cosine similarity between variants.
5.  Saves a JSON report and prints a human-readable summary.

Usage
-----
    python run_caps_test.py                          # defaults
    python run_caps_test.py --top-k 20
    python run_caps_test.py --checkpoint checkpoints/best_model.pt
    python run_caps_test.py --words cat king happy   # subset
    python run_caps_test.py --output caps_report.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent / "src"))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sae_model import SparseAutoencoder


# ============================================================================
# Test words and sentence templates
# Each word maps to a list of template strings containing exactly one
# placeholder '{}' where the capitalisation variant will be inserted.
# Templates are written so every variant reads naturally (or at worst
# sounds like emphasis/stylistic choice).
# ============================================================================

WORD_TEMPLATES: Dict[str, List[str]] = {
    "cat": [
        "The {} sat on the mat and watched the birds outside.",
        "A {} was sleeping peacefully in the warm garden.",
        "She picked up the {} and held it close.",
        "The {} knocked the glass off the table without warning.",
        "He noticed the {} hiding under the bed.",
        "A stray {} wandered into the kitchen looking for food.",
        "The {} stretched lazily and yawned in the sunlight.",
        "She adopted a {} from the local animal shelter.",
        "The {} stared at its reflection in the mirror.",
        "He gave the {} a gentle scratch behind the ears.",
    ],
    "king": [
        "The {} addressed his people from the palace balcony.",
        "A {} once ruled this land with great wisdom.",
        "The {} signed the decree and handed it to his advisor.",
        "Everyone bowed as the {} entered the throne room.",
        "The {} ordered the gates of the city to be opened.",
        "A powerful {} united the warring factions under one banner.",
        "The {} sat alone in the great hall, deep in thought.",
        "His advisors urged the {} to reconsider his decision.",
        "The {} rode into battle at the head of his army.",
        "A just {} is remembered long after his reign ends.",
    ],
    "happy": [
        "She felt {} when she received the unexpected gift.",
        "The children were {} to spend the day at the beach.",
        "He was {} to learn that the project had been approved.",
        "They were {} to see each other after years apart.",
        "A {} crowd gathered in the square to celebrate.",
        "She looked {} when she read the letter from her friend.",
        "He seemed genuinely {} with the outcome of the match.",
        "Being {} does not require having everything you want.",
        "The team was {} after winning the championship.",
        "She was {} to help whenever anyone asked.",
    ],
    "run": [
        "She decided to {} every morning before breakfast.",
        "He began to {} as soon as he heard the alarm.",
        "The children would {} around the playground for hours.",
        "She watched him {} across the field to catch the ball.",
        "He had to {} to catch the last train of the night.",
        "They started to {} when the rain began to fall.",
        "She would {} along the riverbank to clear her mind.",
        "He could {} faster than anyone else on the team.",
        "The coach told them to {} one more lap around the track.",
        "She felt free when she could {} without thinking.",
    ],
    "fast": [
        "The car was {} enough to win the race by a large margin.",
        "She is a {} learner who picks up new skills quickly.",
        "The train was {} and arrived well ahead of schedule.",
        "He typed a {} reply and moved on to the next task.",
        "A {} river current made crossing dangerous for swimmers.",
        "The {} internet connection made downloads almost instant.",
        "She completed the task with {} and impressive precision.",
        "He is known for making {} decisions under heavy pressure.",
        "A {} heartbeat is perfectly normal during exercise.",
        "The {} response from the team impressed everyone present.",
    ],
    "dark": [
        "The room was {} and silent when she entered.",
        "A {} cloud drifted across the face of the moon.",
        "He walked down the {} corridor without a torch.",
        "The forest grew {} and dense as they ventured further in.",
        "She preferred {} coffee without any sugar or milk.",
        "A {} figure appeared at the far end of the street.",
        "The sky turned {} just before the storm arrived.",
        "He sat alone in the {} room, staring at the ceiling.",
        "The {} water of the lake reflected the winter sky.",
        "A {} shadow fell across the doorway as he waited.",
    ],
    "cold": [
        "The water was {} when she dipped her hand in.",
        "A {} wind swept through the valley that morning.",
        "He wrapped the {} pack around his injured knee.",
        "The {} air made their breath visible in small clouds.",
        "She served the soup {} without thinking to reheat it.",
        "A {} silence fell over the room after the announcement.",
        "He stepped outside into the {} morning air.",
        "The {} metal railing stung her palm when she grabbed it.",
        "They huddled together to stay warm in the {} tent.",
        "A {} draught crept under the door as night fell.",
    ],
    "bright": [
        "The {} light from the screen hurt his eyes in the dark.",
        "She wore a {} yellow jacket that stood out in the crowd.",
        "A {} star appeared low on the horizon at dusk.",
        "The {} sunshine filled the room with warmth.",
        "He was considered a {} student throughout his schooling.",
        "A {} flash of lightning lit up the entire sky.",
        "She painted the wall a {} shade of white.",
        "The {} colours of the mural attracted many passers-by.",
        "A {} idea struck him just as he was about to give up.",
        "The {} glow of the screen was the only light in the room.",
    ],
}

# Capitalisation variant generators
# Each returns a list of (variant_label, variant_string) pairs,
# deduplicated so short words don't produce duplicate forms.

def _cap_variants(word: str) -> List[Tuple[str, str]]:
    """
    Generate up to five capitalisation variants for *word*:
      lower      – all lowercase
      upper      – ALL UPPERCASE
      title      – First letter upper, rest lower
      alt_lo     – aLtErNaTiNg starting with lower
      alt_hi     – AlTeRnAtInG starting with upper

    Duplicates (e.g. for 1-letter words) are silently removed while
    preserving order.
    """
    w = word.lower()

    def alternating(s: str, start_upper: bool) -> str:
        return "".join(
            c.upper() if (i % 2 == 0) == start_upper else c.lower()
            for i, c in enumerate(s)
        )

    candidates = [
        ("lower",  w),
        ("upper",  w.upper()),
        ("title",  w.capitalize()),
        ("alt_lo", alternating(w, start_upper=False)),
        ("alt_hi", alternating(w, start_upper=True)),
    ]

    seen: set[str] = set()
    unique: List[Tuple[str, str]] = []
    for label, variant in candidates:
        if variant not in seen:
            seen.add(variant)
            unique.append((label, variant))
    return unique


# ============================================================================
# Core extraction logic  (parallel to run_synonym_test.py)
# ============================================================================

def find_target_token_positions(
    sentence: str,
    target_word: str,
    tokenizer: GPT2Tokenizer,
    max_length: int = 128,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Tokenise *sentence* and return (input_ids, positions) where *positions*
    is the list of token indices at which *target_word*'s first BPE token
    appears.  Handles the GPT-2 leading-space convention.
    """
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    tokens = input_ids[0].tolist()

    candidates: set[int] = set()
    for prefix in ("", " "):
        ids = tokenizer.encode(prefix + target_word, add_special_tokens=False)
        if ids:
            candidates.add(ids[0])

    positions = [i for i, t in enumerate(tokens) if t in candidates]
    return input_ids, positions


@torch.no_grad()
def get_sae_activations_at_positions(
    sentence: str,
    target_word: str,
    tokenizer: GPT2Tokenizer,
    gpt2: GPT2LMHeadModel,
    sae: SparseAutoencoder,
    layer_index: int,
    device: str,
    max_length: int = 128,
) -> List[torch.Tensor]:
    """
    Return one SAE feature vector (d_hidden,) per occurrence of *target_word*
    in *sentence*.  Returns [] if the word is not found in the BPE sequence.
    """
    input_ids, positions = find_target_token_positions(
        sentence, target_word, tokenizer, max_length
    )
    if not positions:
        return []

    input_ids = input_ids.to(device)
    outputs = gpt2(input_ids=input_ids)
    hidden = outputs.hidden_states[layer_index + 1][0]  # (seq, d_model)

    result = []
    for pos in positions:
        act_vec  = hidden[pos].unsqueeze(0)          # (1, d_model)
        feat_vec = sae.encode(act_vec).squeeze(0)    # (d_hidden,)
        result.append(feat_vec.cpu())
    return result


def collect_variant_profile(
    variant: str,
    templates: List[str],
    tokenizer: GPT2Tokenizer,
    gpt2: GPT2LMHeadModel,
    sae: SparseAutoencoder,
    layer_index: int,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """
    Fill every template with *variant*, extract SAE activations at the
    variant's token position(s), and return the mean activation vector
    together with the number of matched positions.
    """
    all_vecs: List[torch.Tensor] = []
    for template in templates:
        sentence = template.format(variant)
        vecs = get_sae_activations_at_positions(
            sentence, variant, tokenizer, gpt2, sae, layer_index, device
        )
        all_vecs.extend(vecs)

    if not all_vecs:
        return torch.zeros(sae.d_hidden), 0

    stacked = torch.stack(all_vecs, dim=0)   # (N, d_hidden)
    return stacked.mean(dim=0), len(all_vecs)


# ============================================================================
# Analysis helpers
# ============================================================================

def top_k_features(mean_acts: torch.Tensor, k: int) -> List[int]:
    """Return indices of the top-k features by mean activation value."""
    return mean_acts.topk(k).indices.tolist()


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return (a @ b / denom).item()


# ============================================================================
# Per-word analysis
# ============================================================================

def analyse_word(
    base_word: str,
    templates: List[str],
    tokenizer: GPT2Tokenizer,
    gpt2: GPT2LMHeadModel,
    sae: SparseAutoencoder,
    layer_index: int,
    device: str,
    top_k: int,
) -> dict:
    """Run the full caps-invariance analysis for one base word."""

    variants = _cap_variants(base_word)
    print(f"\n  Word: '{base_word}'")
    print(f"  Variants: {[v for _, v in variants]}")

    profiles: Dict[str, torch.Tensor] = {}
    n_positions: Dict[str, int]       = {}
    variant_labels: Dict[str, str]    = {}   # variant_str -> label

    for label, variant in variants:
        mean_acts, n = collect_variant_profile(
            variant, templates, tokenizer, gpt2, sae, layer_index, device
        )
        profiles[variant]       = mean_acts
        n_positions[variant]    = n
        variant_labels[variant] = label
        active = int((mean_acts > 0).sum().item())
        match_note = f"{n} positions" if n > 0 else "NOT FOUND in tokens"
        print(f"    {variant:>12}  ({label:>6}): {match_note}, "
              f"{active} features active")

    # Per-variant top-K feature sets
    top_features: Dict[str, List[int]] = {
        v: top_k_features(p, top_k) for v, p in profiles.items()
    }

    # Pairwise metrics
    variant_strs = [v for _, v in variants]
    pairwise = []
    for va, vb in combinations(variant_strs, 2):
        sa, sb = set(top_features[va]), set(top_features[vb])
        shared = sorted(sa & sb)
        pairwise.append({
            "variant_a":          va,
            "label_a":            variant_labels[va],
            "variant_b":          vb,
            "label_b":            variant_labels[vb],
            "jaccard":            round(jaccard(sa, sb), 4),
            "cosine_sim":         round(cosine_sim(profiles[va], profiles[vb]), 4),
            "shared_feature_count": len(shared),
            "shared_features":    shared,
        })

    # Features shared by ALL variants
    all_sets = [set(top_features[v]) for v in variant_strs]
    universal = sorted(set.intersection(*all_sets))

    # Key pair: lower vs upper (the starkest contrast)
    lower_v = base_word.lower()
    upper_v = base_word.upper()
    key_pair = next(
        (p for p in pairwise
         if p["variant_a"] == lower_v and p["variant_b"] == upper_v),
        None,
    )

    mean_jaccard = (
        sum(p["jaccard"] for p in pairwise) / len(pairwise) if pairwise else 0.0
    )
    mean_cosine = (
        sum(p["cosine_sim"] for p in pairwise) / len(pairwise) if pairwise else 0.0
    )

    print(f"    → Mean Jaccard: {mean_jaccard:.3f}  |  Mean cosine: {mean_cosine:.3f}")
    if key_pair:
        print(f"    → lower↔upper  Jaccard: {key_pair['jaccard']:.3f}  "
              f"cosine: {key_pair['cosine_sim']:.3f}")
    print(f"    → Features shared by ALL {len(variant_strs)} variants: "
          f"{len(universal)}")

    return {
        "word":              base_word,
        "variants":          [{"label": l, "form": v} for l, v in variants],
        "top_k":             top_k,
        "n_positions":       {v: n_positions[v] for v in variant_strs},
        "top_features_per_variant": {v: top_features[v] for v in variant_strs},
        "pairwise":          pairwise,
        "universal_shared_features": universal,
        "mean_jaccard":      round(mean_jaccard, 4),
        "mean_cosine_sim":   round(mean_cosine, 4),
        "lower_vs_upper":    key_pair,
        "interpretation": (
            "CASE-INVARIANT (strong)"   if mean_jaccard > 0.40 else
            "PARTIALLY case-sensitive"  if mean_jaccard > 0.20 else
            "CASE-SENSITIVE (features differ)"
        ),
    }


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether SAE features are stable across capitalisation "
                    "variants of the same word."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt",
        help="Path to trained SAE checkpoint. Default: checkpoints/best_model.pt"
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top features per variant to compare. Default: 30"
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="GPT-2 layer index (overrides checkpoint). Default: from checkpoint."
    )
    parser.add_argument(
        "--device", default="auto",
        help="cuda / cpu / auto. Default: auto."
    )
    parser.add_argument(
        "--output", default="caps_test_report.json",
        help="Output JSON path. Default: caps_test_report.json"
    )
    parser.add_argument(
        "--words", nargs="*", default=None,
        choices=list(WORD_TEMPLATES.keys()),
        help=f"Which words to test. Default: all. "
             f"Available: {list(WORD_TEMPLATES.keys())}"
    )
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    # ── Load SAE ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("CAPITALISATION INVARIANCE TEST")
    print("=" * 70)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"\n[1/3] Loading SAE from {ckpt}...")
    payload  = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp       = payload.get("hyperparameters", {})
    state    = payload["model_state_dict"]
    d_model  = hp.get("d_model",  state["W_enc"].shape[1])
    d_hidden = hp.get("d_hidden", state["W_enc"].shape[0])
    l1_coeff = hp.get("l1_coeff", 3e-4)
    layer_index = args.layer or hp.get("layer_index", 8)

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff)
    sae.load_state_dict(state)
    sae.eval().to(device)
    print(f"  d_model={d_model}, d_hidden={d_hidden}, layer={layer_index}")

    # ── Load GPT-2 ────────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2", output_hidden_states=True
    ).to(device)
    gpt2.eval()
    print(f"  GPT-2 ready on {device}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    words_to_run = args.words or list(WORD_TEMPLATES.keys())
    print(f"\n[3/3] Analysing capitalisation variants (top-{args.top_k} features)...")

    all_results = []
    for word in words_to_run:
        result = analyse_word(
            base_word=word,
            templates=WORD_TEMPLATES[word],
            tokenizer=tokenizer,
            gpt2=gpt2,
            sae=sae,
            layer_index=layer_index,
            device=device,
            top_k=args.top_k,
        )
        all_results.append(result)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "settings": {
            "checkpoint":  str(ckpt),
            "top_k":       args.top_k,
            "layer_index": layer_index,
            "d_model":     d_model,
            "d_hidden":    d_hidden,
            "device":      device,
        },
        "words": all_results,
        "overall_mean_jaccard": round(
            sum(r["mean_jaccard"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0,
        "overall_mean_cosine": round(
            sum(r["mean_cosine_sim"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0,
    }

    out_path = Path(args.output)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    # ── Human-readable summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = (f"\n{'Word':<10}  {'Variants':<34}  {'Jaccard':>7}  "
              f"{'Cosine':>7}  {'All-shr':>7}  {'low↔UP Jac':>10}  Signal")
    print(header)
    print("-" * 90)
    for r in all_results:
        forms_str = "/".join(v["form"] for v in r["variants"])
        lu = r["lower_vs_upper"]
        lu_jac = f"{lu['jaccard']:.3f}" if lu else "  n/a"
        print(
            f"{r['word']:<10}  {forms_str:<34}  "
            f"{r['mean_jaccard']:>7.3f}  {r['mean_cosine_sim']:>7.3f}  "
            f"{len(r['universal_shared_features']):>7}  "
            f"{lu_jac:>10}  {r['interpretation']}"
        )

    print("\n" + "-" * 90)
    print(f"Overall mean Jaccard : {report['overall_mean_jaccard']:.3f}")
    print(f"Overall mean cosine  : {report['overall_mean_cosine']:.3f}")
    print(
        "\nJaccard guide:  > 0.40 → features are largely case-invariant\n"
        "                0.20–0.40 → partial case sensitivity\n"
        "                < 0.20 → the SAE treats different cases as unrelated"
    )
    print(f"\nFull report saved to: {args.output}")

    # ── Pairwise detail ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PAIRWISE DETAIL (lower ↔ upper highlighted)")
    print("=" * 70)
    for r in all_results:
        print(f"\nWord: '{r['word']}'")
        for pw in r["pairwise"]:
            marker = " ★" if (pw["label_a"] == "lower" and
                              pw["label_b"] == "upper") else ""
            shared_preview = pw["shared_features"][:8]
            more = (f" …+{pw['shared_feature_count'] - 8}"
                    if pw["shared_feature_count"] > 8 else "")
            print(
                f"  {pw['variant_a']:>12} ↔ {pw['variant_b']:<12}"
                f"  Jaccard={pw['jaccard']:.3f}  cos={pw['cosine_sim']:.3f}"
                f"  shared={pw['shared_feature_count']}"
                f"  feats={shared_preview}{more}{marker}"
            )
        if r["universal_shared_features"]:
            print(f"  → All-variant shared: "
                  f"{r['universal_shared_features'][:10]}")
        else:
            print("  → No features shared by ALL variants")


if __name__ == "__main__":
    main()
