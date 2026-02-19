#!/usr/bin/env python3
"""
Synonym Feature-Overlap Test
=============================

Tests whether SAE features are sensitive to *meaning* rather than
*surface form* by checking if synonyms activate the same features.

Hypothesis:
  If an SAE feature truly encodes a semantic concept, it should fire
  on all words that share that concept — i.e. synonyms should produce
  overlapping top-K active features.

What the script does
--------------------
1.  Loads a trained SAE + GPT-2 (same setup as run_interpretation.py).
2.  For each semantic cluster (a group of synonyms), generates a set of
    sentences in which ONLY that word appears at the target position.
3.  Tokenises each sentence and identifies the exact GPT-2 token positions
    that correspond to the target word.
4.  Extracts SAE feature activations at those specific positions (NOT the
    whole sentence) and averages them across sentences for that word.
5.  For each word, records the top-K most active features.
6.  Computes pairwise Jaccard similarity between synonym feature sets:
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    Higher Jaccard → more shared features → more "synonym-aware" features.
7.  Also records the raw mean activation value of every shared feature, so
    you can inspect what the shared features look like in
    interpretation_reports.json.
8.  Saves a JSON report + prints a human-readable summary.

Usage
-----
    python run_synonym_test.py                         # defaults
    python run_synonym_test.py --top-k 20              # top-20 features per word
    python run_synonym_test.py --checkpoint checkpoints/best_model.pt
    python run_synonym_test.py --output synonym_report.json
    python run_synonym_test.py --clusters happy angry big fast  # subset of clusters
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
# Synonym clusters + template sentences
# Each entry: "word" -> list of sentences.
# The word must appear VERBATIM in every sentence so we can find it.
# Sentences are written so the word is unambiguous in meaning.
# ============================================================================

SYNONYM_CLUSTERS: Dict[str, Dict[str, List[str]]] = {

    # ── emotional state: happiness ──────────────────────────────────────────
    "happy": {
        "happy": [
            "She felt happy when she heard the good news.",
            "The children were happy to see the puppies.",
            "He smiled, happy that the long project was finally over.",
            "I am happy to help you with that task.",
            "They left the party feeling happy and energised.",
            "A happy crowd gathered to celebrate the victory.",
            "She was happy with the results of the experiment.",
            "The dog wagged its tail, looking happy and playful.",
            "He was happy to share his findings with the team.",
            "Being happy does not require wealth or fame.",
        ],
        "joyful": [
            "She felt joyful when she heard the good news.",
            "The children were joyful to see the puppies.",
            "He smiled, joyful that the long project was finally over.",
            "A joyful crowd gathered to celebrate the victory.",
            "She was joyful with the results of the experiment.",
            "The music made everyone feeling joyful and light.",
            "He was joyful to share his findings with the team.",
            "Being joyful does not require wealth or fame.",
            "They danced in a joyful celebration after winning.",
            "Her joyful laughter filled the room.",
        ],
        "elated": [
            "She felt elated when she heard the good news.",
            "The children were elated to see the puppies.",
            "He smiled, elated that the long project was finally over.",
            "An elated crowd gathered to celebrate the victory.",
            "She was elated with the results of the experiment.",
            "He was elated to share his findings with the team.",
            "Being elated after such a win felt completely natural.",
            "They left the ceremony feeling elated and proud.",
            "Her elated expression told everyone the news was good.",
            "The team was elated after breaking the world record.",
        ],
        "content": [
            "She felt content when she heard the good news.",
            "The children were content to play in the garden.",
            "He smiled, content that the long project was finally over.",
            "She was content with the results of the experiment.",
            "He was content to share his findings with the team.",
            "Being content with little is a form of wisdom.",
            "They sat in content silence, watching the sunset.",
            "A content smile crossed her face as she read the letter.",
            "He felt content knowing his family was safe.",
            "Living simply kept her feeling content and calm.",
        ],
        "pleased": [
            "She felt pleased when she heard the good news.",
            "The children were pleased to see the puppies.",
            "He smiled, pleased that the long project was finally over.",
            "She was pleased with the results of the experiment.",
            "He was pleased to share his findings with the team.",
            "The manager was pleased with the team's performance.",
            "A pleased expression appeared on his face.",
            "She was pleased to accept the award on behalf of the group.",
            "I am pleased to announce our latest achievement.",
            "They were pleased by the warm reception they received.",
        ],
    },

    # ── size: large ──────────────────────────────────────────────────────────
    "large": {
        "large": [
            "A large crowd gathered outside the stadium.",
            "The company reported a large increase in revenue.",
            "She carried a large bag filled with groceries.",
            "There is a large lake behind the mountain range.",
            "He noticed a large scratch on the car door.",
            "The library has a large collection of rare books.",
            "A large oak tree stood in the centre of the park.",
            "The elephant is a large and intelligent animal.",
            "They rented a large apartment in the city centre.",
            "A large portion of the budget was spent on research.",
        ],
        "big": [
            "A big crowd gathered outside the stadium.",
            "The company reported a big increase in revenue.",
            "She carried a big bag filled with groceries.",
            "There is a big lake behind the mountain range.",
            "He noticed a big scratch on the car door.",
            "A big oak tree stood in the centre of the park.",
            "The elephant is a big and intelligent animal.",
            "They rented a big apartment in the city centre.",
            "A big portion of the budget was spent on research.",
            "She had a big smile on her face all day.",
        ],
        "huge": [
            "A huge crowd gathered outside the stadium.",
            "The company reported a huge increase in revenue.",
            "She carried a huge bag filled with groceries.",
            "There is a huge lake behind the mountain range.",
            "A huge oak tree stood in the centre of the park.",
            "The elephant is a huge and intelligent animal.",
            "A huge portion of the budget was spent on research.",
            "They discovered a huge cave system underground.",
            "The explosion left a huge crater in the ground.",
            "It was a huge relief when the surgery went well.",
        ],
        "enormous": [
            "An enormous crowd gathered outside the stadium.",
            "The company reported an enormous increase in revenue.",
            "She carried an enormous bag filled with groceries.",
            "There is an enormous lake behind the mountain range.",
            "An enormous oak tree stood in the centre of the park.",
            "An enormous portion of the budget was spent on research.",
            "They discovered an enormous cave system underground.",
            "The explosion left an enormous crater in the ground.",
            "The task required an enormous amount of patience.",
            "An enormous wave crashed against the cliffs.",
        ],
    },

    # ── speed: fast ──────────────────────────────────────────────────────────
    "fast": {
        "fast": [
            "The car was fast enough to win the race easily.",
            "She is a fast learner and picks up new skills quickly.",
            "He typed a fast reply and sent it immediately.",
            "The train was fast and reached the city in an hour.",
            "A fast river current made swimming dangerous.",
            "He is known for his fast decision making under pressure.",
            "The fast food outlet was packed at lunchtime.",
            "She ran a fast lap around the track.",
            "The fast internet connection made downloads instant.",
            "A fast heartbeat is normal during exercise.",
        ],
        "quick": [
            "The car was quick enough to win the race easily.",
            "She is a quick learner and picks up new skills rapidly.",
            "He sent a quick reply and moved on to the next task.",
            "A quick glance at the clock told her she was late.",
            "He is known for his quick decision making under pressure.",
            "She gave a quick nod to signal her agreement.",
            "A quick search online answered the question immediately.",
            "The doctor recommended a quick walk every morning.",
            "He finished the exam in quick time.",
            "A quick phone call resolved the misunderstanding.",
        ],
        "rapid": [
            "The car achieved rapid acceleration on the highway.",
            "She is known for her rapid progress in the field.",
            "The rapid current made crossing the river risky.",
            "He is praised for his rapid decision making under pressure.",
            "Rapid urbanisation changed the landscape significantly.",
            "The patient showed rapid improvement after treatment.",
            "Rapid changes in technology disrupted the industry.",
            "A rapid heartbeat was measured during the stress test.",
            "The rapid spread of information reshaped public opinion.",
            "Rapid cooling of the metal reinforced its structure.",
        ],
        "swift": [
            "The car was swift enough to win the race easily.",
            "She made a swift decision and acted immediately.",
            "A swift current ran through the narrow gorge.",
            "He is praised for his swift response to the crisis.",
            "The swift movement of the hawk startled the pigeons.",
            "A swift glance at the scoreboard told the whole story.",
            "The government took swift action to contain the outbreak.",
            "She completed the task with swift precision.",
            "A swift breeze cooled the air on the warm afternoon.",
            "The swift runner crossed the finish line first.",
        ],
    },

    # ── negative emotion: angry ──────────────────────────────────────────────
    "angry": {
        "angry": [
            "She felt angry when her request was ignored.",
            "The crowd became angry after the announcement was made.",
            "He gave an angry reply and slammed the door.",
            "The angry customer demanded to speak to the manager.",
            "She looked angry when she saw the mess in the kitchen.",
            "An angry storm brewed on the horizon.",
            "He was angry that no one had informed him in advance.",
            "The angry protesters marched through the city streets.",
            "Her voice was calm, but her eyes were angry.",
            "Being angry is natural, but acting on it impulsively is not.",
        ],
        "furious": [
            "She felt furious when her request was ignored.",
            "The crowd became furious after the announcement was made.",
            "He gave a furious reply and slammed the door.",
            "The furious customer demanded to speak to the manager.",
            "She looked furious when she saw the mess in the kitchen.",
            "He was furious that no one had informed him in advance.",
            "The furious protesters marched through the city streets.",
            "Her voice was steady, but she was visibly furious.",
            "Being furious is natural when you have been treated unfairly.",
            "A furious argument broke out between the two sides.",
        ],
        "irate": [
            "She felt irate when her request was ignored.",
            "The irate customer demanded to speak to the manager.",
            "He was irate that no one had informed him in advance.",
            "The irate crowd refused to leave the square.",
            "An irate letter was sent directly to the board of directors.",
            "She sounded irate on the phone when she heard the news.",
            "The staff had to calm the irate passengers at the gate.",
            "He was visibly irate after the meeting ended without a resolution.",
            "An irate response from the public followed the controversial ruling.",
            "The irate driver honked and shouted at the traffic.",
        ],
        "livid": [
            "She felt livid when her request was ignored.",
            "The crowd became livid after the announcement was made.",
            "He was livid that no one had informed him in advance.",
            "The manager was livid after reading the report.",
            "She looked livid when she saw the mess in the kitchen.",
            "He went livid when he found out what had happened.",
            "Being livid about the decision, she called an emergency meeting.",
            "A livid reaction from the public followed the news.",
            "He was absolutely livid and demanded an immediate apology.",
            "She was livid, pacing back and forth across the room.",
        ],
    },
}


# ============================================================================
# Core extraction logic
# ============================================================================

def find_target_token_positions(
    sentence: str,
    target_word: str,
    tokenizer: GPT2Tokenizer,
    max_length: int = 128,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Tokenise *sentence* and return:
      - input_ids  : (1, seq_len) tensor
      - positions  : list of token indices where *target_word* starts

    GPT-2 uses byte-pair encoding so a single word can become multiple
    tokens.  We find the first token of the word's BPE span by searching
    for any tokenisation of target_word (with/without a leading space,
    which GPT-2 adds for mid-sentence words).
    """
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]  # (1, L)
    tokens = input_ids[0].tolist()

    # GPT-2 prepends Ġ (space) to mid-sentence tokens.
    # We try both forms and collect the matching first-token ID.
    candidates = set()
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
    Returns a list of SAE feature-activation vectors (d_hidden,), one per
    occurrence of *target_word* in *sentence*.  Returns empty list if the
    word is not found in the tokenisation.
    """
    input_ids, positions = find_target_token_positions(
        sentence, target_word, tokenizer, max_length
    )
    if not positions:
        return []

    input_ids = input_ids.to(device)
    outputs = gpt2(input_ids=input_ids)
    # hidden_states: tuple of (L+1) tensors, each (1, seq, d_model)
    hidden = outputs.hidden_states[layer_index + 1][0]  # (seq, d_model)

    result = []
    for pos in positions:
        act_vec = hidden[pos].unsqueeze(0)          # (1, d_model)
        feat_vec = sae.encode(act_vec).squeeze(0)   # (d_hidden,)
        result.append(feat_vec.cpu())
    return result


def collect_word_feature_profile(
    word: str,
    sentences: List[str],
    tokenizer: GPT2Tokenizer,
    gpt2: GPT2LMHeadModel,
    sae: SparseAutoencoder,
    layer_index: int,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """
    Average SAE feature activations across all positions in all sentences
    where *word* appears.

    Returns:
      mean_acts   : (d_hidden,) mean feature activation vector
      n_positions : total number of token positions found
    """
    all_vecs = []
    for sentence in sentences:
        vecs = get_sae_activations_at_positions(
            sentence, word, tokenizer, gpt2, sae, layer_index, device
        )
        all_vecs.extend(vecs)

    if not all_vecs:
        return torch.zeros(sae.d_hidden), 0

    stacked = torch.stack(all_vecs, dim=0)      # (N, d_hidden)
    return stacked.mean(dim=0), len(all_vecs)


# ============================================================================
# Analysis helpers
# ============================================================================

def top_k_features(mean_acts: torch.Tensor, k: int) -> List[int]:
    """Return indices of the top-k features by mean activation."""
    return mean_acts.topk(k).indices.tolist()


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return (a @ b / denom).item()


# ============================================================================
# Report building
# ============================================================================

def analyse_cluster(
    cluster_name: str,
    word_sentences: Dict[str, List[str]],
    tokenizer: GPT2Tokenizer,
    gpt2: GPT2LMHeadModel,
    sae: SparseAutoencoder,
    layer_index: int,
    device: str,
    top_k: int,
) -> dict:
    """Run the full synonym analysis for one semantic cluster."""

    print(f"\n  Cluster: '{cluster_name}'")
    print(f"  Words:   {list(word_sentences.keys())}")

    profiles: Dict[str, torch.Tensor] = {}
    n_positions: Dict[str, int] = {}

    for word, sentences in word_sentences.items():
        mean_acts, n = collect_word_feature_profile(
            word, sentences, tokenizer, gpt2, sae, layer_index, device
        )
        profiles[word] = mean_acts
        n_positions[word] = n
        active = int((mean_acts > 0).sum().item())
        print(f"    {word:>12}: {n} positions found, {active} features active")

    # Per-word top-K feature sets
    top_features: Dict[str, List[int]] = {
        w: top_k_features(v, top_k) for w, v in profiles.items()
    }

    # Pairwise Jaccard + cosine
    words = list(profiles.keys())
    pairwise = []
    for w1, w2 in combinations(words, 2):
        s1 = set(top_features[w1])
        s2 = set(top_features[w2])
        shared = sorted(s1 & s2)
        j = jaccard(s1, s2)
        cos = cosine_sim(profiles[w1], profiles[w2])
        pairwise.append({
            "word_a": w1,
            "word_b": w2,
            "jaccard": round(j, 4),
            "cosine_sim": round(cos, 4),
            "shared_feature_count": len(shared),
            "shared_features": shared,
        })

    # Shared features across ALL words in the cluster
    all_sets = [set(top_features[w]) for w in words]
    universal_shared = sorted(set.intersection(*all_sets))

    # Features unique to each word (not in any other word's top-K)
    unique_to: Dict[str, List[int]] = {}
    for w in words:
        others = set.union(*(set(top_features[x]) for x in words if x != w))
        unique_to[w] = sorted(set(top_features[w]) - others)

    # Mean Jaccard across all pairs (summary score)
    mean_jaccard = (
        sum(p["jaccard"] for p in pairwise) / len(pairwise) if pairwise else 0.0
    )
    mean_cosine = (
        sum(p["cosine_sim"] for p in pairwise) / len(pairwise) if pairwise else 0.0
    )

    print(f"    → Mean Jaccard: {mean_jaccard:.3f}  |  Mean cosine: {mean_cosine:.3f}")
    print(f"    → Features shared by ALL {len(words)} words: {len(universal_shared)}")

    return {
        "cluster": cluster_name,
        "words": words,
        "top_k": top_k,
        "n_positions": n_positions,
        "top_features_per_word": {w: top_features[w] for w in words},
        "pairwise": pairwise,
        "universal_shared_features": universal_shared,
        "unique_features_per_word": unique_to,
        "mean_jaccard": round(mean_jaccard, 4),
        "mean_cosine_sim": round(mean_cosine, 4),
        "interpretation": (
            "STRONG synonym signal"   if mean_jaccard > 0.40 else
            "MODERATE synonym signal" if mean_jaccard > 0.20 else
            "WEAK synonym signal"
        ),
    }


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test whether SAE features are activated consistently by synonyms."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt",
        help="Path to trained SAE checkpoint."
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top features per word to compare. Default: 30"
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
        "--output", default="synonym_test_report.json",
        help="Output JSON path. Default: synonym_test_report.json"
    )
    parser.add_argument(
        "--clusters", nargs="*", default=None,
        choices=list(SYNONYM_CLUSTERS.keys()),
        help=f"Which clusters to test. Default: all. "
             f"Available: {list(SYNONYM_CLUSTERS.keys())}"
    )
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    # ── Load SAE ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SYNONYM FEATURE-OVERLAP TEST")
    print("=" * 70)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"\n[1/3] Loading SAE from {ckpt}...")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp     = payload.get("hyperparameters", {})
    state  = payload["model_state_dict"]
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
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(device)
    gpt2.eval()
    print(f"  GPT-2 ready on {device}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    print(f"\n[3/3] Analysing synonym clusters (top-{args.top_k} features)...")

    clusters_to_run = args.clusters or list(SYNONYM_CLUSTERS.keys())
    all_results = []

    for cluster_name in clusters_to_run:
        result = analyse_cluster(
            cluster_name=cluster_name,
            word_sentences=SYNONYM_CLUSTERS[cluster_name],
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
            "checkpoint": str(ckpt),
            "top_k": args.top_k,
            "layer_index": layer_index,
            "d_model": d_model,
            "d_hidden": d_hidden,
            "device": device,
        },
        "clusters": all_results,
        "overall_mean_jaccard": round(
            sum(r["mean_jaccard"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0,
    }

    out_path = Path(args.output)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    # ── Human-readable summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Cluster':<10}  {'Words':<34}  {'Jaccard':>7}  {'Cosine':>7}  "
          f"{'All-shared':>10}  Signal")
    print("-" * 80)
    for r in all_results:
        words_str = "/".join(r["words"])
        print(
            f"{r['cluster']:<10}  {words_str:<34}  "
            f"{r['mean_jaccard']:>7.3f}  {r['mean_cosine_sim']:>7.3f}  "
            f"{len(r['universal_shared_features']):>10}  {r['interpretation']}"
        )

    print("\n" + "-" * 80)
    print(f"Overall mean Jaccard: {report['overall_mean_jaccard']:.3f}")
    print(
        "\nJaccard guide:  > 0.40 → synonyms share features strongly\n"
        "                0.20–0.40 → moderate overlap\n"
        "                < 0.20 → the SAE does not group these synonyms"
    )
    print(f"\nFull report saved to: {args.output}")

    # Per-cluster pairwise detail
    print("\n" + "=" * 70)
    print("PAIRWISE DETAIL")
    print("=" * 70)
    for r in all_results:
        print(f"\nCluster: {r['cluster']}")
        for pw in r["pairwise"]:
            shared_preview = pw["shared_features"][:8]
            more = f" …+{pw['shared_feature_count']-8}" if pw["shared_feature_count"] > 8 else ""
            print(
                f"  {pw['word_a']:>12} ↔ {pw['word_b']:<12}  "
                f"Jaccard={pw['jaccard']:.3f}  cos={pw['cosine_sim']:.3f}  "
                f"shared={pw['shared_feature_count']}  "
                f"features={shared_preview}{more}"
            )
        if r["universal_shared_features"]:
            print(f"  → All-word shared features: {r['universal_shared_features'][:10]}")
        else:
            print("  → No features shared by ALL words in this cluster")


if __name__ == "__main__":
    main()
