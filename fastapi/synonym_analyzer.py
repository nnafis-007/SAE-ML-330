"""
Synonym Analyzer – exposes the synonym feature-overlap test via the API.

Reuses the extraction logic from ``run_synonym_test.py`` and wraps it in a
``BaseAnalyzer`` subclass so the FastAPI backend can serve synonym-overlap
results to the UI.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Make the SAE source package importable
# ---------------------------------------------------------------------------
_SAE_ROOT = Path(__file__).resolve().parent.parent
_SAE_SRC = _SAE_ROOT / "src"
if str(_SAE_SRC) not in sys.path:
    sys.path.insert(0, str(_SAE_SRC))

from sae_model import SparseAutoencoder  # noqa: E402
from . import BaseAnalyzer, register  # noqa: E402

CHECKPOINTS_DIR = _SAE_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Import synonym clusters from the test script
# ---------------------------------------------------------------------------
_SYNONYM_SCRIPT = _SAE_ROOT / "run_synonym_test.py"
sys.path.insert(0, str(_SAE_ROOT))
from run_synonym_test import (  # noqa: E402
    SYNONYM_CLUSTERS,
    collect_word_feature_profile,
    top_k_features,
    jaccard,
    cosine_sim,
)

# ---------------------------------------------------------------------------
# Generic sentence templates for user-supplied words
# Each template has exactly one {} where the word is inserted.
# Variety covers different syntactic positions so BPE tokenisation
# can find the word in several contexts.
# ---------------------------------------------------------------------------
_GENERIC_TEMPLATES: List[str] = [
    "She felt {} when she heard the good news from her friend.",
    "The {} atmosphere in the room was obvious to everyone present.",
    "He described the situation as {} during the team meeting.",
    "It was a {} experience that none of them would ever forget.",
    "Everyone agreed that the result was truly {} and remarkable.",
    "The professor said the answer was {} and moved on quickly.",
    "A {} feeling spread through the crowd after the announcement.",
    "She looked {} as she walked out of the examination hall.",
    "They considered the whole event to be quite {} overall.",
    "He was {} about the new project that had just been approved.",
    "The children seemed {} while playing together in the garden.",
    "We found the entire performance to be {} and well executed.",
    "His {} reaction surprised everyone who was watching closely.",
    "The report concluded that the outcome was {} for the company.",
    "Being {} is something that many people aspire to every day.",
]


class SynonymAnalyzer(BaseAnalyzer):
    """
    Analyzer that runs the synonym feature-overlap test.

    * ``list_models`` scans the SAE checkpoints directory.
    * ``analyze`` runs the synonym test for one or all clusters using the
      selected checkpoint.
    """

    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR):
        self._checkpoints_dir = checkpoints_dir
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # lazy caches
        self._sae_cache: Dict[str, Dict[str, Any]] = {}
        self._gpt2_cache: Dict[str, Any] = {}  # keyed by device
        self._model_list_cache: Optional[List[Dict[str, Any]]] = None

    # -- BaseAnalyzer interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "synonym"

    def list_models(self) -> List[Dict[str, Any]]:
        if self._model_list_cache is None:
            self._model_list_cache = self._scan_checkpoints()
        return self._model_list_cache

    def analyze(self, text: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """
        Run synonym overlap test.

        ``text`` is repurposed as a comma-separated list of cluster names to
        test (or "all" / empty for all clusters).

        Extra kwargs accepted:
            top_k (int): number of top features per word (default 30).
            custom_words (list[str]): user-supplied words to compare.
        """
        top_k = kwargs.get("top_k", 30)
        custom_words = kwargs.get("custom_words", None)

        # Load models
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]
        tokenizer, gpt2 = self._get_gpt2()

        # ----- Custom word mode ------------------------------------------------
        if custom_words and len(custom_words) >= 2:
            # Generate sentences for each word using generic templates
            word_sentences: Dict[str, List[str]] = {}
            for word in custom_words:
                w = word.strip()
                if not w:
                    continue
                word_sentences[w] = [t.format(w) for t in _GENERIC_TEMPLATES]

            if len(word_sentences) < 2:
                return {"error": "Need at least 2 valid words to compare."}

            cluster_name = " / ".join(word_sentences.keys())
            result = self._analyse_cluster(
                cluster_name=cluster_name,
                word_sentences=word_sentences,
                tokenizer=tokenizer,
                gpt2=gpt2,
                sae=sae,
                layer_index=layer_index,
                device=self._device,
                top_k=top_k,
            )

            return {
                "model": model_id,
                "test_type": "synonym",
                "mode": "custom",
                "settings": {
                    "checkpoint": model_id,
                    "top_k": top_k,
                    "layer_index": layer_index,
                    "d_model": info["d_model"],
                    "d_hidden": info["d_hidden"],
                },
                "clusters": [result],
                "available_clusters": list(SYNONYM_CLUSTERS.keys()),
                "overall_mean_jaccard": result["mean_jaccard"],
            }

        # ----- Predefined cluster mode -----------------------------------------
        requested = [c.strip() for c in text.split(",") if c.strip()] if text else []
        if not requested or requested == ["all"]:
            clusters_to_run = list(SYNONYM_CLUSTERS.keys())
        else:
            clusters_to_run = [c for c in requested if c in SYNONYM_CLUSTERS]

        # Load models
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]
        tokenizer, gpt2 = self._get_gpt2()

        all_results = []
        for cluster_name in clusters_to_run:
            result = self._analyse_cluster(
                cluster_name=cluster_name,
                word_sentences=SYNONYM_CLUSTERS[cluster_name],
                tokenizer=tokenizer,
                gpt2=gpt2,
                sae=sae,
                layer_index=layer_index,
                device=self._device,
                top_k=top_k,
            )
            all_results.append(result)

        overall_mean_jaccard = round(
            sum(r["mean_jaccard"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0

        return {
            "model": model_id,
            "test_type": "synonym",
            "settings": {
                "checkpoint": model_id,
                "top_k": top_k,
                "layer_index": layer_index,
                "d_model": info["d_model"],
                "d_hidden": info["d_hidden"],
            },
            "clusters": all_results,
            "available_clusters": list(SYNONYM_CLUSTERS.keys()),
            "overall_mean_jaccard": overall_mean_jaccard,
        }

    # -- cluster analysis (mirrors run_synonym_test.analyse_cluster) -----------

    def _analyse_cluster(
        self,
        cluster_name: str,
        word_sentences: Dict[str, List[str]],
        tokenizer,
        gpt2,
        sae: SparseAutoencoder,
        layer_index: int,
        device: str,
        top_k: int,
    ) -> dict:
        profiles: Dict[str, torch.Tensor] = {}
        n_positions: Dict[str, int] = {}

        for word, sentences in word_sentences.items():
            print(f"[synonym_analyzer] word='{word}' | {len(sentences)} sentence(s):")
            for i, sent in enumerate(sentences, 1):
                print(f"  [{i}] {sent}")
            mean_acts, n = collect_word_feature_profile(
                word, sentences, tokenizer, gpt2, sae, layer_index, device
            )
            profiles[word] = mean_acts
            n_positions[word] = n

        top_features: Dict[str, List[int]] = {
            w: top_k_features(v, top_k) for w, v in profiles.items()
        }

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

        all_sets = [set(top_features[w]) for w in words]
        universal_shared = sorted(set.intersection(*all_sets))

        unique_to: Dict[str, List[int]] = {}
        for w in words:
            others = set.union(*(set(top_features[x]) for x in words if x != w))
            unique_to[w] = sorted(set(top_features[w]) - others)

        mean_jaccard = (
            sum(p["jaccard"] for p in pairwise) / len(pairwise) if pairwise else 0.0
        )
        mean_cosine = (
            sum(p["cosine_sim"] for p in pairwise) / len(pairwise) if pairwise else 0.0
        )

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

    # -- internal helpers ------------------------------------------------------

    def _scan_checkpoints(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        for pt_file in sorted(self._checkpoints_dir.rglob("*.pt")):
            rel = pt_file.relative_to(self._checkpoints_dir)
            try:
                meta = self._peek_checkpoint(str(pt_file))
                models.append({"id": str(rel), "name": str(rel), **meta})
            except Exception as exc:
                models.append({"id": str(rel), "name": str(rel), "error": str(exc)})
        return models

    def _peek_checkpoint(self, path: str) -> Dict[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return {"d_model": 768, "d_hidden": 12288, "l1_coeff": "unknown", "layer_index": 8}

        hp = payload.get("hyperparameters", {})
        state = payload.get("model_state_dict", {})
        d_model = hp.get("d_model") or (int(state["W_enc"].shape[1]) if "W_enc" in state else 768)
        d_hidden = hp.get("d_hidden") or (int(state["W_enc"].shape[0]) if "W_enc" in state else 12288)
        layer_index = hp.get("layer_index", 8)
        l1_coeff = hp.get("l1_coeff", payload.get("l1_coeff", "unknown"))
        return {"d_model": d_model, "d_hidden": d_hidden, "l1_coeff": l1_coeff, "layer_index": layer_index}

    def _load_sae(self, checkpoint_path: str) -> Dict[str, Any]:
        if checkpoint_path in self._sae_cache:
            return self._sae_cache[checkpoint_path]

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hp = payload.get("hyperparameters", {})
        state = payload.get("model_state_dict", None)
        if state is None:
            state = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}

        d_model = int(state["W_enc"].shape[1])
        d_hidden = int(state["W_enc"].shape[0])
        l1_coeff = hp.get("l1_coeff", payload.get("l1_coeff", 3e-4))
        if not isinstance(l1_coeff, (int, float)):
            l1_coeff = 3e-4
        layer_index = hp.get("layer_index", 8)

        sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff)
        sae.load_state_dict(state, strict=False)
        sae.to(self._device).eval()

        entry = {"sae": sae, "layer_index": layer_index, "d_model": d_model, "d_hidden": d_hidden}
        self._sae_cache[checkpoint_path] = entry
        return entry

    def _get_gpt2(self):
        if "tokenizer" not in self._gpt2_cache:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True).to(self._device)
            gpt2.eval()
            self._gpt2_cache["tokenizer"] = tokenizer
            self._gpt2_cache["gpt2"] = gpt2
        return self._gpt2_cache["tokenizer"], self._gpt2_cache["gpt2"]


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------
_instance = SynonymAnalyzer(checkpoints_dir=CHECKPOINTS_DIR)
register(_instance)
