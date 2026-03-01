"""
Caps Analyzer – exposes the capitalisation invariance test via the API.

Reuses the extraction logic from ``run_caps_test.py`` and wraps it in a
``BaseAnalyzer`` subclass so the FastAPI backend can serve caps-invariance
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
# Import caps test helpers from the test script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_SAE_ROOT))
from run_caps_test import (  # noqa: E402
    WORD_TEMPLATES,
    _cap_variants,
    collect_variant_profile,
    top_k_features,
    jaccard,
    cosine_sim,
)


class CapsAnalyzer(BaseAnalyzer):
    """
    Analyzer that runs the capitalisation invariance test.

    * ``list_models`` scans the SAE checkpoints directory.
    * ``analyze`` runs the caps test for one or all words using the
      selected checkpoint.
    """

    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR):
        self._checkpoints_dir = checkpoints_dir
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sae_cache: Dict[str, Dict[str, Any]] = {}
        self._gpt2_cache: Dict[str, Any] = {}
        self._model_list_cache: Optional[List[Dict[str, Any]]] = None

    # -- BaseAnalyzer interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "caps"

    def list_models(self) -> List[Dict[str, Any]]:
        if self._model_list_cache is None:
            self._model_list_cache = self._scan_checkpoints()
        return self._model_list_cache

    def analyze(self, text: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """
        Run capitalisation invariance test.

        ``text`` is repurposed as a comma-separated list of words to test
        (or "all" / empty for all words).

        Extra kwargs:
            top_k (int): number of top features per variant (default 30).
        """
        top_k = kwargs.get("top_k", 30)
        requested = [w.strip().lower() for w in text.split(",") if w.strip()] if text else []
        if not requested or requested == ["all"]:
            words_to_run = list(WORD_TEMPLATES.keys())
        else:
            words_to_run = [w for w in requested if w in WORD_TEMPLATES]

        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]
        tokenizer, gpt2 = self._get_gpt2()

        all_results = []
        for word in words_to_run:
            result = self._analyse_word(
                base_word=word,
                templates=WORD_TEMPLATES[word],
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
        overall_mean_cosine = round(
            sum(r["mean_cosine_sim"] for r in all_results) / len(all_results), 4
        ) if all_results else 0.0

        return {
            "model": model_id,
            "test_type": "caps",
            "settings": {
                "checkpoint": model_id,
                "top_k": top_k,
                "layer_index": layer_index,
                "d_model": info["d_model"],
                "d_hidden": info["d_hidden"],
            },
            "words": all_results,
            "available_words": list(WORD_TEMPLATES.keys()),
            "overall_mean_jaccard": overall_mean_jaccard,
            "overall_mean_cosine": overall_mean_cosine,
        }

    # -- per-word analysis (mirrors run_caps_test.analyse_word) ----------------

    def _analyse_word(
        self,
        base_word: str,
        templates: List[str],
        tokenizer,
        gpt2,
        sae: SparseAutoencoder,
        layer_index: int,
        device: str,
        top_k: int,
    ) -> dict:
        variants = _cap_variants(base_word)

        profiles: Dict[str, torch.Tensor] = {}
        n_positions: Dict[str, int] = {}
        variant_labels: Dict[str, str] = {}

        for label, variant in variants:
            mean_acts, n = collect_variant_profile(
                variant, templates, tokenizer, gpt2, sae, layer_index, device
            )
            profiles[variant] = mean_acts
            n_positions[variant] = n
            variant_labels[variant] = label

        top_features: Dict[str, List[int]] = {
            v: top_k_features(p, top_k) for v, p in profiles.items()
        }

        variant_strs = [v for _, v in variants]
        pairwise = []
        for va, vb in combinations(variant_strs, 2):
            sa, sb = set(top_features[va]), set(top_features[vb])
            shared = sorted(sa & sb)
            pairwise.append({
                "variant_a": va,
                "label_a": variant_labels[va],
                "variant_b": vb,
                "label_b": variant_labels[vb],
                "jaccard": round(jaccard(sa, sb), 4),
                "cosine_sim": round(cosine_sim(profiles[va], profiles[vb]), 4),
                "shared_feature_count": len(shared),
                "shared_features": shared,
            })

        all_sets = [set(top_features[v]) for v in variant_strs]
        universal = sorted(set.intersection(*all_sets))

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

        return {
            "word": base_word,
            "variants": [{"label": l, "form": v} for l, v in variants],
            "top_k": top_k,
            "n_positions": {v: n_positions[v] for v in variant_strs},
            "top_features_per_variant": {v: top_features[v] for v in variant_strs},
            "pairwise": pairwise,
            "universal_shared_features": universal,
            "mean_jaccard": round(mean_jaccard, 4),
            "mean_cosine_sim": round(mean_cosine, 4),
            "lower_vs_upper": key_pair,
            "interpretation": (
                "CASE-INVARIANT (strong)"   if mean_jaccard > 0.40 else
                "PARTIALLY case-sensitive"  if mean_jaccard > 0.20 else
                "CASE-SENSITIVE (features differ)"
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
_instance = CapsAnalyzer(checkpoints_dir=CHECKPOINTS_DIR)
register(_instance)
