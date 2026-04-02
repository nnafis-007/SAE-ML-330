"""Caps invariance analyzer using pretrained SAE-Lens models."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch

_SAE_ROOT = Path(__file__).resolve().parent.parent
if str(_SAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAE_ROOT))

from analyzers import BaseAnalyzer, register  # noqa: E402
from pretrained_sae import PretrainedSAEStore  # noqa: E402


WORD_TEMPLATES: Dict[str, List[str]] = {
    "cat": [
        "The {} sat on the mat and watched the birds outside.",
        "She adopted a {} from the local shelter.",
        "He saw a {} sleeping in the sunlight.",
    ],
    "king": [
        "The {} addressed the crowd from the balcony.",
        "A {} once ruled this land with wisdom.",
        "Everyone bowed as the {} entered.",
    ],
    "happy": [
        "She felt {} when she read the letter.",
        "The children were {} to see the puppies.",
        "He looked {} after the exam result.",
    ],
}


def _cap_variants(word: str) -> List[Tuple[str, str]]:
    w = word.lower()

    def alternating(s: str, start_upper: bool) -> str:
        return "".join(c.upper() if (i % 2 == 0) == start_upper else c.lower() for i, c in enumerate(s))

    candidates = [
        ("lower", w),
        ("upper", w.upper()),
        ("title", w.capitalize()),
        ("alt_lo", alternating(w, start_upper=False)),
        ("alt_hi", alternating(w, start_upper=True)),
    ]

    seen = set()
    unique: List[Tuple[str, str]] = []
    for label, variant in candidates:
        if variant not in seen:
            seen.add(variant)
            unique.append((label, variant))
    return unique


def _top_k_features(vec: torch.Tensor, k: int) -> List[int]:
    kk = min(max(1, k), vec.numel())
    vals, idxs = torch.topk(vec, kk)
    return [int(i) for i, v in zip(idxs.tolist(), vals.tolist()) if v > 0]


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    an = torch.norm(a)
    bn = torch.norm(b)
    if an.item() == 0 or bn.item() == 0:
        return 0.0
    return float(torch.dot(a, b) / (an * bn))


class CapsAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._store = PretrainedSAEStore(device=self._device)

    @property
    def name(self) -> str:
        return "caps"

    def list_models(self) -> List[Dict[str, Any]]:
        return self._store.list_default_models()

    def analyze(self, text: str, model_id: str, **kwargs) -> Dict[str, Any]:
        top_k = int(kwargs.get("top_k", 30))
        requested = [w.strip().lower() for w in text.split(",") if w.strip()] if text else []
        words_to_run = list(WORD_TEMPLATES.keys()) if not requested or requested == ["all"] else [w for w in requested if w in WORD_TEMPLATES]

        bundle = self._store.get_bundle(model_id)
        sae = bundle.sae
        model = bundle.model
        tokenizer = bundle.tokenizer
        meta = bundle.metadata

        all_results = []
        for word in words_to_run:
            result = self._analyse_word(
                base_word=word,
                templates=WORD_TEMPLATES[word],
                tokenizer=tokenizer,
                model=model,
                sae=sae,
                hook_name=meta.hook_name,
                top_k=top_k,
                max_length=min(128, max(1, int(meta.context_size))),
            )
            all_results.append(result)

        overall_j = sum(r["mean_jaccard"] for r in all_results) / len(all_results) if all_results else 0.0
        overall_c = sum(r["mean_cosine_sim"] for r in all_results) / len(all_results) if all_results else 0.0

        return {
            "model": model_id,
            "test_type": "caps",
            "settings": {
                "top_k": top_k,
                "layer_index": meta.layer_index,
                "d_model": meta.d_model,
                "d_hidden": meta.d_hidden,
                "hook_name": meta.hook_name,
            },
            "words": all_results,
            "available_words": list(WORD_TEMPLATES.keys()),
            "overall_mean_jaccard": round(overall_j, 4),
            "overall_mean_cosine": round(overall_c, 4),
        }

    def _analyse_word(
        self,
        base_word: str,
        templates: List[str],
        tokenizer,
        model,
        sae,
        hook_name: str,
        top_k: int,
        max_length: int,
    ) -> Dict[str, Any]:
        variants = _cap_variants(base_word)
        profiles: Dict[str, torch.Tensor] = {}
        n_positions: Dict[str, int] = {}
        labels: Dict[str, str] = {}

        for label, variant in variants:
            mean_acts, n = self._collect_variant_profile(
                variant=variant,
                templates=templates,
                tokenizer=tokenizer,
                model=model,
                sae=sae,
                hook_name=hook_name,
                max_length=max_length,
            )
            profiles[variant] = mean_acts
            n_positions[variant] = n
            labels[variant] = label

        top_features = {v: _top_k_features(p, top_k) for v, p in profiles.items()}
        variant_strs = [v for _, v in variants]

        pairwise = []
        for va, vb in combinations(variant_strs, 2):
            sa = set(top_features[va])
            sb = set(top_features[vb])
            shared = sorted(sa & sb)
            pairwise.append(
                {
                    "variant_a": va,
                    "label_a": labels[va],
                    "variant_b": vb,
                    "label_b": labels[vb],
                    "jaccard": round(_jaccard(sa, sb), 4),
                    "cosine_sim": round(_cosine(profiles[va], profiles[vb]), 4),
                    "shared_feature_count": len(shared),
                    "shared_features": shared,
                }
            )

        all_sets = [set(top_features[v]) for v in variant_strs]
        universal = sorted(set.intersection(*all_sets)) if all_sets else []

        lower_v = base_word.lower()
        upper_v = base_word.upper()
        key_pair = next((p for p in pairwise if p["variant_a"] == lower_v and p["variant_b"] == upper_v), None)

        mean_j = sum(p["jaccard"] for p in pairwise) / len(pairwise) if pairwise else 0.0
        mean_c = sum(p["cosine_sim"] for p in pairwise) / len(pairwise) if pairwise else 0.0

        return {
            "word": base_word,
            "variants": [{"label": l, "form": v} for l, v in variants],
            "top_k": top_k,
            "n_positions": {v: n_positions[v] for v in variant_strs},
            "top_features_per_variant": {v: top_features[v] for v in variant_strs},
            "pairwise": pairwise,
            "universal_shared_features": universal,
            "mean_jaccard": round(mean_j, 4),
            "mean_cosine_sim": round(mean_c, 4),
            "lower_vs_upper": key_pair,
            "interpretation": (
                "CASE-INVARIANT (strong)" if mean_j > 0.40 else "PARTIALLY case-sensitive" if mean_j > 0.20 else "CASE-SENSITIVE (features differ)"
            ),
        }

    def _collect_variant_profile(
        self,
        variant: str,
        templates: List[str],
        tokenizer,
        model,
        sae,
        hook_name: str,
        max_length: int,
    ) -> tuple[torch.Tensor, int]:
        rows: List[torch.Tensor] = []
        candidate_first_tokens = set()
        for prefix in ("", " "):
            ids = tokenizer.encode(prefix + variant, add_special_tokens=False)
            if ids:
                candidate_first_tokens.add(ids[0])

        with torch.no_grad():
            for template in templates:
                sentence = template.format(variant)
                token_data = tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )
                input_ids = token_data["input_ids"].to(self._device)
                token_list = input_ids[0].tolist()
                if not token_list:
                    continue

                _, cache = model.run_with_cache(input_ids, prepend_bos=False)
                feature_mat = sae.encode(cache[hook_name])[0].detach().cpu()

                for pos, tid in enumerate(token_list):
                    if tid in candidate_first_tokens:
                        rows.append(feature_mat[pos])

        if not rows:
            return torch.zeros(sae.d_hidden), 0
        stacked = torch.stack(rows, dim=0)
        return stacked.mean(dim=0), stacked.shape[0]


_instance = CapsAnalyzer()
register(_instance)
