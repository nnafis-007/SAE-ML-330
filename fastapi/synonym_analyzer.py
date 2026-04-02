"""Synonym analyzer using pretrained SAE-Lens models."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

_SAE_ROOT = Path(__file__).resolve().parent.parent
if str(_SAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAE_ROOT))

from analyzers import BaseAnalyzer, register  # noqa: E402
from pretrained_sae import PretrainedSAEStore  # noqa: E402


SYNONYM_CLUSTERS: Dict[str, Dict[str, List[str]]] = {
    "happy": {
        "happy": [
            "She felt happy when she heard the good news.",
            "The children were happy to see the puppies.",
            "He looked happy after the long project ended.",
        ],
        "joyful": [
            "She felt joyful when she heard the good news.",
            "The children were joyful to see the puppies.",
            "He looked joyful after the long project ended.",
        ],
        "elated": [
            "She felt elated when she heard the good news.",
            "The children were elated to see the puppies.",
            "He looked elated after the long project ended.",
        ],
    },
    "large": {
        "large": [
            "A large crowd gathered outside the stadium.",
            "She carried a large bag full of groceries.",
            "There is a large lake behind the mountain range.",
        ],
        "big": [
            "A big crowd gathered outside the stadium.",
            "She carried a big bag full of groceries.",
            "There is a big lake behind the mountain range.",
        ],
        "huge": [
            "A huge crowd gathered outside the stadium.",
            "She carried a huge bag full of groceries.",
            "There is a huge lake behind the mountain range.",
        ],
    },
}

_GENERIC_TEMPLATES: List[str] = [
    "Consider the word {}.",
    "Let us talk about {}.",
    "In this context, {} is used.",
]


def _top_k_features(vec: torch.Tensor, k: int) -> List[int]:
    if k <= 0:
        mask = vec > 0
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        vals = vec[idxs]
        order = torch.argsort(vals, descending=True)
        return idxs[order].tolist()
    kk = min(k, vec.numel())
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


class SynonymAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._store = PretrainedSAEStore(device=self._device)

    @property
    def name(self) -> str:
        return "synonym"

    def list_models(self) -> List[Dict[str, Any]]:
        return self._store.list_default_models()

    def analyze(self, text: str, model_id: str, **kwargs) -> Dict[str, Any]:
        top_k = int(kwargs.get("top_k", 1000))
        custom_words = kwargs.get("custom_words")

        bundle = self._store.get_bundle(model_id)
        sae = bundle.sae
        model = bundle.model
        tokenizer = bundle.tokenizer
        meta = bundle.metadata

        if custom_words and len(custom_words) >= 2:
            word_sentences = {
                w.strip(): [tmpl.format(w.strip()) for tmpl in _GENERIC_TEMPLATES]
                for w in custom_words
                if str(w).strip()
            }
            clusters_to_run = [(" / ".join(word_sentences.keys()), word_sentences)]
            mode = "custom"
        else:
            requested = [c.strip() for c in text.split(",") if c.strip()] if text else []
            cluster_names = list(SYNONYM_CLUSTERS.keys()) if not requested or requested == ["all"] else [c for c in requested if c in SYNONYM_CLUSTERS]
            clusters_to_run = [(name, SYNONYM_CLUSTERS[name]) for name in cluster_names]
            mode = "preset"

        cluster_results: List[Dict[str, Any]] = []
        for cluster_name, word_sentences in clusters_to_run:
            profiles: Dict[str, torch.Tensor] = {}
            n_positions: Dict[str, int] = {}

            for word, sentences in word_sentences.items():
                mean_acts, n_pos = self._collect_word_profile(
                    word=word,
                    sentences=sentences,
                    tokenizer=tokenizer,
                    model=model,
                    sae=sae,
                    hook_name=meta.hook_name,
                    max_length=min(128, max(1, int(meta.context_size))),
                )
                profiles[word] = mean_acts
                n_positions[word] = n_pos

            top_features = {word: _top_k_features(vec, top_k) for word, vec in profiles.items()}
            words = list(profiles.keys())
            pairwise = []
            for w1, w2 in combinations(words, 2):
                s1 = set(top_features[w1])
                s2 = set(top_features[w2])
                shared = sorted(s1 & s2)
                pairwise.append(
                    {
                        "word_a": w1,
                        "word_b": w2,
                        "jaccard": round(_jaccard(s1, s2), 4),
                        "cosine_sim": round(_cosine(profiles[w1], profiles[w2]), 4),
                        "shared_feature_count": len(shared),
                        "shared_features": [{"id": fid, "label": f"Feature {fid}"} for fid in shared],
                    }
                )

            all_sets = [set(top_features[w]) for w in words]
            universal = sorted(set.intersection(*all_sets)) if all_sets else []

            mean_j = sum(p["jaccard"] for p in pairwise) / len(pairwise) if pairwise else 0.0
            mean_c = sum(p["cosine_sim"] for p in pairwise) / len(pairwise) if pairwise else 0.0

            cluster_results.append(
                {
                    "cluster": cluster_name,
                    "words": words,
                    "top_k": top_k,
                    "n_positions": n_positions,
                    "top_features_per_word": top_features,
                    "pairwise": pairwise,
                    "universal_shared_features": [{"id": fid, "label": f"Feature {fid}"} for fid in universal],
                    "mean_jaccard": round(mean_j, 4),
                    "mean_cosine_sim": round(mean_c, 4),
                    "interpretation": (
                        "STRONG synonym signal" if mean_j > 0.40 else "MODERATE synonym signal" if mean_j > 0.20 else "WEAK synonym signal"
                    ),
                }
            )

        overall = sum(c["mean_jaccard"] for c in cluster_results) / len(cluster_results) if cluster_results else 0.0
        return {
            "model": model_id,
            "test_type": "synonym",
            "mode": mode,
            "settings": {
                "top_k": top_k,
                "layer_index": meta.layer_index,
                "d_model": meta.d_model,
                "d_hidden": meta.d_hidden,
                "hook_name": meta.hook_name,
            },
            "clusters": cluster_results,
            "available_clusters": list(SYNONYM_CLUSTERS.keys()),
            "overall_mean_jaccard": round(overall, 4),
        }

    def _collect_word_profile(
        self,
        word: str,
        sentences: List[str],
        tokenizer,
        model,
        sae,
        hook_name: str,
        max_length: int,
    ) -> tuple[torch.Tensor, int]:
        rows: List[torch.Tensor] = []
        candidate_first_tokens = set()
        for prefix in ("", " "):
            ids = tokenizer.encode(prefix + word, add_special_tokens=False)
            if ids:
                candidate_first_tokens.add(ids[0])

        with torch.no_grad():
            for sentence in sentences:
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


_instance = SynonymAnalyzer()
register(_instance)
