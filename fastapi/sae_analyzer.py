"""
SAE analyzer backed by pretrained SAE-Lens models.

Model identifier format:
    release:sae_id
Example:
    gpt2-small-res-jb:blocks.8.hook_resid_pre
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_SAE_ROOT = Path(__file__).resolve().parent.parent
if str(_SAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAE_ROOT))

from analyzers import BaseAnalyzer, register  # noqa: E402
from pretrained_sae import PretrainedSAEStore  # noqa: E402


class SAEAnalyzer(BaseAnalyzer):
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._store = PretrainedSAEStore(device=self._device)
        self._ui_cache_dir = _SAE_ROOT / "checkpoints" / "ui_cache"
        self._ui_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "sae"

    def _cache_path(self, namespace: str, payload: Dict[str, Any]) -> Path:
        cache_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return self._ui_cache_dir / f"{namespace}_{cache_key}.json"

    def _load_json_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_json_cache(self, cache_path: Path, payload: Dict[str, Any]) -> None:
        try:
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            # Best-effort cache write; never fail the API call because of caching.
            pass

    def list_models(self) -> List[Dict[str, Any]]:
        return self._store.list_default_models()

    def analyze(self, text: str, model_id: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        bundle = self._store.get_bundle(model_id)
        sae = bundle.sae
        model = bundle.model
        meta = bundle.metadata

        token_data = bundle.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max(1, int(meta.context_size)),
            add_special_tokens=False,
        )
        input_ids = token_data["input_ids"].to(self._device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, prepend_bos=False)
            hook_acts = cache[meta.hook_name]  # (batch, pos, d_model)
            feature_acts = sae.encode(hook_acts)[0].detach().cpu()  # (pos, d_hidden)

        token_ids = input_ids[0].tolist()
        tokens_data: List[Dict[str, Any]] = []

        for i, tid in enumerate(token_ids):
            token_str = bundle.tokenizer.decode([tid])
            feat_vec = feature_acts[i]
            active_mask = feat_vec > 0
            active_count = int(active_mask.sum().item())

            if top_k <= 0:
                chosen_vals = feat_vec[active_mask]
                chosen_idxs = torch.nonzero(active_mask, as_tuple=False).flatten()
                order = torch.argsort(chosen_vals, descending=True)
                chosen_vals = chosen_vals[order]
                chosen_idxs = chosen_idxs[order]
            else:
                k = min(top_k, max(active_count, 1))
                chosen_vals, chosen_idxs = torch.topk(feat_vec, k)

            feat_list: List[Dict[str, Any]] = []
            for val, idx in zip(chosen_vals.tolist(), chosen_idxs.tolist()):
                if val > 0:
                    feat_id = int(idx)
                    feat_list.append(
                        {
                            "id": feat_id,
                            "activation": round(float(val), 4),
                            "description": f"Feature {feat_id}",
                        }
                    )

            tokens_data.append({"text": token_str, "features": feat_list})

        return {
            "model": model_id,
            "layer_index": meta.layer_index,
            "hook_name": meta.hook_name,
            "context_size": meta.context_size,
            "prepend_bos": meta.prepend_bos,
            "d_hidden": meta.d_hidden,
            "tokens": tokens_data,
        }

    def label_feature(
        self,
        model_id: str,
        feature_idx: int,
        corpus_texts: Optional[List[str]] = None,
        labeling_config: Optional[Dict[str, Any]] = None,
        groq_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        bundle = self._store.get_bundle(model_id)
        sae = bundle.sae
        model = bundle.model
        tokenizer = bundle.tokenizer
        meta = bundle.metadata

        if corpus_texts is None:
            corpus_path = _SAE_ROOT / "corpus_texts.txt"
            if corpus_path.exists():
                raw = corpus_path.read_text(encoding="utf-8")
                corpus_texts = [t.strip() for t in raw.split("\n\n") if t.strip()]
            else:
                corpus_texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning enables computers to learn from data.",
                    "Neural networks have transformed natural language processing.",
                ]

        cache_payload = {
            "model_id": model_id,
            "feature_idx": int(feature_idx),
            "corpus_texts": corpus_texts,
            "labeling_config": labeling_config or {},
            "backend": "groq",
        }
        cache_path = self._cache_path("label_feature", cache_payload)
        cached = self._load_json_cache(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        from analyzers.llm_analysis import FeatureLabeler, LabelingConfig, LabelResult, build_token_maps

        all_acts: List[torch.Tensor] = []
        for sentence in corpus_texts:
            token_data = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=min(128, max(1, int(meta.context_size))),
                add_special_tokens=False,
            )
            input_ids = token_data["input_ids"].to(self._device)
            if input_ids.numel() == 0:
                continue
            with torch.no_grad():
                _, cache = model.run_with_cache(input_ids, prepend_bos=False)
                hook_acts = cache[meta.hook_name][0].detach().cpu()
                all_acts.append(hook_acts)

        if not all_acts:
            raise ValueError("No activations could be collected from the provided corpus_texts")

        activations = torch.cat(all_acts, dim=0)
        token_ids, doc_map, pos_map = build_token_maps(corpus_texts, tokenizer, max_length=128)

        resolved_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        cfg_kwargs: Dict[str, Any] = {
            "backend": "groq",
            "model": "llama-3.3-70b-versatile",
            "top_k": max(1, len(corpus_texts)),
            "request_delay": 0.2,
            "groq_api_key": resolved_api_key,
        }
        if labeling_config:
            cfg_kwargs.update(labeling_config)
        cfg = LabelingConfig(**cfg_kwargs)

        labeler = FeatureLabeler(sae=sae, tokenizer=tokenizer, cfg=cfg, device=self._device)
        result: LabelResult = labeler.label_feature_from_activations(
            feature_idx=feature_idx,
            activations=activations,
            token_ids=token_ids,
            token_doc_map=doc_map,
            token_pos_map=pos_map,
        )

        response = {
            "feature_idx": result.feature_idx,
            "label": result.label,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "top_tokens": result.top_tokens,
            "error": result.error,
            "cache_hit": False,
        }
        self._save_json_cache(cache_path, response)
        return response

    def find_feature_activations(
        self,
        model_id: str,
        feature_id: int,
        dataset_name: str = "openwebtext",
        dataset_config: Optional[str] = None,
        split: str = "train",
        max_sentences: int = 200,
        max_results: int = 100,
        min_activation: float = 0.0,
        text_field: Optional[str] = None,
        max_length: int = 128,
        seed: int = 0,
    ) -> Dict[str, Any]:
        bundle = self._store.get_bundle(model_id)
        sae = bundle.sae
        model = bundle.model
        tokenizer = bundle.tokenizer
        meta = bundle.metadata

        if feature_id < 0 or feature_id >= meta.d_hidden:
            raise ValueError(f"feature_id must be in [0, {meta.d_hidden - 1}]")

        cache_payload = {
            "model_id": model_id,
            "feature_id": int(feature_id),
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "max_sentences": int(max_sentences),
            "max_results": int(max_results),
            "min_activation": float(min_activation),
            "text_field": text_field,
            "max_length": int(max_length),
            "seed": int(seed),
        }
        cache_path = self._cache_path("feature_activations", cache_payload)
        cached = self._load_json_cache(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        from datasets import Audio, load_dataset

        hf_cache_dir = os.environ.get("HF_DATASETS_CACHE")
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
            cache_dir=hf_cache_dir,
        )

        try:
            features = getattr(ds, "features", None) or {}
            for col, feat in features.items():
                if isinstance(feat, Audio):
                    ds = ds.cast_column(col, Audio(decode=False))
        except Exception:
            pass

        try:
            ds = ds.shuffle(seed=seed, buffer_size=max(1000, max_sentences * 4))
        except Exception:
            pass

        candidate_fields = [text_field, "text", "sentence", "transcript", "content", "article", "document"]
        candidate_fields = [f for f in candidate_fields if f]

        available_cols = list((getattr(ds, "features", None) or {}).keys())
        selected_text_fields = [f for f in candidate_fields if f in available_cols]
        if selected_text_fields:
            try:
                ds = ds.select_columns(selected_text_fields)
            except Exception:
                pass

        texts: List[str] = []
        for example in ds:
            value = ""
            for field in candidate_fields:
                raw = example.get(field)
                if isinstance(raw, str) and raw.strip():
                    value = raw.strip()
                    break
            if value:
                texts.append(value)
            if len(texts) >= max_sentences:
                break

        hits: List[Dict[str, Any]] = []
        total_tokens_scanned = 0

        with torch.no_grad():
            for sent_idx, sentence in enumerate(texts):
                token_data = tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(max_length, max(1, int(meta.context_size))),
                    add_special_tokens=False,
                )
                input_ids = token_data["input_ids"].to(self._device)
                token_id_list = input_ids[0].tolist()
                if not token_id_list:
                    continue

                _, cache = model.run_with_cache(input_ids, prepend_bos=False)
                hook_acts = cache[meta.hook_name]  # (1, pos, d_model)
                feature_mat = sae.encode(hook_acts)[0].detach().cpu()

                feat_col = feature_mat[:, feature_id]
                active_positions = torch.nonzero(feat_col > float(min_activation), as_tuple=False).flatten().tolist()
                if not active_positions:
                    total_tokens_scanned += len(token_id_list)
                    continue

                decoded_tokens = [tokenizer.decode([tid]) for tid in token_id_list]

                for pos in active_positions:
                    left = "".join(decoded_tokens[max(0, pos - 6):pos])
                    right = "".join(decoded_tokens[pos + 1:min(len(decoded_tokens), pos + 7)])
                    hits.append(
                        {
                            "sentence_index": sent_idx,
                            "sentence": sentence,
                            "token_index": int(pos),
                            "token": decoded_tokens[pos],
                            "activation": round(float(feat_col[pos].item()), 4),
                            "left_context": left,
                            "right_context": right,
                        }
                    )

                total_tokens_scanned += len(token_id_list)

        hits.sort(key=lambda x: x["activation"], reverse=True)
        limited_hits = hits[:max_results]

        response = {
            "model": model_id,
            "feature_id": feature_id,
            "feature_description": f"Feature {feature_id}",
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "text_fields_checked": selected_text_fields or candidate_fields,
            "scanned_sentences": len(texts),
            "scanned_tokens": total_tokens_scanned,
            "min_activation": min_activation,
            "max_results": max_results,
            "matches": limited_hits,
            "total_matches": len(hits),
            "cache_hit": False,
        }
        self._save_json_cache(cache_path, response)
        return response


_instance = SAEAnalyzer()
register(_instance)
