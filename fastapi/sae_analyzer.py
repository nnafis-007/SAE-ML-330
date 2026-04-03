"""
SAE Analyzer – connects the Sparse Autoencoder pipeline to the FastAPI backend.

Uses:
    - ``sae_model.SparseAutoencoder``   for encoding activations
    - ``data_collection.GPT2ActivationCollector`` for extracting GPT-2 hidden states
    - ``analyzers.llm_analysis.FeatureLabeler`` for on-demand LLM-based feature labeling
    - ``analyzers.llm_analysis.build_token_maps`` for corpus token maps

All heavy objects (GPT-2 model, SAE checkpoints, collectors) are lazily loaded
and cached so repeated requests are fast.
"""

from __future__ import annotations

import json
import os
import sys
import time
import math
import uuid
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Make the SAE source package importable
# ---------------------------------------------------------------------------
_SAE_ROOT = Path(__file__).resolve().parent.parent
_SAE_SRC = _SAE_ROOT / "src"
if str(_SAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAE_ROOT))
if str(_SAE_SRC) not in sys.path:
    sys.path.insert(0, str(_SAE_SRC))

# Heavy imports (transformers, GPT2, etc.) are deferred to first use
# to keep server startup fast.  Only lightweight imports happen here.
from sae_model import SparseAutoencoder                                # noqa: E402

from analyzers import BaseAnalyzer, register                           # noqa: E402

# ---------------------------------------------------------------------------
# Checkpoints directory
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = _SAE_ROOT / "checkpoints"
DEFAULT_CORPUS_PATH = _SAE_ROOT / "corpus.txt"
CACHE_DIR = _SAE_ROOT / ".cache"

# Keep activation normalization consistent with LLM labeling path.
_ACTIVATION_STD_FLOOR = 1e-3


def _move_module_to_device(module: torch.nn.Module, device: str) -> torch.nn.Module:
    """Move *module* to *device*, handling meta-initialized parameters safely."""
    target = torch.device(device)
    has_meta = any(p.is_meta for p in module.parameters()) or any(b.is_meta for b in module.buffers())
    if has_meta:
        return module.to_empty(device=target)
    return module.to(target)


# ---------------------------------------------------------------------------
# SAEAnalyzer
# ---------------------------------------------------------------------------

class SAEAnalyzer(BaseAnalyzer):
    """
    Analyzer backed by a Sparse Autoencoder trained on GPT-2 activations.

    * ``list_models`` scans the checkpoints folder for .pt files.
    * ``analyze`` tokenises the input text with GPT-2, collects hidden-state
      activations from the layer the SAE was trained on, encodes them through
      the SAE, and returns per-token top-K feature activations.
    * ``label_feature`` optionally calls an LLM (via ``analyzers/llm_analysis.py``) to
      generate a human-readable label for a single feature.
    """

    def __init__(self, checkpoints_dir: str | Path = CHECKPOINTS_DIR):
        self._checkpoints_dir = Path(checkpoints_dir)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Caches (populated lazily)
        self._sae_cache: Dict[str, Dict[str, Any]] = {}
        self._collector_cache: Dict[int, Any] = {}
        self._labels_cache: Dict[str, Dict[int, Dict[str, str]]] = {}
        self._model_list_cache: Optional[List[Dict[str, Any]]] = None
        self._feature_activation_cache: Dict[str, Dict[str, Any]] = {}
        self._text_corpus_cache: Dict[str, List[str]] = {}
        self._cache_dir = CACHE_DIR
        self._corpus_cache_dir = self._cache_dir / "dataset_texts"
        self._activation_cache_dir = self._cache_dir / "feature_label_inputs"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_cache_dir.mkdir(parents=True, exist_ok=True)
        self._activation_cache_dir.mkdir(parents=True, exist_ok=True)

    # -- BaseAnalyzer interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "sae"

    def list_models(self) -> List[Dict[str, Any]]:
        """Return cached checkpoint list (scanned at startup)."""
        if self._model_list_cache is None:
            self._model_list_cache = self._scan_checkpoints()
        return self._model_list_cache

    def _scan_checkpoints(self) -> List[Dict[str, Any]]:
        """Walk *checkpoints_dir* and read metadata from each .pt file."""
        models: List[Dict[str, Any]] = []
        for pt_file in sorted(self._checkpoints_dir.rglob("*.pt")):
            rel = pt_file.relative_to(self._checkpoints_dir)
            try:
                meta = self._peek_checkpoint(str(pt_file))
                models.append({"id": str(rel), "name": str(rel), **meta})
            except Exception as exc:
                models.append({
                    "id": str(rel),
                    "name": str(rel),
                    "error": str(exc),
                })
        return models

    def analyze(
        self,
        text: str,
        model_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the full SAE analysis pipeline on *text*.

        Returns a dict compatible with the frontend::

            {
                "model": "<model_id>",
                "layer_index": 8,
                "d_hidden": 12288,
                "tokens": [
                    {
                        "text": " word",
                        "features": [
                            {"id": 42, "activation": 5.31, "description": "Feature 42"},
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text is empty. Please provide non-empty text for analysis.")

        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]
        d_hidden: int = int(info.get("d_hidden", 0))
        if d_hidden <= 0:
            raise ValueError(f"Selected SAE checkpoint has invalid hidden size (d_hidden={d_hidden}).")
        top_k = max(1, int(top_k))

        collector = self._get_collector(layer_index)
        tokenizer = collector.tokenizer

        # ---- tokenise input --------------------------------------------------
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(collector.device)
        token_id_list = input_ids[0].tolist()
        if not token_id_list:
            raise ValueError("Tokenizer produced no tokens from input text. Provide at least one non-whitespace character.")

        # ---- GPT-2 hidden states ---------------------------------------------
        with torch.no_grad():
            outputs = collector.model(input_ids)
            # hidden_states: (embedding, layer0, layer1, …)
            hidden_states = outputs.hidden_states[layer_index + 1]
            activations = hidden_states[0]           # (seq_len, d_model)
            activations = self._standardize_hidden_states(activations)

        # ---- SAE encode ------------------------------------------------------
        with torch.no_grad():
            features = sae.encode(activations.to(self._device))  # (seq_len, d_hidden)

        # ---- pre-computed labels (if any) ------------------------------------
        labels = self._load_feature_labels(checkpoint_path)

        # ---- build per-token response ----------------------------------------
        tokens_data: List[Dict[str, Any]] = []
        for i, tid in enumerate(token_id_list):
            token_str = tokenizer.decode([tid])
            
            feat_vec = features[i].cpu()
            active = int((feat_vec > 0).sum().item())
            k = min(top_k, max(active, 1))
            top_vals, top_idxs = torch.topk(feat_vec, k)

            feat_list: List[Dict[str, Any]] = []
            for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
                if val > 0:
                    feat_id = int(idx)
                    feat_list.append({
                        "id": feat_id,
                        "activation": round(val, 4),
                        "name": labels.get(feat_id, {}).get("name", f"Feature {feat_id}"),
                        "description": labels.get(feat_id, {}).get("description", f"Feature {feat_id}"),
                    })

            tokens_data.append({"text": token_str, "features": feat_list})

        return {
            "model": model_id,
            "layer_index": layer_index,
            "d_hidden": d_hidden,
            "tokens": tokens_data,
        }

    def trace_feature_in_sentence(
        self,
        text: str,
        model_id: str,
        feature_id: int,
        min_activation: float = 0.0,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """Return per-token activation values for one SAE feature on one sentence."""
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]
        d_hidden: int = int(info["d_hidden"])

        if feature_id < 0 or feature_id >= d_hidden:
            raise ValueError(f"feature_id must be in [0, {d_hidden - 1}]")

        collector = self._get_collector(layer_index)
        tokenizer = collector.tokenizer
        labels = self._load_feature_labels(checkpoint_path)

        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(collector.device)
        token_id_list = input_ids[0].tolist()

        with torch.no_grad():
            outputs = collector.model(input_ids)
            hidden_states = outputs.hidden_states[layer_index + 1]
            activations = hidden_states[0]
            activations = self._standardize_hidden_states(activations)
            feature_values = sae.encode(activations.to(self._device))[:, feature_id].cpu()

        token_rows: List[Dict[str, Any]] = []
        max_activation = 0.0
        active_count = 0
        for i, tid in enumerate(token_id_list):
            token_text = tokenizer.decode([tid])
            activation = float(feature_values[i].item()) if i < feature_values.shape[0] else 0.0
            is_active = activation > float(min_activation)
            if is_active:
                active_count += 1
                max_activation = max(max_activation, activation)

            token_rows.append({
                "index": i,
                "text": token_text,
                "activation": round(activation, 6),
                "is_active": is_active,
            })

        return {
            "model": model_id,
            "layer_index": layer_index,
            "d_hidden": d_hidden,
            "feature_id": feature_id,
            "feature_name": labels.get(feature_id, {}).get("name", f"Feature {feature_id}"),
            "feature_description": labels.get(feature_id, {}).get("description", f"Feature {feature_id}"),
            "active_token_count": active_count,
            "token_count": len(token_rows),
            "max_activation": round(max_activation, 6),
            "min_activation": float(min_activation),
            "tokens": token_rows,
        }

    def get_feature_info(
        self,
        model_id: str,
        feature_id: int,
    ) -> Dict[str, Any]:
        """Return feature label/description metadata without scanning sentences."""
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        d_hidden: int = int(info["d_hidden"])
        layer_index: int = int(info["layer_index"])

        if feature_id < 0 or feature_id >= d_hidden:
            raise ValueError(f"feature_id must be in [0, {d_hidden - 1}]")

        labels = self._load_feature_labels(checkpoint_path)
        feature_meta = labels.get(feature_id, {})
        return {
            "model": model_id,
            "layer_index": layer_index,
            "d_hidden": d_hidden,
            "feature_id": int(feature_id),
            "feature_name": feature_meta.get("name", f"Feature {feature_id}"),
            "feature_description": feature_meta.get("description", f"Feature {feature_id}"),
        }

    # -- On-demand feature labeling (uses analyzers.llm_analysis.FeatureLabeler) --------

    def label_feature(
        self,
        model_id: str,
        feature_idx: int,
        corpus_texts: Optional[List[str]] = None,
        corpus_path: Optional[str] = None,
        labeling_config: Optional[Dict[str, Any]] = None,
        groq_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Label a single SAE feature using an LLM via
        ``analyzers.llm_analysis.FeatureLabeler``.

        Parameters
        ----------
        model_id : str
            Relative checkpoint path (e.g. ``"ghost/best_model.pt"``).
        feature_idx : int
            Index of the feature to label.
        corpus_texts : list[str] | None
            Texts to scan for activating examples.  Falls back to corpus_texts.txt.
        labeling_config : dict | None
            Overrides for ``LabelingConfig`` fields (backend, model, top_k …).
        groq_api_key : str | None
            Groq API key. Falls back to the ``GROQ_API_KEY`` environment variable.

        Returns
        -------
        dict with label, explanation, confidence, top_tokens …
        """
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]

        collector = self._get_collector(layer_index)
        tokenizer = collector.tokenizer

        # Corpus
        if corpus_texts is None:
            sentence_budget = 200
            if labeling_config:
                sentence_budget = max(1, int(labeling_config.get("num_sentences", sentence_budget)))

        corpus_path = corpus_path or str(DEFAULT_CORPUS_PATH)
        corpus_texts, activations, token_ids, doc_map, pos_map, source_signature = self._get_cached_label_inputs(
            model_id=model_id,
            layer_index=layer_index,
            collector=collector,
            tokenizer=tokenizer,
            corpus_texts=corpus_texts,
            corpus_path=corpus_path,
            sentence_budget=max(1, len(corpus_texts)) if corpus_texts else sentence_budget,
            max_length=128,
            request_tag="label_feature",
        )

        print(
            "[SAEAnalyzer] label_feature corpus source | "
            f"path={corpus_path} collected_sentences={len(corpus_texts)} signature={source_signature[:16]}"
        )

        # Labeling config (lazy imports)
        from analyzers.llm_analysis import FeatureLabeler, LabelingConfig, LabelResult  # noqa: E811

        resolved_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        request_id = str(uuid.uuid4())
        cfg_kwargs: Dict[str, Any] = {
            "backend": "groq",
            "model": "llama-3.3-70b-versatile",
            "top_k": min(15, activations.shape[0]),
            "request_delay": 0.2,
            "groq_api_key": resolved_api_key,
            "skip_first_token": True,
            "request_id": request_id,
        }
        if labeling_config:
            # Ignore unsupported keys (e.g. dataset_name) so UI can pass
            # transport metadata without breaking LabelingConfig construction.
            allowed = set(LabelingConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in labeling_config.items() if k in allowed}
            ignored = [k for k in labeling_config.keys() if k not in allowed]
            if ignored:
                print(f"[SAEAnalyzer] Ignoring unsupported labeling_config keys: {ignored}")
            cfg_kwargs.update(filtered)

        # Enforce one activation scale across SAE + LLM paths for UI consistency.
        cfg_kwargs["normalize_mode"] = "standardize"
        cfg = LabelingConfig(**cfg_kwargs)

        # Guardrail for API responsiveness: unbounded context prompts can become
        # very large and slow (or hit provider limits) when top_k <= 0.
        if int(getattr(cfg, "top_k", 0) or 0) <= 0:
            cfg.top_k = min(40, int(activations.shape[0]))

        labeler = FeatureLabeler(sae=sae, tokenizer=tokenizer, cfg=cfg, device=self._device)

        result: LabelResult = labeler.label_feature_from_activations(
            feature_idx=feature_idx,
            activations=activations,
            token_ids=token_ids,
            token_doc_map=doc_map,
            token_pos_map=pos_map,
        )

        # Persist runtime label updates in a model-specific file so edits can
        # be tracked per checkpoint and immediately reflected in subsequent UI calls.
        response_payload = {
            "request_id": request_id,
            "feature_idx": result.feature_idx,
            "label": result.label,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "top_tokens": result.top_tokens,
            "llm_activation_mode": cfg.normalize_mode,
            "llm_prompt_examples": [
                {
                    "token": ctx.token,
                    "context": ctx.context,
                    "activation": round(float(ctx.activation_value), 4),
                }
                for ctx in result.top_contexts
            ],
            "error": result.error,
        }

        self._save_runtime_feature_label(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            feature_idx=int(result.feature_idx),
            label_result=response_payload,
        )

        return response_payload

    def find_feature_activations(
        self,
        model_id: str,
        feature_id: int,
        dataset_name: Optional[str] = None,
        dataset_config: str = "local",
        split: str = "local",
        corpus_path: Optional[str] = None,
        max_sentences: Optional[int] = None,
        target_activating_examples: Optional[int] = None,
        page: int = 1,
        page_size: int = 25,
        min_activation: float = 0.0,
        text_field: Optional[str] = None,
        max_length: int = 128,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """
        Find sentence/token examples where a given SAE feature activates.

        This scans a sampled subset of the requested dataset split, computes
        token-level SAE activations, and returns tokens whose value for the
        selected ``feature_id`` exceeds ``min_activation``.
        """
        start_time = time.time()
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]
        d_hidden: int = int(info["d_hidden"])

        normalization_mode = "standardize"
        corpus_path = corpus_path or str(DEFAULT_CORPUS_PATH)

        print(
            "[SAEAnalyzer] Feature activation lookup started | "
            f"model={model_id} feature_id={feature_id} layer={layer_index} "
            f"corpus_path={corpus_path} max_sentences={max_sentences} "
            f"target_activating_examples={target_activating_examples} page={page} page_size={page_size} "
            f"min_activation={min_activation} "
            f"normalization={normalization_mode} filter='activation > min_activation'"
        )

        page = max(1, int(page))
        page_size = max(1, int(page_size))
        if max_sentences is not None:
            max_sentences = max(1, int(max_sentences))
        if target_activating_examples is not None:
            target_activating_examples = max(1, int(target_activating_examples))

        sentence_budget = (
            max_sentences
            if max_sentences is not None
            else max(1000, int(target_activating_examples or 50) * 50)
        )
        source_signature = self._build_source_signature(
            corpus_path=corpus_path,
            sentence_budget=sentence_budget,
            max_length=max_length,
            model_id=model_id,
            layer_index=layer_index,
            request_tag="feature_activations",
        )

        cache_key = json.dumps(
            {
                "model_id": model_id,
                "feature_id": feature_id,
                "source_signature": source_signature,
                "max_sentences": max_sentences,
                "target_activating_examples": target_activating_examples,
                "min_activation": float(min_activation),
                "text_field": text_field,
                "max_length": int(max_length),
                "seed": int(seed),
            },
            sort_keys=True,
        )

        if cache_key in self._feature_activation_cache:
            cached = self._feature_activation_cache[cache_key]
            hits = cached["hits"]
            total_matches = len(hits)
            total_pages = max(1, math.ceil(total_matches / page_size)) if total_matches > 0 else 1
            page = min(page, total_pages)
            start = (page - 1) * page_size
            end = start + page_size
            paged_hits = hits[start:end]
            print(
                "[SAEAnalyzer] Feature activation cache hit | "
                f"feature_id={feature_id} total_matches={total_matches} page={page}/{total_pages}"
            )
            return {
                "model": model_id,
                "feature_id": feature_id,
                "feature_description": cached["feature_description"],
                "dataset": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
                "text_fields_checked": cached["text_fields_checked"],
                "scanned_sentences": cached["scanned_sentences"],
                "raw_examples_scanned": cached["raw_examples_scanned"],
                "scanned_tokens": cached["scanned_tokens"],
                "min_activation": min_activation,
                "activation_filter": "activation > min_activation",
                "normalization_mode": normalization_mode,
                "target_activating_examples": target_activating_examples,
                "activating_sentences": cached["activating_sentences"],
                "activating_sentences_count": len(cached["activating_sentences"]),
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next_page": page < total_pages,
                "has_prev_page": page > 1,
                "matches": paged_hits,
                "total_matches": total_matches,
                "source_signature": source_signature,
                "corpus_path": corpus_path,
            }

        if feature_id < 0 or feature_id >= d_hidden:
            raise ValueError(f"feature_id must be in [0, {d_hidden - 1}]")

        collector = self._get_collector(layer_index)
        tokenizer = collector.tokenizer
        labels = self._load_feature_labels(checkpoint_path)

        corpus_texts = self._load_local_corpus_texts(
            corpus_path=corpus_path,
            sentence_budget=sentence_budget,
        )
        selected_text_fields = ["indexed_corpus"]

        texts_scanned = 0
        examples_scanned = 0
        hits: List[Dict[str, Any]] = []
        activating_sentences: List[str] = []
        activating_seen: set = set()
        total_tokens_scanned = 0
        progress_interval = 25
        inference_batch_size = 12 if self._device == "cuda" else 6

        def _process_sentence_batch(batch_sentences: List[str]) -> bool:
            """Encode a sentence batch and return True when stop conditions are met."""
            nonlocal texts_scanned, total_tokens_scanned, hits, activating_sentences, activating_seen

            if not batch_sentences:
                return False

            enc = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(collector.device)

            outputs = collector.model(**enc)
            hidden_batch = outputs.hidden_states[layer_index + 1]
            input_ids_batch = enc["input_ids"]
            attention_mask_batch = enc["attention_mask"]

            for row_idx, sentence in enumerate(batch_sentences):
                token_mask = attention_mask_batch[row_idx].bool()
                token_ids = input_ids_batch[row_idx][token_mask].tolist()
                if not token_ids:
                    continue

                texts_scanned += 1

                hidden_states = hidden_batch[row_idx][token_mask]
                hidden_states = self._standardize_hidden_states(hidden_states)
                feature_mat = sae.encode(hidden_states.to(self._device)).cpu()

                feat_col = feature_mat[:, feature_id]
                active_positions = torch.nonzero(
                    feat_col > float(min_activation), as_tuple=False
                ).flatten().tolist()
                # Skip first token to avoid GPT-2 attention sink effects.
                active_positions = [p for p in active_positions if p > 0]

                if active_positions:
                    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

                    if sentence not in activating_seen:
                        activating_seen.add(sentence)
                        activating_sentences.append(sentence)

                    for pos in active_positions:
                        token_text = decoded_tokens[pos]
                        left = "".join(decoded_tokens[max(0, pos - 6):pos])
                        right = "".join(decoded_tokens[pos + 1:min(len(decoded_tokens), pos + 7)])
                        hits.append({
                            "sentence_index": texts_scanned - 1,
                            "sentence": sentence,
                            "token_index": int(pos),
                            "token": token_text,
                            "activation": round(float(feat_col[pos].item()), 4),
                            "left_context": left,
                            "right_context": right,
                        })

                total_tokens_scanned += len(token_ids)

                if texts_scanned % progress_interval == 0:
                    print(
                        "[SAEAnalyzer] Progress | "
                        f"processed_sentences={texts_scanned} "
                        f"tokens_scanned={total_tokens_scanned} matches_so_far={len(hits)}"
                    )

                if max_sentences is not None and texts_scanned >= max_sentences:
                    return True

                if max_sentences is None and target_activating_examples is not None:
                    if len(activating_sentences) >= target_activating_examples:
                        return True

            return False

        with torch.no_grad():
            sentence_batch: List[str] = []
            should_stop = False
            for sentence in corpus_texts:
                examples_scanned += 1
                if not sentence:
                    continue

                sentence_batch.append(sentence)
                if len(sentence_batch) >= inference_batch_size:
                    should_stop = _process_sentence_batch(sentence_batch)
                    sentence_batch = []
                    if should_stop:
                        break

            if not should_stop and sentence_batch:
                _process_sentence_batch(sentence_batch)

        hits.sort(key=lambda x: x["activation"], reverse=True)
        total_matches = len(hits)
        total_pages = max(1, math.ceil(total_matches / page_size)) if total_matches > 0 else 1
        page = min(page, total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        paged_hits = hits[start:end]

        elapsed = time.time() - start_time
        print(
            "[SAEAnalyzer] Feature activation lookup finished | "
            f"sentences_collected={texts_scanned} raw_examples_scanned={examples_scanned} "
            f"tokens_scanned={total_tokens_scanned} total_matches={total_matches} shown={len(paged_hits)} "
            f"elapsed_sec={elapsed:.2f}"
        )

        # Keep a small cache of full-hit results so page navigation is fast.
        if len(self._feature_activation_cache) >= 12:
            oldest_key = next(iter(self._feature_activation_cache.keys()))
            self._feature_activation_cache.pop(oldest_key, None)
        self._feature_activation_cache[cache_key] = {
            "hits": hits,
            "feature_description": labels.get(feature_id, {}).get("name", f"Feature {feature_id}"),
            "text_fields_checked": selected_text_fields or candidate_fields,
            "scanned_sentences": texts_scanned,
            "raw_examples_scanned": examples_scanned,
            "scanned_tokens": total_tokens_scanned,
            "activating_sentences": activating_sentences,
        }

        return {
            "model": model_id,
            "feature_id": feature_id,
            "feature_description": labels.get(feature_id, {}).get("name", f"Feature {feature_id}"),
            "dataset": dataset_name or "local_corpus",
            "dataset_config": dataset_config,
            "split": split,
            "text_fields_checked": selected_text_fields,
            "scanned_sentences": texts_scanned,
            "raw_examples_scanned": examples_scanned,
            "scanned_tokens": total_tokens_scanned,
            "min_activation": min_activation,
            "activation_filter": "activation > min_activation",
            "normalization_mode": normalization_mode,
            "target_activating_examples": target_activating_examples,
            "activating_sentences": activating_sentences,
            "activating_sentences_count": len(activating_sentences),
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next_page": page < total_pages,
            "has_prev_page": page > 1,
            "matches": paged_hits,
            "total_matches": total_matches,
            "source_signature": source_signature,
            "corpus_path": corpus_path,
        }

    def bulk_label_features(
        self,
        model_id: str,
        feature_start: Optional[int] = None,
        feature_end: Optional[int] = None,
        feature_ids: Optional[List[int]] = None,
        num_sentences: int = 200,
        llm_top_k: int = 25,
        min_activation: float = 0.0,
        corpus_path: Optional[str] = None,
        labeling_config: Optional[Dict[str, Any]] = None,
        groq_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bulk-label a contiguous feature range using one LLM-analysis run."""
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = int(info["layer_index"])
        d_hidden: int = int(info["d_hidden"])

        if feature_ids:
            normalized_feature_ids: List[int] = []
            seen = set()
            for raw_id in feature_ids:
                feat_id = int(raw_id)
                if feat_id < 0:
                    raise ValueError("feature_ids must contain only non-negative integers")
                if feat_id >= d_hidden:
                    raise ValueError(f"feature id {feat_id} must be <= {d_hidden - 1}")
                if feat_id not in seen:
                    normalized_feature_ids.append(feat_id)
                    seen.add(feat_id)
            if not normalized_feature_ids:
                raise ValueError("feature_ids must contain at least one valid feature id")
            feature_indices = normalized_feature_ids
            resolved_feature_start = min(feature_indices)
            resolved_feature_end = max(feature_indices)
        else:
            if feature_start is None or feature_end is None:
                raise ValueError("Provide either feature_ids or both feature_start and feature_end")
            feature_start = int(feature_start)
            feature_end = int(feature_end)
            if feature_start < 0 or feature_end < 0 or feature_end < feature_start:
                raise ValueError("feature range must satisfy 0 <= start <= end")
            if feature_end >= d_hidden:
                raise ValueError(f"feature_end must be <= {d_hidden - 1}")
            feature_indices = list(range(int(feature_start), int(feature_end) + 1))
            resolved_feature_start = int(feature_start)
            resolved_feature_end = int(feature_end)

        collector = self._get_collector(layer_index)
        tokenizer = collector.tokenizer

        corpus_path = corpus_path or str(DEFAULT_CORPUS_PATH)
        corpus_texts, activations, token_ids, doc_map, pos_map, source_signature = self._get_cached_label_inputs(
            model_id=model_id,
            layer_index=layer_index,
            collector=collector,
            tokenizer=tokenizer,
            corpus_texts=None,
            corpus_path=corpus_path,
            sentence_budget=max(1, int(num_sentences)),
            max_length=128,
            request_tag="bulk_label_features",
        )
        from analyzers.llm_analysis import FeatureLabeler, LabelingConfig

        print(
            "[SAEAnalyzer] bulk_label_features corpus source | "
            f"path={corpus_path} collected_sentences={len(corpus_texts)} signature={source_signature[:16]}"
        )

        resolved_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        request_id = str(uuid.uuid4())
        cfg_kwargs: Dict[str, Any] = {
            "backend": "groq",
            "model": "llama-3.3-70b-versatile",
            "top_k": max(1, int(llm_top_k)),
            "request_delay": 0.2,
            "groq_api_key": resolved_api_key,
            "skip_first_token": True,
            "request_id": request_id,
            "min_activation": float(max(0.0, min_activation)),
            "normalize_mode": "standardize",
        }
        if labeling_config:
            allowed = set(LabelingConfig.__dataclass_fields__.keys())
            cfg_kwargs.update({k: v for k, v in labeling_config.items() if k in allowed})

        cfg = LabelingConfig(**cfg_kwargs)
        labeler = FeatureLabeler(sae=sae, tokenizer=tokenizer, cfg=cfg, device=self._device)

        results = labeler.label_features_from_activations(
            feature_indices=feature_indices,
            activations=activations,
            token_ids=token_ids,
            token_doc_map=doc_map,
            token_pos_map=pos_map,
            save_path=None,
            resume=False,
        )

        labeled: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for feat_idx in feature_indices:
            r = results.get(feat_idx)
            if r is None:
                failed.append({"feature_id": feat_idx, "error": "No result returned"})
                continue

            response_payload = {
                "request_id": request_id,
                "feature_idx": int(r.feature_idx),
                "label": r.label,
                "explanation": r.explanation,
                "confidence": r.confidence,
                "top_tokens": r.top_tokens,
                "llm_activation_mode": cfg.normalize_mode,
                "llm_prompt_examples": [
                    {
                        "token": ctx.token,
                        "context": ctx.context,
                        "activation": round(float(ctx.activation_value), 4),
                    }
                    for ctx in r.top_contexts
                ],
                "error": r.error,
            }

            # Persist each feature label to model-specific JSON for runtime use.
            self._save_runtime_feature_label(
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                feature_idx=int(r.feature_idx),
                label_result=response_payload,
            )

            if r.error:
                msg = str(r.error)
                if "no activating examples" in msg.lower():
                    skipped.append({"feature_id": feat_idx, "reason": msg})
                else:
                    failed.append({"feature_id": feat_idx, "error": msg})
            else:
                labeled.append({
                    "feature_id": feat_idx,
                    "label": r.label,
                    "confidence": r.confidence,
                    "request_id": request_id,
                    "sent_to_llm": int(cfg.top_k),
                })

        return {
            "model_id": model_id,
            "feature_start": int(resolved_feature_start),
            "feature_end": int(resolved_feature_end),
            "feature_ids": feature_indices,
            "request_id": request_id,
            "num_sentences": int(num_sentences),
            "sentences_used": len(corpus_texts),
            "llm_top_k": int(cfg.top_k),
            "min_activation": float(cfg.min_activation),
            "source_signature": source_signature,
            "corpus_path": corpus_path,
            "labeled": labeled,
            "skipped": skipped,
            "failed": failed,
        }

    @staticmethod
    def _standardize_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
        """Standardize hidden states over token rows to keep activation scale consistent."""
        mean = hidden_states.mean(dim=0, keepdim=True)
        std = hidden_states.std(dim=0, keepdim=True)
        std = std.clamp_min(_ACTIVATION_STD_FLOOR)
        return (hidden_states - mean) / (std + 1e-8)

    def _build_source_signature(
        self,
        corpus_path: str,
        sentence_budget: int,
        max_length: int,
        model_id: str,
        layer_index: int,
        request_tag: str,
    ) -> str:
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Local corpus file not found: {corpus_file}")
        stats = corpus_file.stat()
        signature = (
            f"corpus|{corpus_file.resolve()}|size={stats.st_size}|mtime_ns={stats.st_mtime_ns}|"
            f"budget={int(sentence_budget)}|max_length={int(max_length)}|model={model_id}|"
            f"layer={int(layer_index)}|tag={request_tag}"
        )
        return signature

    def _save_texts_jsonl(self, texts: List[str], source_signature: str) -> str:
        digest = hashlib.blake2b(source_signature.encode("utf-8"), digest_size=8).hexdigest()
        out_path = self._cache_dir / f"hf_texts_{digest}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for idx, text in enumerate(texts):
                row = {"idx": idx, "text": text}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(out_path)

    def _load_local_corpus_texts(
        self,
        corpus_path: str,
        sentence_budget: int,
    ) -> List[str]:
        """Load local indexed corpus and reuse from memory/disk cache."""
        sentence_budget = max(1, int(sentence_budget))
        source_signature = self._build_source_signature(
            corpus_path=corpus_path,
            sentence_budget=sentence_budget,
            max_length=128,
            model_id="local_corpus",
            layer_index=-1,
            request_tag="corpus_texts",
        )
        key_base = source_signature
        mem_key = f"{key_base}|budget={sentence_budget}"

        cached = self._text_corpus_cache.get(mem_key)
        if cached is not None:
            return cached[:sentence_budget]

        digest = hashlib.sha1(key_base.encode("utf-8")).hexdigest()[:16]
        disk_cache_path = self._corpus_cache_dir / f"{digest}.json"

        texts: List[str] = []
        if disk_cache_path.exists():
            try:
                payload = json.loads(disk_cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("texts"), list):
                    texts = [str(t) for t in payload["texts"] if isinstance(t, str) and t.strip()]
                if len(texts) >= sentence_budget:
                    self._text_corpus_cache[mem_key] = texts[:sentence_budget]
                    print(
                        "[SAEAnalyzer] Local corpus text cache hit (disk) | "
                        f"path={corpus_path} texts={len(texts[:sentence_budget])}"
                    )
                    return texts[:sentence_budget]
            except Exception:
                texts = []

        from analyzers.llm_analysis import parse_indexed_corpus  # lazy import
        texts = parse_indexed_corpus(
            corpus_path=corpus_path,
            max_entries=sentence_budget,
            skip_header=True,
        )

        if texts:
            try:
                disk_cache_path.write_text(
                    json.dumps({"texts": texts, "source_signature": source_signature}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

        self._text_corpus_cache[mem_key] = texts[:sentence_budget]
        print(
            "[SAEAnalyzer] Local corpus text cache ready | "
            f"path={corpus_path} texts={len(texts[:sentence_budget])}"
        )
        return texts[:sentence_budget]

    def _get_cached_label_inputs(
        self,
        model_id: str,
        layer_index: int,
        collector: Any,
        tokenizer: Any,
        corpus_texts: Optional[List[str]],
        corpus_path: str,
        sentence_budget: int,
        max_length: int,
        request_tag: str,
    ) -> tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Build or load cached activations/token maps with source-signature validation."""
        from analyzers.llm_analysis import build_token_maps  # lazy import

        sentence_budget = max(1, int(sentence_budget))
        max_length = max(1, int(max_length))

        if corpus_texts is None:
            corpus_texts = self._load_local_corpus_texts(
                corpus_path=corpus_path,
                sentence_budget=sentence_budget,
            )
            source_signature = self._build_source_signature(
                corpus_path=corpus_path,
                sentence_budget=sentence_budget,
                max_length=max_length,
                model_id=model_id,
                layer_index=layer_index,
                request_tag=request_tag,
            )
        else:
            corpus_texts = [str(t).strip() for t in corpus_texts if isinstance(t, str) and str(t).strip()]
            explicit_hash = hashlib.blake2b(
                "\n".join(corpus_texts).encode("utf-8"), digest_size=12
            ).hexdigest()
            source_signature = (
                f"inline_texts|count={len(corpus_texts)}|hash={explicit_hash}|"
                f"max_length={max_length}|model={model_id}|layer={layer_index}|tag={request_tag}"
            )

        digest = hashlib.blake2b(source_signature.encode("utf-8"), digest_size=8).hexdigest()
        cache_path = self._activation_cache_dir / f"label_inputs_{digest}.pt"

        if cache_path.exists():
            try:
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                cached_signature = cache.get("source_signature")
                if cached_signature == source_signature:
                    print(
                        "[SAEAnalyzer] Label input cache hit | "
                        f"path={cache_path.name} signature={source_signature[:16]}"
                    )
                    return (
                        corpus_texts,
                        cache["activations"],
                        cache["token_ids"],
                        cache["token_doc_map"],
                        cache["token_pos_map"],
                        source_signature,
                    )
                print(
                    "[SAEAnalyzer] Label input cache signature mismatch | "
                    f"cached={str(cached_signature)[:16]} current={source_signature[:16]}"
                )
            except Exception as exc:
                print(f"[SAEAnalyzer] Failed to load label input cache ({cache_path.name}): {exc}")

        activations = collector.collect_activations(
            texts=corpus_texts,
            batch_size=8,
            max_length=max_length,
        )
        token_ids, token_doc_map, token_pos_map = build_token_maps(
            corpus_texts,
            tokenizer,
            max_length=max_length,
        )
        texts_jsonl_path = self._save_texts_jsonl(corpus_texts, source_signature)

        torch.save(
            {
                "version": 1,
                "source_signature": source_signature,
                "texts_jsonl_path": texts_jsonl_path,
                "activations": activations,
                "token_ids": token_ids,
                "token_doc_map": token_doc_map,
                "token_pos_map": token_pos_map,
            },
            cache_path,
        )
        print(
            "[SAEAnalyzer] Label input cache saved | "
            f"path={cache_path.name} signature={source_signature[:16]}"
        )
        return corpus_texts, activations, token_ids, token_doc_map, token_pos_map, source_signature

    # -- Internal helpers ------------------------------------------------------

    def _peek_checkpoint(self, path: str) -> Dict[str, Any]:
        """Extract metadata from a checkpoint *without* loading full tensors.

        Uses ``torch.load`` with ``weights_only=False`` but then only reads
        scalar metadata.  The heavy tensor data is memory-mapped by the OS
        and never actually materialised.
        """
        import pickle, struct, zipfile  # noqa: E401

        # Fast path: try to read only the hyperparameters dict from the
        # zip-format checkpoint that PyTorch >= 1.6 produces.  This avoids
        # deserialising the large W_enc / W_dec tensors entirely.
        d_model: Optional[int] = None
        d_hidden: Optional[int] = None
        l1_coeff: Any = "unknown"
        layer_index: int = 8

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return {"d_model": 768, "d_hidden": 12288, "l1_coeff": "unknown", "layer_index": 8}

        hp = payload.get("hyperparameters", {})
        if hp:
            d_model = hp.get("d_model")
            d_hidden = hp.get("d_hidden")
            l1_coeff = hp.get("l1_coeff", l1_coeff)
            layer_index = hp.get("layer_index", layer_index)

        # Fall back to inspecting tensor shapes if hp was incomplete
        if d_model is None or d_hidden is None:
            state = payload.get("model_state_dict", {})
            if "W_enc" in state:
                d_hidden = int(state["W_enc"].shape[0])
                d_model = int(state["W_enc"].shape[1])
            elif "W_enc" in payload:
                d_hidden = int(payload["W_enc"].shape[0])
                d_model = int(payload["W_enc"].shape[1])

        # Also grab l1_coeff from top-level keys (pruned models store it there)
        if l1_coeff == "unknown" and "l1_coeff" in payload:
            l1_coeff = payload["l1_coeff"]

        return {
            "d_model": d_model or 768,
            "d_hidden": d_hidden or 12288,
            "l1_coeff": l1_coeff,
            "layer_index": layer_index,
        }

    def _load_sae(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load (and cache) an SAE from a checkpoint file."""
        if checkpoint_path in self._sae_cache:
            return self._sae_cache[checkpoint_path]

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hp = payload.get("hyperparameters", {})
        state = payload.get("model_state_dict", None)

        # Some checkpoints store the state_dict at top level (e.g. pruned models)
        if state is None:
            state = {k: v for k, v in payload.items()
                     if isinstance(v, torch.Tensor)}

        d_model = int(state["W_enc"].shape[1])
        d_hidden = int(state["W_enc"].shape[0])
        l1_coeff = hp.get("l1_coeff", payload.get("l1_coeff", 3e-4))
        if not isinstance(l1_coeff, (int, float)):
            l1_coeff = 3e-4
        layer_index = hp.get("layer_index", 8)

        sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff)
        sae = _move_module_to_device(sae, self._device)
        sae.load_state_dict(state, strict=False)
        sae.eval()

        entry = {
            "sae": sae,
            "layer_index": layer_index,
            "d_model": d_model,
            "d_hidden": d_hidden,
        }
        self._sae_cache[checkpoint_path] = entry
        return entry

    def _get_collector(self, layer_index: int):
        """Lazily create (and cache) a GPT-2 activation collector.

        The ``data_collection`` module is imported here (not at module level)
        because it pulls in ``transformers`` + ``scipy`` etc., which takes
        20-40 s.  Deferring keeps server startup fast.
        """
        if layer_index not in self._collector_cache:
            from data_collection import GPT2ActivationCollector  # heavy import
            self._collector_cache[layer_index] = GPT2ActivationCollector(
                model_name="gpt2",
                layer_index=layer_index,
                device=self._device,
            )
        return self._collector_cache[layer_index]

    def _load_feature_labels(self, checkpoint_path: str) -> Dict[int, Dict[str, str]]:
        """Load feature label metadata from checkpoint + logs JSON files.

        Output shape:
            {feature_id: {"name": <label>, "description": <explanation-or-label>}}
        """
        if checkpoint_path in self._labels_cache:
            return self._labels_cache[checkpoint_path]

        labels: Dict[int, Dict[str, str]] = {}

        checkpoint_rel = Path(checkpoint_path).relative_to(self._checkpoints_dir)
        model_id = str(checkpoint_rel).replace("\\", "/")

        def _assign(feature_id: int, label: Any, explanation: Any = None) -> None:
            # Keep first assignment when duplicates exist across files.
            if feature_id in labels:
                return
            label_text = str(label).strip() if label is not None else ""
            if not label_text:
                return
            desc_text = str(explanation).strip() if explanation is not None else ""
            labels[feature_id] = {
                "name": label_text,
                "description": desc_text or label_text,
            }

        # 1) checkpoint-adjacent feature_labels.json
        labels_path = Path(checkpoint_path).parent / "feature_labels.json"
        if labels_path.exists():
            try:
                with open(labels_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, val in data.items():
                        if not str(key).isdigit():
                            continue
                        feature_id = int(key)
                        if isinstance(val, dict):
                            _assign(feature_id, val.get("label", f"Feature {feature_id}"), val.get("explanation"))
                        else:
                            _assign(feature_id, str(val))
            except Exception:
                pass

        # 2) model-specific runtime labels in logs/feature_labels_<model>.json
        model_labels_path = self._get_model_labels_path(model_id)
        if model_labels_path.exists():
            try:
                with open(model_labels_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key, val in data.items():
                        if not str(key).isdigit():
                            continue
                        feature_id = int(key)
                        if isinstance(val, dict):
                            _assign(feature_id, val.get("label", f"Feature {feature_id}"), val.get("explanation"))
                        elif isinstance(val, str):
                            _assign(feature_id, val)
            except Exception:
                pass

        # 3) legacy logs/feature_labels*.json files (fallback)
        logs_dir = _SAE_ROOT / "logs"
        if logs_dir.exists():
            for json_path in sorted(logs_dir.glob("*.json")):
                if json_path.name == model_labels_path.name:
                    continue
                try:
                    with open(json_path, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                if not isinstance(data, dict):
                    continue

                for key, val in data.items():
                    feature_id: Optional[int] = None
                    if str(key).isdigit():
                        feature_id = int(key)
                    elif isinstance(val, dict) and isinstance(val.get("feature_idx"), int):
                        feature_id = int(val["feature_idx"])

                    if feature_id is None:
                        continue

                    if isinstance(val, dict):
                        _assign(feature_id, val.get("label", f"Feature {feature_id}"), val.get("explanation"))
                    elif isinstance(val, str):
                        _assign(feature_id, val)

        self._labels_cache[checkpoint_path] = labels
        return labels

    def _get_model_labels_path(self, model_id: str) -> Path:
        """Return model-specific feature-label file path under logs/."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._")
        if not safe:
            safe = "unknown_model"
        return _SAE_ROOT / "logs" / f"feature_labels_{safe}.json"

    def _save_runtime_feature_label(
        self,
        model_id: str,
        checkpoint_path: str,
        feature_idx: int,
        label_result: Dict[str, Any],
    ) -> None:
        """Persist one feature label update and invalidate cache for live UI refresh."""
        path = self._get_model_labels_path(model_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = existing
            except Exception:
                payload = {}

        payload[str(feature_idx)] = {
            "feature_idx": feature_idx,
            "label": (label_result.get("label") or f"Feature {feature_idx}"),
            "explanation": (label_result.get("explanation") or label_result.get("label") or f"Feature {feature_idx}"),
            "confidence": label_result.get("confidence"),
            "request_id": label_result.get("request_id"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Drop cached labels so /analyze picks the latest labels immediately.
        self._labels_cache.pop(checkpoint_path, None)


# ---------------------------------------------------------------------------
# Auto-register when this module is imported
# ---------------------------------------------------------------------------
_instance = SAEAnalyzer(checkpoints_dir=CHECKPOINTS_DIR)
register(_instance)
