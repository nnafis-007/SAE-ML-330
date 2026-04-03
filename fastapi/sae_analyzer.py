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
        self._labels_cache: Dict[str, Dict[int, str]] = {}
        self._model_list_cache: Optional[List[Dict[str, Any]]] = None
        self._feature_activation_cache: Dict[str, Dict[str, Any]] = {}
        self._text_corpus_cache: Dict[str, List[str]] = {}
        self._dataset_cache_dir = _SAE_ROOT / ".cache" / "dataset_texts"
        self._dataset_cache_dir.mkdir(parents=True, exist_ok=True)

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
        checkpoint_path = str(self._checkpoints_dir / model_id)
        info = self._load_sae(checkpoint_path)
        sae: SparseAutoencoder = info["sae"]
        layer_index: int = info["layer_index"]

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

        # ---- GPT-2 hidden states ---------------------------------------------
        with torch.no_grad():
            outputs = collector.model(input_ids)
            # hidden_states: (embedding, layer0, layer1, …)
            hidden_states = outputs.hidden_states[layer_index + 1]
            activations = hidden_states[0]           # (seq_len, d_model)

        # ---- SAE encode ------------------------------------------------------
        with torch.no_grad():
            features = sae.encode(activations.to(self._device))  # (seq_len, d_hidden)

        # ---- pre-computed labels (if any) ------------------------------------
        labels = self._load_feature_labels(checkpoint_path)

        # ---- build per-token response ----------------------------------------
        tokens_data: List[Dict[str, Any]] = []
        for i, tid in enumerate(token_id_list):
            token_str = tokenizer.decode([tid])
            
            # Skip features for the first token (often an attention sink/BOS)
            if i == 0:
                tokens_data.append({
                    "text": token_str,
                    "features": [],
                    "info": "First token (attention sink) skipped"
                })
                continue
                
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
                        "description": labels.get(feat_id, f"Feature {feat_id}"),
                    })

            tokens_data.append({"text": token_str, "features": feat_list})

        return {
            "model": model_id,
            "layer_index": layer_index,
            "d_hidden": info["d_hidden"],
            "tokens": tokens_data,
        }

    # -- On-demand feature labeling (uses analyzers.llm_analysis.FeatureLabeler) --------

    def label_feature(
        self,
        model_id: str,
        feature_idx: int,
        corpus_texts: Optional[List[str]] = None,
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
            # Default to Hugging Face People's Speech so feature interpretation
            # uses a consistent dataset when UI doesn't provide sentence samples.
            dataset_name = "MLCommons/peoples_speech"
            dataset_config = "validation"
            split = "validation"
            sentence_budget = 200
            if labeling_config:
                sentence_budget = max(1, int(labeling_config.get("num_sentences", sentence_budget)))

            try:
                corpus_texts = self._load_dataset_texts(
                    dataset_name=dataset_name,
                    dataset_config=dataset_config,
                    split=split,
                    sentence_budget=sentence_budget,
                    text_field=None,
                    seed=0,
                )

                print(
                    "[SAEAnalyzer] label_feature corpus source | "
                    f"dataset={dataset_name}/{dataset_config}:{split} collected_sentences={len(corpus_texts)}"
                )
            except Exception as exc:
                print(f"[SAEAnalyzer] Failed to load People's Speech for label_feature ({exc}); using fallback corpus.")
                corpus_texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning enables computers to learn from data.",
                    "Neural networks have transformed natural language processing.",
                ]

        # Collect activations for this corpus
        activations = collector.collect_activations(
            texts=corpus_texts, batch_size=8, max_length=128,
        )

        from analyzers.llm_analysis import build_token_maps  # lazy import

        token_ids, doc_map, pos_map = build_token_maps(
            corpus_texts, tokenizer, max_length=128,
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

        return {
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

    def find_feature_activations(
        self,
        model_id: str,
        feature_id: int,
        dataset_name: str = "MLCommons/peoples_speech",
        dataset_config: str = "validation",
        split: str = "validation",
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

        # This endpoint currently uses raw GPT-2 hidden states (no mean/std normalization).
        normalization_mode = "none"

        print(
            "[SAEAnalyzer] Feature activation lookup started | "
            f"model={model_id} feature_id={feature_id} layer={layer_index} "
            f"dataset={dataset_name}/{dataset_config}:{split} max_sentences={max_sentences} "
            f"target_activating_examples={target_activating_examples} page={page} page_size={page_size} "
            f"min_activation={min_activation} "
            f"normalization={normalization_mode} filter='activation >= min_activation'"
        )

        page = max(1, int(page))
        page_size = max(1, int(page_size))
        if max_sentences is not None:
            max_sentences = max(1, int(max_sentences))
        if target_activating_examples is not None:
            target_activating_examples = max(1, int(target_activating_examples))

        cache_key = json.dumps(
            {
                "model_id": model_id,
                "feature_id": feature_id,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "split": split,
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
                "activation_filter": "activation >= min_activation",
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
            }

        if feature_id < 0 or feature_id >= d_hidden:
            raise ValueError(f"feature_id must be in [0, {d_hidden - 1}]")

        collector = self._get_collector(layer_index)
        tokenizer = collector.tokenizer
        labels = self._load_feature_labels(checkpoint_path)

        candidate_fields = [
            text_field,
            "text",
            "sentence",
            "transcript",
            "content",
            "article",
            "document",
        ]
        candidate_fields = [f for f in candidate_fields if f]

        sentence_budget = (
            max_sentences
            if max_sentences is not None
            else max(1000, int(target_activating_examples or 50) * 50)
        )

        corpus_texts = self._load_dataset_texts(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            sentence_budget=sentence_budget,
            text_field=text_field,
            seed=seed,
        )
        selected_text_fields = candidate_fields

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
                feature_mat = sae.encode(hidden_states.to(self._device)).cpu()

                feat_col = feature_mat[:, feature_id]
                active_positions = torch.nonzero(
                    feat_col >= float(min_activation), as_tuple=False
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
            "feature_description": labels.get(feature_id, f"Feature {feature_id}"),
            "text_fields_checked": selected_text_fields or candidate_fields,
            "scanned_sentences": texts_scanned,
            "raw_examples_scanned": examples_scanned,
            "scanned_tokens": total_tokens_scanned,
            "activating_sentences": activating_sentences,
        }

        return {
            "model": model_id,
            "feature_id": feature_id,
            "feature_description": labels.get(feature_id, f"Feature {feature_id}"),
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "text_fields_checked": selected_text_fields or candidate_fields,
            "scanned_sentences": texts_scanned,
            "raw_examples_scanned": examples_scanned,
            "scanned_tokens": total_tokens_scanned,
            "min_activation": min_activation,
            "activation_filter": "activation >= min_activation",
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
        }

    def _load_dataset_texts(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str,
        sentence_budget: int,
        text_field: Optional[str],
        seed: int,
    ) -> List[str]:
        """Load dataset texts once and reuse from memory/disk cache across requests.

        For UI responsiveness, return any cached texts immediately (even if fewer
        than the current sentence budget) instead of blocking on remote dataset
        resolution.
        """
        sentence_budget = max(1, int(sentence_budget))
        cache_payload = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "text_field": text_field,
            "seed": int(seed),
        }
        key_base = json.dumps(cache_payload, sort_keys=True)

        cached = self._text_corpus_cache.get(key_base)
        if cached:
            return cached[:sentence_budget]

        digest = hashlib.sha1(key_base.encode("utf-8")).hexdigest()[:16]
        disk_cache_path = self._dataset_cache_dir / f"{digest}.json"

        texts: List[str] = []
        if disk_cache_path.exists():
            try:
                payload = json.loads(disk_cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("texts"), list):
                    texts = [str(t) for t in payload["texts"] if isinstance(t, str) and t.strip()]
                if texts:
                    self._text_corpus_cache[key_base] = texts
                    returned = texts[:sentence_budget]
                    print(
                        "[SAEAnalyzer] Dataset text cache hit (disk) | "
                        f"dataset={dataset_name}/{dataset_config}:{split} texts={len(returned)}"
                    )
                    return returned
            except Exception:
                texts = []

        from datasets import Audio, load_dataset  # noqa: E402

        print(
            "[SAEAnalyzer] Preparing dataset text cache | "
            f"dataset={dataset_name}/{dataset_config}:{split} budget={sentence_budget}"
        )

        ds = None
        try:
            # Fast path: avoid hub/network metadata calls when local HF cache exists.
            ds = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=True,
                local_files_only=True,
            )
            print(
                "[SAEAnalyzer] Dataset load mode | "
                f"dataset={dataset_name}/{dataset_config}:{split} mode=local-cache"
            )
        except Exception:
            ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
            print(
                "[SAEAnalyzer] Dataset load mode | "
                f"dataset={dataset_name}/{dataset_config}:{split} mode=remote-resolve"
            )

        # Keep audio columns undecoded so text extraction does not require torchcodec.
        try:
            features = getattr(ds, "features", None) or {}
            for col, feat in features.items():
                if isinstance(feat, Audio):
                    ds = ds.cast_column(col, Audio(decode=False))
        except Exception:
            pass

        try:
            ds = ds.shuffle(seed=seed, buffer_size=max(1000, sentence_budget * 4))
        except Exception:
            pass

        candidate_fields = [
            text_field,
            "text",
            "sentence",
            "transcript",
            "content",
            "article",
            "document",
        ]
        candidate_fields = [f for f in candidate_fields if f]

        for example in ds:
            sentence = ""
            for field in candidate_fields:
                raw = example.get(field)
                if isinstance(raw, str) and raw.strip():
                    sentence = raw.strip()
                    break
            if sentence:
                texts.append(sentence)
            if len(texts) >= sentence_budget:
                break

        if texts:
            try:
                disk_cache_path.write_text(
                    json.dumps({"texts": texts}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

        self._text_corpus_cache[key_base] = texts
        print(
            "[SAEAnalyzer] Dataset text cache ready | "
            f"dataset={dataset_name}/{dataset_config}:{split} texts={len(texts[:sentence_budget])}"
        )
        return texts[:sentence_budget]

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
        sae.load_state_dict(state, strict=False)
        sae.to(self._device).eval()

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

    def _load_feature_labels(self, checkpoint_path: str) -> Dict[int, str]:
        """Load pre-computed feature labels from JSON if available."""
        if checkpoint_path in self._labels_cache:
            return self._labels_cache[checkpoint_path]

        labels: Dict[int, str] = {}
        # Look for feature_labels.json next to the checkpoint
        labels_path = Path(checkpoint_path).parent / "feature_labels.json"
        if labels_path.exists():
            with open(labels_path, encoding="utf-8") as f:
                data = json.load(f)
            for key, val in data.items():
                if isinstance(val, dict):
                    labels[int(key)] = val.get("label", f"Feature {key}")
                else:
                    labels[int(key)] = str(val)

        self._labels_cache[checkpoint_path] = labels
        return labels


# ---------------------------------------------------------------------------
# Auto-register when this module is imported
# ---------------------------------------------------------------------------
_instance = SAEAnalyzer(checkpoints_dir=CHECKPOINTS_DIR)
register(_instance)
