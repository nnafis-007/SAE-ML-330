"""
LLM-Based Feature Auto-Labeling for Sparse Autoencoders
========================================================

This module assigns human-readable semantic labels to SAE features by:

  1. Collecting the top-K token-level contexts that most strongly activate
     each feature (the "max-activating examples").
  2. Sending those snippets to an LLM together with a structured prompt.
  3. Parsing the LLM's response into a short label + confidence score.
  4. Persisting all labels to a JSON file for downstream use.

Supported LLM back-ends
-----------------------
* **OpenAI** (GPT-4o, GPT-4, GPT-3.5-turbo …) – requires ``openai`` package.
* **Groq** (Llama-3, Mixtral …) – fast, OpenAI-compatible, free tier available.
* **Ollama** (local models) – requires ``ollama`` package and local server.
"""

from __future__ import annotations

import json
import argparse
import logging
import os
import re
import sys
import time
import hashlib
import heapq
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sae_model import SparseAutoencoder

# Safe imports for helpers potentially moved or defined in data_collection
try:
    from data_collection import (
        GPT2ActivationCollector,
        _extract_text_from_example,
        _prune_dataset_to_text_columns,
        _resolve_hf_cache_dir
    )
except ImportError:
    # Fallbacks if data_collection is not available or structure differs
    GPT2ActivationCollector = None
    def _extract_text_from_example(example, text_field): return example.get(text_field or "text", "")
    def _prune_dataset_to_text_columns(ds, text_field): return ds
    def _resolve_hf_cache_dir(x): return x


LOGGER = logging.getLogger("llm_analysis")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabelingConfig:
    """
    All hyper-parameters governing the labeling process.
    """
    backend: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    top_k: int = 0  # 0 or negative => include all contexts above min_activation
    context_window: int = 10
    batch_size: int = 512
    max_features: Optional[int] = None
    request_delay: float = 0.5
    temperature: float = 0.2
    max_tokens: int = 120
    
    # API Keys
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    ollama_host: str = "http://localhost:11434"
    
    # Logging
    prompt_log_path: Optional[str] = "llm_prompts.log"
    request_id: Optional[str] = None
    
    # Normalization
    normalize_mode: str = "standardize"
    std_floor: float = 1e-3
    activation_mean: Optional[List[float]] = None
    activation_std: Optional[List[float]] = None
    
    # Filtering
    skip_first_token: bool = True
    global_top_features_k: int = 10
    min_activation: float = 0.0
    include_global_top_features: bool = False
    top_tokens_k: int = 10 


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TokenContext:
    token: str
    context: str
    activation_value: float

@dataclass
class LabelResult:
    feature_idx: int
    label: str
    explanation: str
    confidence: str
    top_tokens: List[str]
    top_contexts: List[TokenContext] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM back-end helpers
# ---------------------------------------------------------------------------

class _OpenAIBackend:
    def __init__(self, cfg: LabelingConfig):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install 'openai' package.") from exc

        api_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OpenAI API key missing.")
        self.client = openai.OpenAI(api_key=api_key)
        self.cfg = cfg

    def call(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return response.choices[0].message.content or ""


class _OllamaBackend:
    def __init__(self, cfg: LabelingConfig):
        try:
            import ollama
            self._ollama = ollama
        except ImportError as exc:
            raise ImportError("Install 'ollama' package.") from exc
        self.cfg = cfg

    def call(self, system_prompt: str, user_prompt: str) -> str:
        response = self._ollama.chat(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        )
        return response["message"]["content"]


class _GroqBackend:
    def __init__(self, cfg: LabelingConfig):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Install 'openai' package for Groq backend.") from exc

        api_key = cfg.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            apikey_path = PROJECT_ROOT / "apikey.txt"
            if apikey_path.exists():
                api_key = apikey_path.read_text().strip().replace("+", "")

        if not api_key:
            raise EnvironmentError("Groq API key missing. Set GROQ_API_KEY or add to apikey.txt.")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.cfg = cfg

    def call(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return response.choices[0].message.content or ""


def _build_backend(cfg: LabelingConfig):
    if cfg.backend == "openai":
        return _OpenAIBackend(cfg)
    elif cfg.backend == "groq":
        return _GroqBackend(cfg)
    elif cfg.backend == "ollama":
        return _OllamaBackend(cfg)
    else:
        raise ValueError(f"Unknown backend '{cfg.backend}'.")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in neural network interpretability, specifically in analysing \
Sparse Autoencoders (SAEs) trained on language model activations.

Your task: given a list of text snippets in which a particular SAE feature \
strongly activates, identify the **semantic concept** the feature is detecting.

Rules:
- Produce a short label (at most 8 words) that captures the concept.
- Provide a one-sentence explanation.
- Rate your confidence as "high", "medium", or "low".
- Reply ONLY in the following JSON format and nothing else:

{
  "label": "<short concept label>",
  "explanation": "<one sentence>",
  "confidence": "high|medium|low"
}
"""

_USER_PROMPT_TEMPLATE = """\
Feature index: {feature_idx}

The feature most strongly activates on the following tokens and their surrounding context.
The activating token is marked with >>> <<< to help you focus on it.

Top {n_examples} activating examples (ordered strongest first):
{examples_block}

{top_tokens_block}

Based on these examples, what concept does this SAE feature detect?
"""


# ---------------------------------------------------------------------------
# Core: FeatureLabeler
# ---------------------------------------------------------------------------

class FeatureLabeler:
    """
    Labels SAE features with semantic concepts using an LLM.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        tokenizer: GPT2Tokenizer,
        cfg: LabelingConfig = LabelingConfig(),
        device: Optional[str] = None,
    ):
        self.sae = sae
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae.to(self.device).eval()
        self._backend = _build_backend(cfg)
        LOGGER.info(
            "FeatureLabeler initialized | backend=%s model=%s device=%s top_k=%s min_activation=%s normalize_mode=%s",
            cfg.backend,
            cfg.model,
            self.device,
            cfg.top_k,
            cfg.min_activation,
            cfg.normalize_mode,
        )

    def _get_normalization_stats(self, activations: torch.Tensor) -> Dict[str, Any]:
        cfg = self.cfg
        if cfg.activation_mean is not None and cfg.activation_std is not None:
            LOGGER.info("Using provided normalization stats from config (activation_mean/std).")
            return {
                "mean": torch.tensor(cfg.activation_mean).to(activations.dtype),
                "std": torch.tensor(cfg.activation_std).to(activations.dtype),
                "normalize_mode": cfg.normalize_mode,
                "std_floor": cfg.std_floor,
            }
        
        warnings.warn(
            "Computing normalization stats from the provided activation batch. "
            "For best results, provide training set statistics in LabelingConfig.",
            UserWarning
        )
        LOGGER.info("Computed normalization stats from current activation batch.")
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        return {
            "mean": mean,
            "std": std,
            "normalize_mode": cfg.normalize_mode,
            "std_floor": cfg.std_floor,
        }

    def _apply_activation_normalization(
        self,
        x: torch.Tensor,
        norm_stats: Dict[str, Any],
    ) -> torch.Tensor:
        mode = str(norm_stats.get("normalize_mode", "standardize")).lower().strip()
        mean = norm_stats.get("mean")
        std = norm_stats.get("std")
        std_floor = float(norm_stats.get("std_floor", self.cfg.std_floor))

        if mean is None:
            return x

        mean = mean.to(device=x.device, dtype=x.dtype)

        if mode in {"none", "off", "no"}:
            return x
        if mode in {"center", "center_only", "mean"}:
            return x - mean
        if mode in {"standardize", "zscore", "z-score"}:
            if std is None:
                return x - mean
            std = std.to(device=x.device, dtype=x.dtype)
            if std_floor > 0:
                std = std.clamp_min(std_floor)
            return (x - mean) / (std + 1e-8)

        raise ValueError(f"Unknown normalize_mode='{mode}'.")

    # ------------------------------------------------------------------
    # Optimized Core Logic (Streaming / Memory Efficient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _stream_latents_and_accumulate(
        self,
        feature_indices: List[int],
        activations: torch.Tensor,
        token_ids: List[List[int]],
        token_doc_map: List[int],
        token_pos_map: List[int],
        norm_stats: Dict[str, Any],
    ) -> Tuple[Dict[int, List[TokenContext]], Optional[torch.Tensor]]:
        """
        Streams through activations, encodes to latents, and accumulates 
        top-K contexts for each feature. 
        
        Returns:
            Tuple of:
            - Dict mapping feature_idx to list of TokenContext.
            - Mean activations tensor (for global top features, if configured).
        """
        cfg = self.cfg
        n_tokens = int(activations.shape[0])
        
        # Accumulators
        # Use a heap to keep track of top K efficiently.
        # Heap stores tuples: (activation_value, unique_id, TokenContext)
        # unique_id breaks ties and ensures stable sorting.
        feature_heaps: Dict[int, List[Tuple[float, int, TokenContext]]] = {
            idx: [] for idx in feature_indices
        }
        global_activation_sums = torch.zeros(self.sae.d_hidden) if cfg.include_global_top_features else None
        
        # Valid row logic
        # We compute valid rows inside the loop to avoid allocating a huge tensor
        # if skip_first_token is True.
        
        batch_size = cfg.batch_size
        uid_counter = 0
        
        LOGGER.info(
            "Streaming SAE encoding | tokens=%d batch_size=%d features=%d",
            n_tokens, batch_size, len(feature_indices)
        )

        for start in tqdm(range(0, n_tokens, batch_size), desc="Encoding & Collecting"):
            end = min(start + batch_size, n_tokens)
            batch_acts = activations[start:end].to(self.device)
            
            # Normalize
            batch_normed = self._apply_activation_normalization(batch_acts, norm_stats)
            
            # Encode
            latents = self.sae.encode(batch_normed) # (Batch, d_hidden)
            
            # Global stats
            if global_activation_sums is not None:
                global_activation_sums += latents.sum(dim=0).cpu()

            # Process each feature
            # Note: If feature_indices is very large, iterating python-side is slow.
            # But typically we label few features at a time.
            for feat_idx in feature_indices:
                col = latents[:, feat_idx].cpu()
                
                # Filter by min_activation
                if cfg.min_activation > 0:
                    mask = col >= cfg.min_activation
                    valid_local_indices = mask.nonzero(as_tuple=True)[0]
                    vals = col[valid_local_indices]
                else:
                    valid_local_indices = torch.arange(col.shape[0])
                    vals = col

                # Skip first token logic
                # We check global positions: start + local_idx
                # Optimized: vectorize check if needed, but loop is fine for small batches
                
                for local_idx, val in zip(valid_local_indices.tolist(), vals.tolist()):
                    global_idx = start + local_idx
                    
                    if cfg.skip_first_token:
                        if token_pos_map[global_idx] == 0:
                            continue

                    # Top-K Heap Management
                    heap = feature_heaps[feat_idx]
                    k = cfg.top_k if cfg.top_k > 0 else 10000 # Arbitrary large number if unlimited
                    
                    # Optimization: only push if value is high enough
                    # Heaps in Python are min-heaps. We store (-val) to simulate max-heap
                    # or just store val and pop smallest.
                    
                    # We want Top K highest values.
                    # Heap stores (activation, uid, context). Smallest activation at root.
                    
                    # Simple optimization: if we have K items and this is smaller than min, skip.
                    # But we need context info first.
                    
                    # Construct context
                    doc_id = token_doc_map[global_idx]
                    pos = token_pos_map[global_idx]
                    toks = token_ids[doc_id]
                    
                    cw = cfg.context_window
                    lo = max(0, pos - cw)
                    hi = min(len(toks), pos + cw + 1)

                    prefix_text = self.tokenizer.decode(toks[lo:pos], skip_special_tokens=True)
                    target_text = self.tokenizer.decode(toks[pos:pos + 1], skip_special_tokens=True)
                    suffix_text = self.tokenizer.decode(toks[pos + 1:hi], skip_special_tokens=True)

                    context_str = f"{prefix_text}>>>{target_text}<<<{suffix_text}"
                    token_s = target_text.strip()
                    
                    ctx_obj = TokenContext(token=token_s, context=context_str, activation_value=val)
                    
                    if len(heap) < k:
                        heapq.heappush(heap, (val, uid_counter, ctx_obj))
                        uid_counter += 1
                    else:
                        # Check if current value is larger than the smallest in heap
                        if val > heap[0][0]:
                            heapq.heapreplace(heap, (val, uid_counter, ctx_obj))
                            uid_counter += 1

        # Finalize: Sort heaps by activation desc
        final_contexts: Dict[int, List[TokenContext]] = {}
        for feat_idx, heap in feature_heaps.items():
            # Sort descending by value
            sorted_heap = sorted(heap, key=lambda x: x[0], reverse=True)
            final_contexts[feat_idx] = [item[2] for item in sorted_heap]
            
        return final_contexts, global_activation_sums

    # ------------------------------------------------------------------
    # LLM Interaction
    # ------------------------------------------------------------------

    def _build_examples_block(self, contexts: List[TokenContext]) -> str:
        lines = []
        for i, ctx in enumerate(contexts, 1):
            lines.append(f"  {i:>2}. [act={ctx.activation_value:.3f}]  {ctx.context}")
        return "\n".join(lines)

    def _top_tokens(self, contexts: List[TokenContext]) -> List[str]:
        from collections import Counter
        counts = Counter(c.token for c in contexts if c.token)
        k = int(self.cfg.top_tokens_k)
        if k == 0: return []
        if k < 0: return [tok for tok, _ in counts.most_common()]
        return [tok for tok, _ in counts.most_common(k)]

    def _call_llm(
        self,
        feature_idx: int,
        contexts: List[TokenContext],
        global_top_features: str = "(not computed)",
    ) -> LabelResult:
        LOGGER.info(
            "Calling LLM for feature %d | contexts=%d prompt_logging=%s",
            feature_idx,
            len(contexts),
            bool(self.cfg.prompt_log_path),
        )
        examples_block = self._build_examples_block(contexts)
        top_tokens = self._top_tokens(contexts)
        
        if int(self.cfg.top_tokens_k) == 0:
            top_tokens_block = ""
        else:
            label = "all" if int(self.cfg.top_tokens_k) < 0 else f"top {len(top_tokens)}"
            top_tokens_str = ", ".join(f'"{t}"' for t in top_tokens) if top_tokens else "(none)"
            top_tokens_block = f"Most frequent activating tokens ({label}): {top_tokens_str}\n"

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            feature_idx=feature_idx,
            n_examples=len(contexts),
            examples_block=examples_block,
            top_tokens_block=top_tokens_block,
        )

        if self.cfg.include_global_top_features:
            user_prompt += f"\n\nTop activated features corpus-wide:\n{global_top_features}\n"

        raw = ""
        try:
            if self.cfg.prompt_log_path:
                with open(self.cfg.prompt_log_path, "a", encoding="utf-8") as f:
                    req = self.cfg.request_id or "n/a"
                    f.write(f"\n{'='*80}\nREQUEST_ID {req}\nFEATURE {feature_idx}\nUSER PROMPT:\n{user_prompt}\n")

            raw = self._backend.call(_SYSTEM_PROMPT, user_prompt)
            
            if self.cfg.prompt_log_path:
                with open(self.cfg.prompt_log_path, "a", encoding="utf-8") as f:
                    f.write(f"RESPONSE:\n{raw}\n{'='*80}\n")

            parsed = self._parse_response(raw)
            LOGGER.info("LLM labeling succeeded for feature %d.", feature_idx)
            return LabelResult(
                feature_idx=feature_idx,
                label=parsed.get("label", "unlabeled"),
                explanation=parsed.get("explanation", ""),
                confidence=parsed.get("confidence", "low"),
                top_tokens=top_tokens,
                top_contexts=contexts,
                raw_response=raw,
            )
        except Exception as exc:
            LOGGER.error("LLM labeling failed for feature %d: %s", feature_idx, exc)
            return LabelResult(
                feature_idx=feature_idx,
                label="error",
                explanation="",
                confidence="low",
                top_tokens=[],
                top_contexts=contexts,
                raw_response=raw,
                error=str(exc),
            )

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, str]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        label = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
        explanation = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw)
        confidence = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw)
        return {
            "label": label.group(1) if label else "unknown",
            "explanation": explanation.group(1) if explanation else raw[:200],
            "confidence": confidence.group(1) if confidence else "low",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_feature_from_activations(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        token_ids: List[List[int]],
        token_doc_map: List[int],
        token_pos_map: List[int],
    ) -> LabelResult:
        LOGGER.info("Single-feature labeling requested | feature=%d", feature_idx)
        results = self.label_features_from_activations(
            feature_indices=[feature_idx],
            activations=activations,
            token_ids=token_ids,
            token_doc_map=token_doc_map,
            token_pos_map=token_pos_map,
            save_path=None,
            resume=False,
        )
        return results[feature_idx]

    def label_features_from_activations(
        self,
        feature_indices: List[int],
        activations: torch.Tensor,
        token_ids: List[List[int]],
        token_doc_map: List[int],
        token_pos_map: List[int],
        save_path: Optional[str] = "feature_labels.json",
        resume: bool = True,
    ) -> Dict[int, LabelResult]:
        """
        Label multiple features using pre-collected GPT-2 activations.
        
        OPTIMIZED: Streams activations to avoid OOM. Accumulates top-K contexts per feature.
        """
        existing: Dict[int, LabelResult] = {}
        if resume and save_path and Path(save_path).exists():
            existing = self._load_labels(save_path)
            already = {k for k, v in existing.items() if v.error is None}
            feature_indices = [i for i in feature_indices if i not in already]
            LOGGER.info("Resuming from saved labels | completed=%d remaining=%d", len(already), len(feature_indices))

        if self.cfg.max_features is not None:
            feature_indices = feature_indices[:self.cfg.max_features]

        if not feature_indices:
            LOGGER.info("No features left to label after resume/max_features filtering.")
            return existing

        run_start = time.time()
        LOGGER.info(
            "Starting labeling run | features=%d tokens=%d docs=%d save_path=%s",
            len(feature_indices),
            int(activations.shape[0]),
            len(token_ids),
            save_path or "(disabled)",
        )

        results: Dict[int, LabelResult] = dict(existing)
        
        # 1. Compute Stats
        norm_stats = self._get_normalization_stats(activations)
        
        # 2. Stream encoding and collection
        feature_contexts, global_acts = self._stream_latents_and_accumulate(
            feature_indices=feature_indices,
            activations=activations,
            token_ids=token_ids,
            token_doc_map=token_doc_map,
            token_pos_map=token_pos_map,
            norm_stats=norm_stats
        )
        
        # Global top features string
        global_top_str = "(not computed)"
        if global_acts is not None:
            mean_acts = global_acts / activations.shape[0]
            vals, idxs = torch.topk(mean_acts, self.cfg.global_top_features_k)
            parts = [f"{int(i)}:{float(v):.3f}" for v, i in zip(vals.tolist(), idxs.tolist())]
            global_top_str = ", ".join(parts)

        # 3. LLM Calls
        total_features = len(feature_indices)
        for feat_pos, feat_idx in enumerate(tqdm(feature_indices, desc="Labeling features"), start=1):
            if feat_pos == 1 or feat_pos % 5 == 0 or feat_pos == total_features:
                LOGGER.info(
                    "Feature labeling progress | feature=%d position=%d/%d",
                    feat_idx,
                    feat_pos,
                    total_features,
                )
            
            contexts = feature_contexts[feat_idx]

            if not contexts:
                LOGGER.info("Feature %d produced no contexts above threshold (dead feature on this corpus).", feat_idx)
                result = LabelResult(
                    feature_idx=feat_idx,
                    label="dead feature",
                    explanation="Feature never activates on this corpus.",
                    confidence="high",
                    top_tokens=[],
                    error="no activating examples found",
                )
            else:
                result = self._call_llm(feat_idx, contexts, global_top_str)
                time.sleep(self.cfg.request_delay)

            results[feat_idx] = result

            if save_path:
                self._save_labels(results, save_path)

        if save_path:
            LOGGER.info("Labels saved to %s", save_path)

        total_done = len(feature_indices)
        total_success = sum(1 for idx in feature_indices if results[idx].error is None)
        total_errors = total_done - total_success
        LOGGER.info(
            "Labeling run completed | completed=%d success=%d errors=%d elapsed_sec=%.2f",
            total_done,
            total_success,
            total_errors,
            time.time() - run_start,
        )

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_labels(results: Dict[int, LabelResult], path: str) -> None:
        serialisable = {}
        for feat_idx, r in results.items():
            d = asdict(r)
            serialisable[str(feat_idx)] = d
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_labels(path: str) -> Dict[int, LabelResult]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        results = {}
        for key, val in raw.items():
            feat_idx = int(key)
            val["top_contexts"] = [TokenContext(**ctx) for ctx in val.get("top_contexts", [])]
            results[feat_idx] = LabelResult(**val)
        return results

    @staticmethod
    def print_label(result: LabelResult) -> None:
        status = "✓" if result.error is None else "✗"
        print(f"\n{status} Feature {result.feature_idx}")
        print(f"   Label      : {result.label}")
        print(f"   Confidence : {result.confidence}")
        print(f"   Explanation: {result.explanation}")
        if result.top_tokens:
            print(f"   Top tokens : {', '.join(result.top_tokens[:10])}")
        if result.error:
            print(f"   Error      : {result.error}")


# ---------------------------------------------------------------------------
# Utility: build token maps
# ---------------------------------------------------------------------------

def build_token_maps(
    texts: List[str],
    tokenizer: GPT2Tokenizer,
    max_length: int = 128,
) -> tuple:
    token_ids: List[List[int]] = []
    token_doc_map: List[int] = []
    token_pos_map: List[int] = []

    for doc_idx, text in enumerate(tqdm(texts, desc="Tokenising corpus", leave=False)):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = enc["input_ids"][0].tolist()
        token_ids.append(ids)
        for pos in range(len(ids)):
            token_doc_map.append(doc_idx)
            token_pos_map.append(pos)

    return token_ids, token_doc_map, token_pos_map


def _dataset_text_cache_key(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_field: Optional[str],
    shuffle_buffer_size: int,
    seed: int,
) -> str:
    payload = {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "text_field": text_field,
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "seed": int(seed),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_cached_texts(cache_path: Path) -> List[str]:
    if not cache_path.exists():
        return []

    texts: List[str] = []
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "") if isinstance(obj, dict) else ""
            except json.JSONDecodeError:
                text = line
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def _append_cached_texts(cache_path: Path, texts: List[str]) -> None:
    if not texts:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        for text in texts:
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _split_into_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [c.strip() for c in chunks if c and c.strip()]


def _append_cached_sentences(cache_path: Path, texts: List[str]) -> int:
    if not texts:
        return 0

    sentence_count = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        for text in texts:
            for sentence in _split_into_sentences(text):
                fh.write(json.dumps({"sentence": sentence}, ensure_ascii=False) + "\n")
                sentence_count += 1

    return sentence_count


def _fetch_dataset_texts_segment(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_field: Optional[str],
    shuffle_buffer_size: int,
    seed: int,
    start_at: int,
    limit: int,
    hf_cache_dir: Optional[str],
) -> List[str]:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    dataset = _prune_dataset_to_text_columns(dataset, text_field)
    if shuffle_buffer_size and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=int(shuffle_buffer_size))

    wanted_until = start_at + limit
    valid_seen = 0
    collected: List[str] = []

    for example in tqdm(dataset, desc="Extending HF text cache", total=wanted_until):
        text = _extract_text_from_example(example, text_field)
        if not text.strip():
            continue
        if valid_seen >= start_at:
            collected.append(text)
            if len(collected) >= limit:
                break
        valid_seen += 1

    return collected


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if GPT2ActivationCollector is None:
        print("Error: GPT2ActivationCollector could not be imported. Please ensure data_collection.py is available.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Label SAE features with an LLM using GPT-2 activations."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MLCommons/peoples_speech",
        help="Hugging Face dataset name. Use empty string to use built-in sample_texts.",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=4000,
        help="Number of texts to load from dataset when --dataset is set.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split name.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config name (subset).",
    )
    parser.add_argument(
        "--dataset-text-field",
        type=str,
        default=None,
        help="Optional explicit text field in the dataset records.",
    )
    parser.add_argument(
        "--dataset-shuffle-buffer-size",
        type=int,
        default=0,
        help="Streaming shuffle buffer size for Hugging Face datasets. Set 0 to disable shuffle.",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=0,
        help="Random seed for streaming dataset shuffle.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPT-2 activation collection.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length per text for activation collection and token maps.",
    )
    parser.add_argument(
        "--feature-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[0, 10],
        help="Feature index range to interpret [START:END]. Example: --feature-range 0 20",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="feature_labels.json",
        help="Output JSON file for labels.",
    )
    parser.add_argument(
        "--prompt-log-path",
        type=str,
        default="llm_prompts.log",
        help="Log file path for all LLM prompts and responses.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="groq",
        choices=["openai", "groq", "ollama"],
        help="LLM backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Model name for selected backend.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Delay between LLM requests.",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="standardize",
        choices=["standardize", "center", "none"],
        help="Normalization mode before SAE encoding. Default: standardize",
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=1e-3,
        help="Std clamp floor for standardization. Default: 1e-3",
    )
    parser.add_argument(
        "--include-first-token",
        action="store_true",
        help="Include first-token activations (disabled by default due to GPT-2 attention sink).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume labeling from --save-path if it exists.",
    )
    parser.add_argument(
        "--activation-cache-path",
        type=str,
        default=".cache/llm_analysis_activation_cache.pt",
        help="Path to activation/token-map cache (.pt).",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=str,
        default=".cache",
        help="Directory for persistent Hugging Face text caches.",
    )
    parser.add_argument(
        "--no-cache-load",
        action="store_true",
        help="Do not load from --activation-cache-path even if it exists.",
    )
    parser.add_argument(
        "--no-cache-save",
        action="store_true",
        help="Do not save activation/token-map cache after collection.",
    )
    parser.add_argument(
        "--min-activation",
        type=float,
        default=0.0,
        help="Minimum feature activation to include in LLM examples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Maximum number of activating contexts to send to the LLM. Use 0 or negative to send all.",
    )
    parser.add_argument(
        "--include-global-top-features",
        action="store_true",
        help="Include corpus-global top activated feature IDs in the LLM prompt.",
    )
    parser.add_argument(
        "--top-tokens-k",
        type=int,
        default=10,
        help="How many most frequent activating tokens to include. 0=none, negative=all, positive=top-k.",
    )
    args = parser.parse_args()
    
    feature_start, feature_end = args.feature_range
    if feature_start < 0 or feature_end <= feature_start:
        parser.error("--feature-range must be two integers with 0 <= START < END")

    LOGGER.info(
        "llm_analysis started | dataset=%s split=%s config=%s num_texts=%d batch_size=%d max_length=%d feature_range=[%d:%d] backend=%s model=%s",
        args.dataset or "(built-in)",
        args.dataset_split,
        args.dataset_config,
        args.num_texts,
        args.batch_size,
        args.max_length,
        feature_start,
        feature_end,
        args.backend,
        args.model,
    )

    # 1. Load SAE
    LOGGER.info("Loading SAE checkpoint...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "FC_SAE_Best.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    
    if not ckpt_path.exists():
        LOGGER.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    payload = torch.load(str(ckpt_path), map_location="cpu")
    hp = payload.get("hyperparameters", {})
    state = payload["model_state_dict"]
    d_model = hp.get("d_model", state["W_enc"].shape[1])
    d_hidden = hp.get("d_hidden", state["W_enc"].shape[0])
    l1_coeff = hp.get("l1_coeff", payload.get("l1_coeff", 3e-4))
    layer_index = hp.get("layer_index", 8)

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff)
    sae.load_state_dict(state)
    sae.eval()

    # 2. Data Collection
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    use_dataset = bool(args.dataset and args.dataset.strip())
    cache_path = Path(args.activation_cache_path)
    cache_loaded = False

    cache_meta = {
        "dataset": (args.dataset or "").strip(),
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_text_field": args.dataset_text_field,
        "dataset_shuffle_buffer_size": int(args.dataset_shuffle_buffer_size),
        "dataset_seed": int(args.dataset_seed),
        "num_texts": int(args.num_texts),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "layer_index": int(layer_index),
        "checkpoint": str(ckpt_path),
        "use_dataset": bool(use_dataset),
    }
    cache_fingerprint = hashlib.sha1(
        json.dumps(cache_meta, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if args.no_cache_load:
        LOGGER.info("Skipping activation cache load due to --no-cache-load")
    elif cache_path.exists():
        LOGGER.info("Loading activation cache from %s", cache_path)
        try:
            cache = torch.load(str(cache_path))
            saved_meta = cache.get("meta", {}) if isinstance(cache, dict) else {}
            saved_fp = saved_meta.get("fingerprint") if isinstance(saved_meta, dict) else None

            if not saved_fp:
                raise ValueError("cache metadata missing")
            if saved_fp != cache_fingerprint:
                raise ValueError("cache metadata mismatch")

            activations = cache["activations"].float()
            sample_texts = cache["texts"]
            token_ids = cache["token_ids"]
            doc_map = cache["token_doc_map"]
            pos_map = cache["token_pos_map"]

            assert len(doc_map) == activations.shape[0], "Cache size mismatch (doc_map)"
            assert len(pos_map) == activations.shape[0], "Cache size mismatch (pos_map)"
            assert len(token_ids) == len(sample_texts), "Cache size mismatch (token_ids/texts)"

            LOGGER.info(
                "Activation cache loaded | tokens=%d texts=%d",
                int(activations.shape[0]),
                len(sample_texts),
            )
            cache_loaded = True
        except Exception as e:
            LOGGER.warning("Cache load failed or stale (%s). Recollecting activations.", e)

    if not cache_loaded:
        if use_dataset:
             collector = GPT2ActivationCollector(layer_index=layer_index)
             text_cache_key = _dataset_text_cache_key(
                 dataset_name=args.dataset,
                 dataset_config=args.dataset_config,
                 split=args.dataset_split,
                 text_field=args.dataset_text_field,
                 shuffle_buffer_size=args.dataset_shuffle_buffer_size,
                 seed=args.dataset_seed,
             )
             text_cache_path = PROJECT_ROOT / args.text_cache_dir / f"hf_texts_{text_cache_key}.jsonl"
             sentence_cache_path = PROJECT_ROOT / args.text_cache_dir / f"hf_sentences_{text_cache_key}.jsonl"
             cached_texts = _load_cached_texts(text_cache_path)
             
             if cached_texts and not sentence_cache_path.exists():
                 _append_cached_sentences(sentence_cache_path, cached_texts)

             if len(cached_texts) < args.num_texts:
                 need = args.num_texts - len(cached_texts)
                 resolved_hf_cache_dir = _resolve_hf_cache_dir(None)
                 try:
                     new_texts = _fetch_dataset_texts_segment(
                         dataset_name=args.dataset,
                         dataset_config=args.dataset_config,
                         split=args.dataset_split,
                         text_field=args.dataset_text_field,
                         shuffle_buffer_size=args.dataset_shuffle_buffer_size,
                         seed=args.dataset_seed,
                         start_at=len(cached_texts),
                         limit=need,
                         hf_cache_dir=resolved_hf_cache_dir,
                     )
                 except Exception as e:
                     LOGGER.error("Failed to extend HF text cache: %s", e)
                     raise

                 if not new_texts:
                     LOGGER.error("Could not fetch additional texts.")
                     sys.exit(1)
                 _append_cached_texts(text_cache_path, new_texts)
                 _append_cached_sentences(sentence_cache_path, new_texts)
                 cached_texts.extend(new_texts)

             sample_texts = cached_texts[:args.num_texts]
             activations = collector.collect_activations(
                 sample_texts,
                 batch_size=args.batch_size,
                 max_length=args.max_length,
             )
        else:
             LOGGER.info("Using built-in sample texts for activation collection.")
             sample_texts = [
                "Massive gargantuan behemoths roam desolate barren wastelands.",
                "Fast swift rapid movements characterize agile nimble creatures.",
                "Happy joyful cheerful emotions brighten gloomy somber days.",
                "Intelligent smart clever scholars study complex intricate subjects.",
                "Powerful strong mighty warriors defend ancient historic cities.",
             ]
             collector = GPT2ActivationCollector(layer_index=layer_index)
             activations = collector.collect_activations(sample_texts, batch_size=args.batch_size, max_length=args.max_length)

        token_ids, doc_map, pos_map = build_token_maps(sample_texts, tokenizer, args.max_length)
        
        if args.no_cache_save:
            LOGGER.info("Skipping activation cache save due to --no-cache-save")
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "activations": activations.cpu(),
                "texts": sample_texts,
                "token_ids": token_ids,
                "token_doc_map": doc_map,
                "token_pos_map": pos_map,
                "meta": {
                    **cache_meta,
                    "fingerprint": cache_fingerprint,
                    "saved_at": time.time(),
                },
            }, str(cache_path))
            LOGGER.info(
                "Saved activation cache | path=%s tokens=%d texts=%d",
                cache_path,
                int(activations.shape[0]),
                len(sample_texts),
            )

    # 3. Config Setup
    cfg = LabelingConfig(
        backend=args.backend,
        model=args.model,
        request_delay=args.request_delay,
        prompt_log_path=args.prompt_log_path,
        normalize_mode=args.normalize_mode,
        std_floor=args.std_floor,
        skip_first_token=not args.include_first_token,
        min_activation=args.min_activation,
        top_k=args.top_k,
        include_global_top_features=args.include_global_top_features,
        top_tokens_k=args.top_tokens_k,
    )
    
    labeler = FeatureLabeler(sae, tokenizer, cfg)

    # 4. Feature Selection
    max_feature_id = int(sae.d_hidden)
    start_idx = min(feature_start, max_feature_id)
    end_idx = min(feature_end, max_feature_id)
    if start_idx >= end_idx:
        LOGGER.error(
            "Requested feature range [%d:%d] is empty for model feature size %d.",
            feature_start,
            feature_end,
            max_feature_id,
        )
        sys.exit(1)

    target_features = list(range(start_idx, end_idx))
    LOGGER.info(
        "Selected direct feature index range | d_hidden=%d requested=[%d:%d] effective=[%d:%d] selected=%d",
        max_feature_id,
        feature_start,
        feature_end,
        start_idx,
        end_idx,
        len(target_features),
    )
    print(f"\nModel feature count (d_hidden): {max_feature_id}")
    print(
        f"Selected feature index range [{feature_start}:{feature_end}] "
        f"(effective [{start_idx}:{end_idx}]): {target_features}"
    )

    # 5. Run Labeling
    LOGGER.info("Starting LLM labeling for %d features...", len(target_features))
    print("\nLabeling features...")
    results = labeler.label_features_from_activations(
        feature_indices=target_features,
        activations=activations,
        token_ids=token_ids,
        token_doc_map=doc_map,
        token_pos_map=pos_map,
        save_path=args.save_path,
        resume=args.resume,
    )

    for r in results.values():
        FeatureLabeler.print_label(r)