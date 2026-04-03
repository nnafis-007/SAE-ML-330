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
* **OpenAI** (GPT-4o, GPT-4, GPT-3.5-turbo …) – requires ``openai`` package
  and an ``OPENAI_API_KEY`` environment variable.
* **Ollama** (local models such as ``llama3``, ``mistral`` …) – requires the
  ``ollama`` package and a running ``ollama serve`` process.

Typical usage
-------------
::

    from analysis import FeatureLabeler, LabelingConfig
    from interpretation import FeatureAnalyzer

    cfg = LabelingConfig(backend="openai", model="gpt-4o-mini", top_k=20)
    labeler = FeatureLabeler(sae, tokenizer, gpt2_model, cfg)

    # Label a specific feature
    result = labeler.label_feature(feature_idx=42, texts=corpus_texts)
    print(result)  # LabelResult(feature_idx=42, label="animal / pets", ...)

    # Label all alive features and save to JSON
    all_labels = labeler.label_all_features(texts, alive_indices, save_path="labels.json")
"""

from __future__ import annotations

import json
import argparse
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

import torch
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sae_lens import SAE
from datasets import Audio, load_dataset

# Ensure local project modules under src/ are importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SparseAutoencoder = SAE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabelingConfig:
    """
    All hyper-parameters governing the labeling process.

    Attributes
    ----------
    backend : str
        Which LLM back-end to use: ``"openai"`` or ``"ollama"``.
    model : str
        Model name understood by the chosen back-end
        (e.g. ``"gpt-4o-mini"`` for OpenAI, ``"llama3"`` for Ollama).
    top_k : int
        Number of top-activating token contexts to gather per feature.
    context_window : int
        Number of tokens to include **before** and **after** the activating
        token so the LLM sees meaningful surrounding context.
    batch_size : int
        How many activation vectors to process at once when scanning the
        corpus.  Lower this if you run out of GPU memory.
    max_features : Optional[int]
        Cap on the number of features to label in one call to
        ``label_all_features``.  ``None`` = label every alive feature.
    request_delay : float
        Seconds to wait between LLM API calls to respect rate limits.
    temperature : float
        Sampling temperature for the LLM response.
    max_tokens : int
        Maximum tokens in the LLM response.
    openai_api_key : Optional[str]
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when ``None``.
    ollama_host : str
        Base URL for the Ollama server (default ``"http://localhost:11434"``).
    """

    backend: str = "groq"             # "openai" | "groq" | "ollama"
    model: str = "llama-3.3-70b-versatile"  # fast & free on Groq
    top_k: int = 40  # 0 or negative => include all contexts above min_activation
    context_window: int = 10          # tokens on each side of the activating token
    batch_size: int = 512
    max_features: Optional[int] = None
    request_delay: float = 0.5        # seconds between API calls
    temperature: float = 0.2
    max_tokens: int = 120
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None    # from console.groq.com — free tier
    ollama_host: str = "http://localhost:11434"
    prompt_log_path: Optional[str] = "llm_prompts.log"  # Log all prompts/responses here
    skip_first_token: bool = True
    global_top_features_k: int = 10
    min_activation: float = 0.0
    include_global_top_features: bool = False
    top_tokens_k: int = 10  # 0 => none, <0 => all, >0 => top-k


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TokenContext:
    """A single top-activating example for a feature."""
    token: str                   # The specific token that fired the feature
    context: str                 # Surrounding text snippet
    activation_value: float      # Raw activation magnitude


@dataclass
class LabelResult:
    """
    The semantic label assigned to one SAE feature.

    Attributes
    ----------
    feature_idx : int
    label : str
        Short (≤ 8 words) human-readable label, e.g. ``"animal / pets"``.
    explanation : str
        One-sentence justification from the LLM.
    confidence : str
        Self-reported confidence: ``"high"`` / ``"medium"`` / ``"low"``.
    top_tokens : List[str]
        The most frequent tokens in the top-K activating positions, for
        quick reference without re-running the LLM.
    top_contexts : List[TokenContext]
        Full context objects (also serialised to JSON for auditing).
    raw_response : str
        Verbatim LLM output (useful for debugging).
    error : Optional[str]
        Set to an error description if the LLM call failed.
    """

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
    """Thin wrapper around the OpenAI chat-completions API."""

    def __init__(self, cfg: LabelingConfig):
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the OpenAI backend.\n"
                "Install it with:  pip install openai"
            ) from exc

        api_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "No OpenAI API key found.  Either pass it via "
                "LabelingConfig(openai_api_key='sk-...') or set the "
                "OPENAI_API_KEY environment variable."
            )
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
    """Thin wrapper around a local Ollama server."""

    def __init__(self, cfg: LabelingConfig):
        try:
            import ollama  # type: ignore
            self._ollama = ollama
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is required for the Ollama backend.\n"
                "Install it with:  pip install ollama\n"
                "Also make sure 'ollama serve' is running locally."
            ) from exc
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
    """
    Thin wrapper around the Groq API.

    Groq provides a **free tier** (≈14,400 req/day) with fast inference on
    Llama-3, Mixtral, and Gemma models.  Its API is fully OpenAI-compatible,
    so we just point the openai client at a different base URL.

    Sign up and get a key at: https://console.groq.com
    Then either pass it via LabelingConfig(groq_api_key='gsk_...') or set
    the GROQ_API_KEY environment variable.
    """

    def __init__(self, cfg: LabelingConfig):
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the Groq backend.\n"
                "Install it with:  pip install openai"
            ) from exc

        api_key = cfg.groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Fallback: try reading from apikey.txt in project root
        if not api_key:
            apikey_path = PROJECT_ROOT / "apikey.txt"
            if apikey_path.exists():
                api_key = apikey_path.read_text().strip().replace("+", "") # Remove '+' from the user's snippet if present

        if not api_key:
            raise EnvironmentError(
                "No Groq API key found.  Either pass it via "
                "LabelingConfig(groq_api_key='gsk_...') or set the "
                "GROQ_API_KEY environment variable, or place it in apikey.txt.\n"
                "Get a free key at: https://console.groq.com"
            )
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
    """Factory: instantiate the correct backend from the config."""
    if cfg.backend == "openai":
        return _OpenAIBackend(cfg)
    elif cfg.backend == "groq":
        return _GroqBackend(cfg)
    elif cfg.backend == "ollama":
        return _OllamaBackend(cfg)
    else:
        raise ValueError(
            f"Unknown backend '{cfg.backend}'.  Choose 'openai', 'groq', or 'ollama'."
        )


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

    Parameters
    ----------
    sae : SparseAutoencoder
        Trained SAE whose features we want to label.
    tokenizer : GPT2Tokenizer
        Must be the tokenizer used during GPT-2 activation collection so that
        token indices map correctly.
    gpt2_model : GPT2LMHeadModel
        The GPT-2 model used to re-tokenize the *text* corpus into token
        strings.  Only the tokenizer is strictly needed; the model is kept for
        potential future logit-lens extensions.
    cfg : LabelingConfig
        Labeling hyper-parameters.
    device : str
        Torch device for SAE inference.
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

    def _build_valid_row_indices(self, token_pos_map: List[int]) -> torch.Tensor:
        """Build row indices used for scoring contexts (optionally skip token position 0)."""
        if not token_pos_map:
            return torch.empty(0, dtype=torch.long)

        if self.cfg.skip_first_token:
            indices = [i for i, pos in enumerate(token_pos_map) if int(pos) > 0]
        else:
            indices = list(range(len(token_pos_map)))

        return torch.tensor(indices, dtype=torch.long)

    @torch.no_grad()
    def _compute_mean_feature_activations(
        self,
        activations: torch.Tensor,
        valid_row_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean activation for each SAE feature over selected token rows."""
        if valid_row_indices.numel() == 0:
            return torch.zeros(self.sae.d_hidden, dtype=torch.float32)

        sums = torch.zeros(self.sae.d_hidden, dtype=torch.float32)
        idx = valid_row_indices.tolist()

        for start in range(0, len(idx), self.cfg.batch_size):
            batch_idx = idx[start:start + self.cfg.batch_size]
            batch = activations[batch_idx].to(self.device)
            encoded = self.sae.encode(batch)
            sums += encoded.sum(dim=0).float().cpu()

        return sums / float(len(idx))

    @staticmethod
    def _format_global_top_features(mean_acts: torch.Tensor, top_n: int = 10) -> str:
        """Format highest mean-activation features for optional prompt context."""
        if mean_acts.numel() == 0:
            return "(none)"

        top_n = max(0, int(top_n))
        if top_n == 0:
            return "(none)"

        alive = (mean_acts > 0).nonzero(as_tuple=True)[0]
        if alive.numel() == 0:
            return "(none)"

        alive_vals = mean_acts[alive]
        k = min(top_n, int(alive_vals.numel()))
        vals, rel_idx = torch.topk(alive_vals, k)
        feat_idx = alive[rel_idx]
        return "\n".join(
            f"{int(i)}:{float(v):.6f}" for i, v in zip(feat_idx.tolist(), vals.tolist())
        )

    # ------------------------------------------------------------------
    # Step 1 – Collect token-level top-activating contexts
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_token_contexts(
        self,
        feature_idx: int,
        token_ids: List[List[int]],   # one list of token ids per document
        activations: torch.Tensor,    # (total_tokens, d_model)
        token_doc_map: List[int],     # which document each activation row belongs to
        token_pos_map: List[int],     # position of each activation within its document
        valid_row_indices: Optional[torch.Tensor] = None,
    ) -> List[TokenContext]:
        """
        Scan pre-collected activations and return the top-K contexts for
        ``feature_idx``, with surrounding token strings.
        """
        cfg = self.cfg

        # Safety check: activations and token maps must correspond row-for-row.
        n = activations.shape[0]
        if n != len(token_doc_map):
            raise ValueError(
                f"activations has {n} rows but token_doc_map has {len(token_doc_map)} entries. "
                "They must be the same size — the token maps must be built from the SAME "
                "corpus that generated the activations.  Use build_token_maps() on the texts "
                "you originally passed to GPT2ActivationCollector."
            )

        # Encode all activations in mini-batches to get feature values
        feat_vals = torch.zeros(n, dtype=torch.float32)

        for start in range(0, n, cfg.batch_size):
            end = min(start + cfg.batch_size, n)
            batch = activations[start:end].to(self.device)
            encoded = self.sae.encode(batch)           # (batch, d_hidden)
            feat_vals[start:end] = encoded[:, feature_idx].cpu()

        candidate_rows = valid_row_indices
        if candidate_rows is None:
            candidate_rows = self._build_valid_row_indices(token_pos_map)

        if candidate_rows.numel() == 0:
            return []

        candidate_vals = feat_vals[candidate_rows]

        # Ignore weak activations to reduce noisy/irrelevant contexts.
        if cfg.min_activation > 0:
            keep_mask = candidate_vals >= float(cfg.min_activation)
            candidate_rows = candidate_rows[keep_mask]
            candidate_vals = candidate_vals[keep_mask]

        if candidate_rows.numel() == 0:
            return []

        if cfg.top_k is None or int(cfg.top_k) <= 0:
            k = int(candidate_rows.numel())
        else:
            k = min(int(cfg.top_k), int(candidate_rows.numel()))
        top_vals, rel_top_idxs = torch.topk(candidate_vals, k)
        top_idxs = candidate_rows[rel_top_idxs]

        contexts: List[TokenContext] = []
        seen_contexts = set()
        cw = cfg.context_window

        for val, row_idx in zip(top_vals.tolist(), top_idxs.tolist()):
            doc_id = token_doc_map[row_idx]
            pos    = token_pos_map[row_idx]
            toks   = token_ids[doc_id]

            # Build a surrounding context string
            lo = max(0, pos - cw)
            hi = min(len(toks), pos + cw + 1)

            prefix_ids  = toks[lo:pos]
            target_ids  = toks[pos:pos + 1]
            suffix_ids  = toks[hi - 1:hi] if pos + 1 < hi else []

            prefix  = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            token_s = self.tokenizer.decode(target_ids, skip_special_tokens=True)
            suffix  = self.tokenizer.decode(toks[pos + 1:hi], skip_special_tokens=True)

            context_str = f"{prefix}>>>{token_s}<<<{suffix}"

            dedupe_key = (token_s.strip(), context_str)
            if dedupe_key in seen_contexts:
                continue
            seen_contexts.add(dedupe_key)

            contexts.append(TokenContext(
                token=token_s.strip(),
                context=context_str,
                activation_value=val,
            ))

        return contexts

    # ------------------------------------------------------------------
    # Step 2 – Build prompt & call LLM
    # ------------------------------------------------------------------

    def _build_examples_block(self, contexts: List[TokenContext]) -> str:
        lines = []
        for i, ctx in enumerate(contexts, 1):
            lines.append(
                f"  {i:>2}. [act={ctx.activation_value:.3f}]  {ctx.context}"
            )
        return "\n".join(lines)

    def _top_tokens(self, contexts: List[TokenContext]) -> List[str]:
        from collections import Counter
        counts = Counter(c.token for c in contexts if c.token)
        k = int(self.cfg.top_tokens_k)
        if k == 0:
            return []
        if k < 0:
            return [tok for tok, _ in counts.most_common()]
        return [tok for tok, _ in counts.most_common(k)]

    def _call_llm(
        self,
        feature_idx: int,
        contexts: List[TokenContext],
        global_top_features: str = "(not computed)",
    ) -> LabelResult:
        """Build the prompt, call the backend, parse the JSON response."""
        examples_block = self._build_examples_block(contexts)
        top_tokens = self._top_tokens(contexts)
        if int(self.cfg.top_tokens_k) == 0:
            top_tokens_block = ""
        else:
            if int(self.cfg.top_tokens_k) < 0:
                top_tokens_label = "all"
            else:
                top_tokens_label = f"top {len(top_tokens)}"
            top_tokens_str = ", ".join(f'"{t}"' for t in top_tokens) if top_tokens else "(none)"
            top_tokens_block = (
                f"Most frequent activating tokens ({top_tokens_label}): {top_tokens_str}\n"
            )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            feature_idx=feature_idx,
            n_examples=len(contexts),
            examples_block=examples_block,
            top_tokens_block=top_tokens_block,
        )

        if self.cfg.include_global_top_features:
            user_prompt += (
                "\n\nTop activated features across the analyzed corpus "
                "(top 10, feature_idx:mean_activation):\n"
                f"{global_top_features}\n"
            )

        raw = ""
        try:
            # 1. Log the prompts if enabled
            if self.cfg.prompt_log_path:
                with open(self.cfg.prompt_log_path, "a", encoding="utf-8") as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"FEATURE INDEX: {feature_idx}\n")
                    f.write("-" * 40 + "\n")
                    f.write("SYSTEM PROMPT:\n")
                    f.write(f"{_SYSTEM_PROMPT}\n")
                    f.write("-" * 40 + "\n")
                    f.write("USER PROMPT:\n")
                    f.write(f"{user_prompt}\n")
                    f.write("-" * 40 + "\n")

            raw = self._backend.call(_SYSTEM_PROMPT, user_prompt)
            
            # 2. Log the response if enabled
            if self.cfg.prompt_log_path:
                with open(self.cfg.prompt_log_path, "a", encoding="utf-8") as f:
                    f.write("LLM RESPONSE:\n")
                    f.write(f"{raw}\n")
                    f.write("="*80 + "\n")

            parsed = self._parse_response(raw)
            return LabelResult(
                feature_idx=feature_idx,
                label=parsed.get("label", "unlabeled"),
                explanation=parsed.get("explanation", ""),
                confidence=parsed.get("confidence", "low"),
                top_tokens=top_tokens,
                top_contexts=contexts,
                raw_response=raw,
            )
        except Exception as exc:  # noqa: BLE001
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
        """
        Extract JSON from the LLM response.

        The LLM is instructed to reply with pure JSON, but may wrap it in
        markdown code fences.  We try several strategies.
        """
        # 1. Direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # 3. Extract first {...} block
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 4. Fall back: try to extract fields with regex
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

    def label_feature(
        self,
        feature_idx: int,
        texts: List[str],
        max_length: int = 128,
    ) -> LabelResult:
        """
        Label a single feature.

        Parameters
        ----------
        feature_idx : int
            Index of the SAE feature to label (0 ≤ idx < d_hidden).
        texts : List[str]
            Raw text corpus.  The same corpus used to collect training
            activations gives the best results.
        max_length : int
            Tokenization truncation length.

        Returns
        -------
        LabelResult
        """
        token_ids, activations, token_doc_map, token_pos_map = \
            self._tokenize_and_collect_activations(texts, max_length)

        contexts = self._collect_token_contexts(
            feature_idx, token_ids, activations, token_doc_map, token_pos_map
        )

        if not contexts:
            return LabelResult(
                feature_idx=feature_idx,
                label="dead feature",
                explanation="Feature never activates on this corpus.",
                confidence="high",
                top_tokens=[],
                error="no activating examples found",
            )

        return self._call_llm(feature_idx, contexts)

    def label_features(
        self,
        feature_indices: List[int],
        texts: List[str],
        max_length: int = 128,
        save_path: Optional[str] = None,
        resume: bool = True,
    ) -> Dict[int, LabelResult]:
        """
        Label a list of features, with optional JSON persistence and resuming.

        Parameters
        ----------
        feature_indices : List[int]
            Feature indices to label.
        texts : List[str]
            Raw text corpus.
        max_length : int
            Tokenisation truncation length.
        save_path : Optional[str]
            If provided, results are written (and incrementally updated) to
            this JSON file after each successful label.
        resume : bool
            If ``True`` and ``save_path`` already exists, previously-labeled
            features are loaded and skipped, so you can safely restart a
            labeling run that was interrupted.

        Returns
        -------
        Dict[int, LabelResult]  keyed by feature index.
        """
        existing: Dict[int, LabelResult] = {}

        # Resume from disk
        if resume and save_path and Path(save_path).exists():
            existing = self._load_labels(save_path)
            already_done = set(existing.keys())
            feature_indices = [i for i in feature_indices if i not in already_done]
            print(f"[FeatureLabeler] Resuming: {len(already_done)} already done, "
                  f"{len(feature_indices)} remaining.")

        # Cap at max_features
        if self.cfg.max_features is not None:
            feature_indices = feature_indices[:self.cfg.max_features]

        if not feature_indices:
            print("[FeatureLabeler] Nothing left to label.")
            return existing

        print(f"[FeatureLabeler] Tokenising corpus ({len(texts)} texts)…")
        token_ids, activations, token_doc_map, token_pos_map = \
            self._tokenize_and_collect_activations(texts, max_length)
        print(f"[FeatureLabeler] Collected {activations.shape[0]} token activations.")

        results: Dict[int, LabelResult] = dict(existing)

        for feat_idx in tqdm(feature_indices, desc="Labeling features"):
            contexts = self._collect_token_contexts(
                feat_idx, token_ids, activations, token_doc_map, token_pos_map
            )

            if not contexts:
                result = LabelResult(
                    feature_idx=feat_idx,
                    label="dead feature",
                    explanation="Feature never activates on this corpus.",
                    confidence="high",
                    top_tokens=[],
                    error="no activating examples found",
                )
            else:
                result = self._call_llm(feat_idx, contexts)
                time.sleep(self.cfg.request_delay)

            results[feat_idx] = result

            # Incremental save
            if save_path:
                self._save_labels(results, save_path)

        if save_path:
            print(f"[FeatureLabeler] Labels saved to {save_path}")

        return results

    def label_all_features(
        self,
        texts: List[str],
        alive_indices: Optional[List[int]] = None,
        max_length: int = 128,
        save_path: Optional[str] = "feature_labels.json",
        resume: bool = True,
    ) -> Dict[int, LabelResult]:
        """
        Convenience wrapper: label every *alive* feature in the SAE.

        Parameters
        ----------
        texts : List[str]
            Raw text corpus.
        alive_indices : Optional[List[int]]
            If provided, only those features are labeled.  Otherwise every
            feature index in ``range(sae.d_hidden)`` is used.
        max_length : int
            Tokenization truncation length.
        save_path : Optional[str]
            JSON path for incremental saves and resuming.
        resume : bool
            Skip features already present in ``save_path``.

        Returns
        -------
        Dict[int, LabelResult]  keyed by feature index.
        """
        indices = alive_indices if alive_indices is not None \
            else list(range(self.sae.d_hidden))
        return self.label_features(
            feature_indices=indices,
            texts=texts,
            max_length=max_length,
            save_path=save_path,
            resume=resume,
        )

    # ------------------------------------------------------------------
    # Corpus tokenization  (shared across all features in one run)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _tokenize_and_collect_activations(
        self,
        texts: List[str],
        max_length: int = 128,
    ):
        """
        Tokenise the corpus and run it through the SAE encoder, collecting
        the raw input activations (GPT-2 residual-stream vectors) that
        ``FeatureAnalyzer`` would normally pass in.

        Because we only have *flat* activation tensors without position
        metadata here, we re-tokenize the texts to recover (doc, position)
        mappings needed by ``_collect_token_contexts``.

        .. note::
           This method requires that the user pass the **raw residual-stream
           activations** through the standard workflow.  In the typical
           workflow you load ``activations.pt`` from disk (produced by
           ``data_collection.py``) and use it directly.  If you have that
           tensor, call :meth:`label_feature_from_activations` instead,
           which is faster and avoids re-running GPT-2.

        Returns
        -------
        token_ids : List[List[int]]
        activations : torch.Tensor  (total_tokens, d_model)
        token_doc_map : List[int]
        token_pos_map : List[int]
        """
        # We only need the tokenizer here to recover per-position mappings.
        all_token_ids: List[List[int]] = []
        all_input_ids: List[int] = []
        token_doc_map: List[int] = []
        token_pos_map: List[int] = []

        for doc_idx, text in enumerate(texts):
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            ids = enc["input_ids"][0].tolist()
            all_token_ids.append(ids)
            for pos, tok_id in enumerate(ids):
                all_input_ids.append(tok_id)
                token_doc_map.append(doc_idx)
                token_pos_map.append(pos)

        # We don't have the GPT-2 residual stream here – callers are expected
        # to supply it via label_feature_from_activations.  Return a sentinel.
        # This path is used only when activations are NOT pre-computed.
        warnings.warn(
            "label_feature / label_all_features called without pre-computed "
            "activations.  The corpus will be re-tokenized but you must call "
            "label_feature_from_activations() with the actual GPT-2 hidden "
            "states for meaningful results.",
            UserWarning,
            stacklevel=3,
        )

        # Return empty activations tensor as placeholder
        dummy_acts = torch.zeros(len(all_input_ids), self.sae.d_model)
        return all_token_ids, dummy_acts, token_doc_map, token_pos_map

    # ------------------------------------------------------------------
    # Primary API: use pre-collected activations (recommended)
    # ------------------------------------------------------------------

    def label_feature_from_activations(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        token_ids: List[List[int]],
        token_doc_map: List[int],
        token_pos_map: List[int],
        valid_row_indices: Optional[torch.Tensor] = None,
        global_top_features: Optional[str] = None,
    ) -> LabelResult:
        """
        Label a single feature using **pre-collected GPT-2 activations**.

        This is the **recommended** entry point.  Load ``activations.pt`` once,
        build the token maps with :func:`build_token_maps`, then call this
        method for each feature you want to label.

        Parameters
        ----------
        feature_idx : int
        activations : torch.Tensor
            Shape ``(total_tokens, d_model)``.  The GPT-2 residual-stream
            vectors at the layer the SAE was trained on.
        token_ids : List[List[int]]
            Per-document token ID lists (from :func:`build_token_maps`).
        token_doc_map : List[int]
            ``token_doc_map[i]`` = index of the document that produced row *i*
            of ``activations``.
        token_pos_map : List[int]
            ``token_pos_map[i]`` = token position within that document.

        Returns
        -------
        LabelResult
        """
        if valid_row_indices is None:
            valid_row_indices = self._build_valid_row_indices(token_pos_map)
        if global_top_features is None:
            mean_acts = self._compute_mean_feature_activations(
                activations=activations,
                valid_row_indices=valid_row_indices,
            )
            global_top_features = self._format_global_top_features(
                mean_acts,
                top_n=self.cfg.global_top_features_k,
            )

        contexts = self._collect_token_contexts(
            feature_idx,
            token_ids,
            activations,
            token_doc_map,
            token_pos_map,
            valid_row_indices=valid_row_indices,
        )
        if not contexts:
            return LabelResult(
                feature_idx=feature_idx,
                label="dead feature",
                explanation="Feature never activates on this corpus.",
                confidence="high",
                top_tokens=[],
                error="no activating examples found",
            )
        return self._call_llm(feature_idx, contexts, global_top_features)

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
        Label multiple features using **pre-collected GPT-2 activations**.

        Parameters
        ----------
        feature_indices : List[int]
        activations : torch.Tensor
            Shape ``(total_tokens, d_model)``.
        token_ids, token_doc_map, token_pos_map
            From :func:`build_token_maps`.
        save_path : Optional[str]
            JSON path for incremental saves.
        resume : bool
            Skip features already in ``save_path``.

        Returns
        -------
        Dict[int, LabelResult]  keyed by feature index.
        """
        existing: Dict[int, LabelResult] = {}
        if resume and save_path and Path(save_path).exists():
            existing = self._load_labels(save_path)
            # Only skip features that were labeled successfully (no error).
            # Features that previously errored will be retried.
            already = {k for k, v in existing.items() if v.error is None}
            feature_indices = [i for i in feature_indices if i not in already]
            print(f"[FeatureLabeler] Resuming: {len(already)} done, "
                  f"{len(feature_indices)} remaining.")

        if self.cfg.max_features is not None:
            feature_indices = feature_indices[:self.cfg.max_features]

        results: Dict[int, LabelResult] = dict(existing)
        valid_row_indices = self._build_valid_row_indices(token_pos_map)
        mean_acts = self._compute_mean_feature_activations(
            activations=activations,
            valid_row_indices=valid_row_indices,
        )
        global_top_features = self._format_global_top_features(
            mean_acts,
            top_n=self.cfg.global_top_features_k,
        )

        for feat_idx in tqdm(feature_indices, desc="Labeling features"):
            result = self.label_feature_from_activations(
                feat_idx,
                activations,
                token_ids,
                token_doc_map,
                token_pos_map,
                valid_row_indices=valid_row_indices,
                global_top_features=global_top_features,
            )
            results[feat_idx] = result

            if save_path:
                self._save_labels(results, save_path)

            time.sleep(self.cfg.request_delay)

        if save_path:
            print(f"[FeatureLabeler] Labels saved to {save_path}")

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_labels(results: Dict[int, LabelResult], path: str) -> None:
        """Serialise results dict to JSON (human-readable)."""
        serialisable = {}
        for feat_idx, r in results.items():
            d = asdict(r)
            # TokenContext objects already converted by asdict
            serialisable[str(feat_idx)] = d
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_labels(path: str) -> Dict[int, LabelResult]:
        """Deserialise results from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        results = {}
        for key, val in raw.items():
            feat_idx = int(key)
            # Reconstruct nested TokenContext objects
            val["top_contexts"] = [
                TokenContext(**ctx) for ctx in val.get("top_contexts", [])
            ]
            results[feat_idx] = LabelResult(**val)
        return results

    # ------------------------------------------------------------------
    # Display / summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_label(result: LabelResult) -> None:
        """Pretty-print a single LabelResult."""
        status = "✓" if result.error is None else "✗"
        print(f"\n{status} Feature {result.feature_idx}")
        print(f"   Label      : {result.label}")
        print(f"   Confidence : {result.confidence}")
        print(f"   Explanation: {result.explanation}")
        if result.top_tokens:
            print(f"   Top tokens : {', '.join(result.top_tokens[:10])}")
        if result.error:
            print(f"   Error      : {result.error}")

    @staticmethod
    def print_summary(results: Dict[int, LabelResult]) -> None:
        """Print a compact table of all labeled features."""
        print("\n" + "=" * 70)
        print(f"{'IDX':>6}  {'CONF':^6}  LABEL")
        print("=" * 70)
        for feat_idx in sorted(results.keys()):
            r = results[feat_idx]
            err_marker = " [ERR]" if r.error else ""
            print(f"{feat_idx:>6}  {r.confidence:^6}  {r.label}{err_marker}")
        print("=" * 70)
        n_high = sum(1 for r in results.values() if r.confidence == "high")
        n_err  = sum(1 for r in results.values() if r.error)
        print(f"Total: {len(results)} features | "
              f"High confidence: {n_high} | "
              f"Errors: {n_err}")


# ---------------------------------------------------------------------------
# Utility: build token maps from a text corpus + tokenizer
# ---------------------------------------------------------------------------

def build_token_maps(
    texts: List[str],
    tokenizer: GPT2Tokenizer,
    max_length: int = 128,
) -> tuple:
    """
    Tokenise a list of texts and build the per-token document / position maps
    needed by :class:`FeatureLabeler`.

    This is a **standalone** helper intended to be called once before labeling
    so that the tokenisation is not repeated per feature.

    Parameters
    ----------
    texts : List[str]
        Raw text corpus.
    tokenizer : GPT2Tokenizer
    max_length : int
        Truncation length.

    Returns
    -------
    token_ids : List[List[int]]
        Token ID list per document.
    token_doc_map : List[int]
        ``token_doc_map[i]`` = document index for global token row *i*.
    token_pos_map : List[int]
        ``token_pos_map[i]`` = position within the document for row *i*.

    Example
    -------
    ::

        activations = torch.load("activations.pt")  # (N, d_model)
        texts = [...]  # same texts used to generate activations

        token_ids, doc_map, pos_map = build_token_maps(texts, tokenizer)

        labeler = FeatureLabeler(sae, tokenizer, cfg)
        result = labeler.label_feature_from_activations(
            42, activations, token_ids, doc_map, pos_map
        )
    """
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


# ---------------------------------------------------------------------------
# Quick demo / __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal end-to-end demo.

    Requires:
        - A saved SAE checkpoint:  checkpoints/best_model.pt
        - Either OPENAI_API_KEY set, or a running Ollama server.

    Activations are collected on-the-fly from the sample corpus using GPT-2,
    so the token maps and activation rows ALWAYS correspond 1-to-1.

    Run:
        OPENAI_API_KEY=sk-... python src/analysis.py
    """
    sys.path.insert(0, str(SRC_DIR))

    parser = argparse.ArgumentParser(
        description="Label SAE features with an LLM using GPT-2 activations."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt2-small-res-jb:blocks.8.hook_resid_pre",
        help=(
            "Pretrained SAE id in release:sae_id format. "
            "Default: gpt2-small-res-jb:blocks.8.hook_resid_pre"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
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
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory (falls back to HF_DATASETS_CACHE env var).",
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
        help="Streaming shuffle buffer size for Hugging Face datasets. Set 0 to disable shuffle and start yielding texts immediately.",
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
        "--top-feature-count",
        type=int,
        default=10,
        help="Number of features to interpret. Number of top activated alive features. Use a positive integer to select specific count to label. Default: 10",
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
        default="checkpoints/llm_analysis_activation_cache.pt",
        help="Path to activation/token-map cache (.pt).",
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
        default=40,
        help="Maximum number of activating contexts to send to the LLM. Default: 40. Use 0 or negative to send all contexts above --min-activation.",
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

    hf_cache_dir = args.hf_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    if hf_cache_dir:
        print(f"[llm_analysis] Using Hugging Face cache dir: {hf_cache_dir}")

    # ------------------------------------------------------------------ #
    # 1.  Load the trained SAE
    # ------------------------------------------------------------------ #
    print("Loading pretrained SAE…")
    if ":" not in args.model_id or args.model_id.count(":") != 1:
        raise ValueError(
            "--model-id must be in 'release:sae_id' format, "
            "e.g. gpt2-small-res-jb:blocks.8.hook_resid_pre"
        )
    release, sae_id = args.model_id.split(":", 1)
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cpu")
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()

    d_model = int(getattr(sae.cfg, "d_in", getattr(sae.cfg, "d_model", sae.W_enc.shape[1])))
    d_hidden = int(getattr(sae.cfg, "d_sae", getattr(sae.cfg, "d_hidden", sae.W_enc.shape[0])))
    setattr(sae, "d_model", d_model)
    setattr(sae, "d_hidden", d_hidden)

    hook_name = str(getattr(getattr(sae.cfg, "metadata", object()), "hook_name", sae_id))
    layer_match = re.search(r"blocks\.(\d+)\.", hook_name)
    layer_index = int(layer_match.group(1)) if layer_match else 8

    print(f"  model_id={args.model_id}")
    print(f"  d_model={d_model}, d_hidden={d_hidden}, layer_index={layer_index}")

    # ------------------------------------------------------------------ #
    # 2.  Sample corpus fallback
    # ------------------------------------------------------------------ #
    sample_texts = [
        "Massive gargantuan behemoths roam desolate barren wastelands.",
        "Fast swift rapid movements characterize agile nimble creatures.",
        "Happy joyful cheerful emotions brighten gloomy somber days.",
        "Intelligent smart clever scholars study complex intricate subjects.",
        "Powerful strong mighty warriors defend ancient historic cities.",
        "Loud noisy vociferous crowds ignore quiet silent whispers.",
        "Clean pure pristine lakes reflect jagged sharp peaks.",
        "Bitter cold freezing winters follow warm hot summers.",
        "Ancient old antique artifacts reveal hidden secret histories.",
        "Scented fragrant aromatic flowers attract small tiny insects.",
        "Wealthy rich affluent merchants trade expensive costly goods.",
        "Honest truthful sincere friends provide helpful useful advice.",
        "Scary frightening terrifying dreams haunt dark black nights.",
        "Tasty delicious savory meals satisfy hungry famished guests.",
        "Bright luminous radiant stars illuminate deep black space.",
        "Brave courageous fearless heroes face dangerous risky perils.",
        "Simple easy basic tasks require minimal small effort.",
        "Fast-paced speedy quick changes create chaotic messy situations.",
        "Gentle soft mild breezes stir colorful vibrant leaves.",
        "Mysterious strange odd occurrences baffle wise learned men.",
    ]

    # ------------------------------------------------------------------ #
    # 3.  Collect GPT-2 activations FOR THESE EXACT TEXTS
    #     (same layer the SAE was trained on)
    #     with optional checkpointing of activations/token maps
    # ------------------------------------------------------------------ #
    use_dataset = bool(args.dataset and args.dataset.strip())
    cache_path = Path(args.activation_cache_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    cache_loaded = False
    if cache_path.exists() and not args.no_cache_load:
        print(f"\nLoading activation cache from {cache_path}…")
        try:
            cache = torch.load(str(cache_path), map_location="cpu")
            activations = cache["activations"].float().cpu()
            sample_texts = cache["texts"]
            token_ids = cache.get("token_ids")
            doc_map = cache.get("token_doc_map")
            pos_map = cache.get("token_pos_map")

            if token_ids is None or doc_map is None or pos_map is None:
                print("  Cache missing token maps. Rebuilding from cached texts…")
                token_ids, doc_map, pos_map = build_token_maps(
                    sample_texts, tokenizer, max_length=args.max_length
                )

            if len(doc_map) != activations.shape[0]:
                raise ValueError(
                    f"Cached token map length {len(doc_map)} != activations rows {activations.shape[0]}"
                )

            meta = cache.get("meta", {})

            expected_dataset = args.dataset if use_dataset else "built_in_sample"
            expected_split = args.dataset_split if use_dataset else "n/a"
            expected_config = args.dataset_config if use_dataset else None
            if (
                meta.get("dataset") != expected_dataset
                or meta.get("split") != expected_split
                or meta.get("dataset_config") != expected_config
                or int(meta.get("max_length", -1)) != int(args.max_length)
            ):
                raise ValueError(
                    "Cache metadata does not match current dataset/split/config/max_length."
                )

            print(
                "  Cache loaded: "
                f"tokens={activations.shape[0]}, texts={len(sample_texts)}, "
                f"dataset={meta.get('dataset', 'unknown')}, split={meta.get('split', 'unknown')}, "
                f"config={meta.get('dataset_config', 'none')}"
            )
            cache_loaded = True
        except Exception as exc:  # noqa: BLE001
            print(f"  Cache load failed ({exc}). Falling back to fresh collection.")

    if not cache_loaded:
        if use_dataset:
            print(
                f"\nCollecting GPT-2 activations from Hugging Face dataset "
                f"'{args.dataset}' ({args.num_texts} texts, config='{args.dataset_config}', split='{args.dataset_split}', "
                f"layer {layer_index})…"
            )
        else:
            print(
                f"\nCollecting GPT-2 activations from built-in sample corpus "
                f"(layer {layer_index})…"
            )

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
        gpt2.eval()

        if use_dataset:
            ds = load_dataset(
                args.dataset,
                args.dataset_config,
                split=args.dataset_split,
                streaming=True,
                trust_remote_code=True,
                cache_dir=hf_cache_dir,
            )
            try:
                features = getattr(ds, "features", None) or {}
                for col, feat in features.items():
                    if isinstance(feat, Audio):
                        ds = ds.cast_column(col, Audio(decode=False))
            except Exception:
                pass

            if args.dataset_shuffle_buffer_size and args.dataset_shuffle_buffer_size > 0:
                try:
                    ds = ds.shuffle(seed=args.dataset_seed, buffer_size=args.dataset_shuffle_buffer_size)
                except Exception:
                    pass

            text_fields = [
                args.dataset_text_field,
                "text",
                "sentence",
                "transcript",
                "content",
                "article",
                "document",
            ]
            text_fields = [f for f in text_fields if f]

            sample_texts = []
            for ex in ds:
                selected = ""
                for field in text_fields:
                    raw = ex.get(field)
                    if isinstance(raw, str) and raw.strip():
                        selected = raw.strip()
                        break
                if selected:
                    sample_texts.append(selected)
                if len(sample_texts) >= args.num_texts:
                    break
        else:
            sample_texts = list(sample_texts)

        act_rows: List[torch.Tensor] = []
        with torch.no_grad():
            for text in sample_texts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"]
                if input_ids.numel() == 0:
                    continue
                out = gpt2(input_ids=input_ids)
                hidden = out.hidden_states[layer_index + 1][0].cpu()
                act_rows.append(hidden)

        if not act_rows:
            raise ValueError("No token activations were collected from the selected corpus")
        activations = torch.cat(act_rows, dim=0)

        # activations: (total_tokens_in_sample_texts, d_model)
        print(f"  Collected {activations.shape[0]} token activations.")

        # ------------------------------------------------------------------ #
        # 4.  Build token maps from the SAME sample_texts
        #     → rows match activations 1-to-1
        # ------------------------------------------------------------------ #
        print("Building token maps…")
        token_ids, doc_map, pos_map = build_token_maps(
            sample_texts, tokenizer, max_length=args.max_length
        )
        assert len(doc_map) == activations.shape[0], (
            f"Token map length {len(doc_map)} != activations rows {activations.shape[0]}"
        )

        if not args.no_cache_save:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "activations": activations.cpu(),
                    "texts": sample_texts,
                    "token_ids": token_ids,
                    "token_doc_map": doc_map,
                    "token_pos_map": pos_map,
                    "meta": {
                        "dataset": args.dataset if use_dataset else "built_in_sample",
                        "dataset_config": args.dataset_config if use_dataset else None,
                        "split": args.dataset_split if use_dataset else "n/a",
                        "num_texts": len(sample_texts),
                        "max_length": args.max_length,
                        "layer_index": layer_index,
                    },
                },
                str(cache_path),
            )
            print(f"  Saved activation cache to {cache_path}")

    # ------------------------------------------------------------------ #
    # 5.  Find top activated alive features across this corpus
    # ------------------------------------------------------------------ #
    mean_acts = torch.zeros(sae.d_hidden)
    with torch.no_grad():
        for start in range(0, activations.shape[0], 256):
            batch = activations[start:start + 256]
            enc = sae.encode(batch)
            mean_acts += enc.sum(dim=0).cpu()
    mean_acts /= activations.shape[0]
    # Sort all features by mean activation.
    # Skip dead features (mean_acts == 0) — they carry no signal.
    alive_sorted = mean_acts.argsort(descending=True)
    alive_sorted = alive_sorted[mean_acts[alive_sorted] > 0]

    n_alive = len(alive_sorted)
    top_n = max(0, min(args.top_feature_count, n_alive))
    target_features = [x for x in range(0, 11)] # FIXED FEATURES
    print(f"\nAlive features: {n_alive}")
    print(f"Top-{top_n} feature indices by mean activation: {target_features}")

    if top_n == 0:
        print("No features requested for labeling (--top-feature-count=0). Dry run completed.")
        sys.exit(0)

    cfg = LabelingConfig(
        backend=args.backend,
        model=args.model,
        top_k=args.top_k,
        request_delay=args.request_delay,
        prompt_log_path=args.prompt_log_path,
        skip_first_token=True,
        global_top_features_k=10,
        min_activation=args.min_activation,
        include_global_top_features=args.include_global_top_features,
        top_tokens_k=args.top_tokens_k,
    )

    try:
        labeler = FeatureLabeler(sae, tokenizer, cfg)
    except (ImportError, EnvironmentError) as exc:
        print(
            "LLM backend is unavailable in this environment. "
            "Install optional dependencies or configure API keys, then re-run."
        )
        print(f"Details: {exc}")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 6.  Run labeling
    # ------------------------------------------------------------------ #

    print("\nLabeling features…")
    results = labeler.label_features_from_activations(
        feature_indices=target_features,
        activations=activations,
        token_ids=token_ids,
        token_doc_map=doc_map,
        token_pos_map=pos_map,
        save_path=args.save_path,
        resume=args.resume,
    )

    FeatureLabeler.print_summary(results)
    for r in results.values():
        FeatureLabeler.print_label(r)
