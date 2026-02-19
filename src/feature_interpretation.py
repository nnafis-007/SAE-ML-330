"""
Statistical & Semantic Feature Interpretation for Sparse Autoencoders
=====================================================================

This module answers the question: **"What does SAE feature X *mean*?"**

It combines three complementary approaches:

1. **Statistical Inference** (PMI + Chi-squared)
   - Measures which tokens are *disproportionately* associated with each
     feature relative to the corpus background frequency.
   - Pointwise Mutual Information (PMI) surfaces distinctive tokens.
   - Chi-squared test verifies the association is statistically significant.

2. **Decoder Weight Projection (Logit Lens)**
   - Each feature's decoder column is a direction in the residual stream.
   - Projecting this onto GPT-2's unembedding matrix tells us which tokens
     the feature *causally pushes the model toward predicting*.
   - This is the gold standard in mechanistic interpretability (Anthropic,
     "Towards Monosemanticity"; Neel Nanda et al.).

3. **Linguistic Category Analysis**
   - Tags all activating tokens with POS tags (noun, verb, …).
   - Computes category distributions and tests whether the feature
     disproportionately activates on a specific category.

The module produces a human-readable **FeatureInterpretation** report per
feature and an aggregate summary across all analyzed features.

Theoretical References
----------------------
- PMI: Church & Hanks (1990), "Word association norms, mutual information,
  and lexicography"
- Logit Lens: nostalgebraist (2020), later formalised by Dar et al. (2022)
  "Analyzing Transformers in Embedding Space"
- SAE Interpretability: Cunningham et al. (2023) "Sparse Autoencoders Find
  Highly Interpretable Features in Language Models";
  Bricken et al. (2023) "Towards Monosemanticity" (Anthropic)

Usage
-----
::

    from feature_interpretation import FeatureInterpreter, InterpretationConfig
    import torch

    # Load your trained SAE & GPT-2 model
    sae = ...          # SparseAutoencoder
    gpt2_model = ...   # GPT2LMHeadModel
    tokenizer = ...    # GPT2Tokenizer
    activations = torch.load("activations.pt")  # (N, d_model)

    cfg = InterpretationConfig(top_k=30, significance_level=0.01)
    interp = FeatureInterpreter(sae, gpt2_model, tokenizer, cfg)

    # Interpret a single feature
    report = interp.interpret_feature(feature_idx=42, activations=activations)
    interp.print_report(report)

    # Interpret multiple features and save
    reports = interp.interpret_features(
        feature_indices=[42, 100, 7949],
        activations=activations,
        save_path="interpretation_reports.json",
    )
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Optional heavy imports: graceful fallback
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    GPT2LMHeadModel = None  # type: ignore
    GPT2Tokenizer = None    # type: ignore

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sae_model import SparseAutoencoder


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InterpretationConfig:
    """
    Hyper-parameters for the full interpretation pipeline.

    Attributes
    ----------
    top_k : int
        Number of top-activating token positions to analyze per feature.
    activation_threshold : float
        Minimum activation value for a position to count as "active".
        Defaults to 0.0 (any positive ReLU output counts).
    significance_level : float
        p-value threshold for Chi-squared test.  Associations with
        p > significance_level are marked as NOT significant.
    min_token_count : int
        Ignore tokens that appear fewer than this many times in the
        corpus when computing PMI (prevents noisy high-PMI singletons).
    logit_lens_top_k : int
        Number of top vocabulary tokens to report from the logit lens.
    batch_size : int
        Batch size for SAE encoding (memory control).
    pos_tag : bool
        Whether to run POS-tag analysis.  Requires ``nltk``.
    context_window : int
        Number of tokens before/after the activating token to include
        in the context snippet.
    """

    top_k: int = 50
    activation_threshold: float = 0.0
    significance_level: float = 0.01
    min_token_count: int = 2
    logit_lens_top_k: int = 20
    batch_size: int = 512
    pos_tag: bool = True
    context_window: int = 8


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class TokenAssociation:
    """Result of PMI + Chi-squared for a single (feature, token) pair."""
    token: str
    pmi: float                  # Pointwise Mutual Information (bits)
    chi2_stat: float            # Chi-squared statistic
    p_value: float              # p-value for the chi-squared test
    significant: bool           # p_value < significance_level
    count_active: int           # Times this token appears when feature fires
    count_corpus: int           # Times this token appears in the full corpus
    prob_given_active: float    # P(token | feature active)
    prob_corpus: float          # P(token) in full corpus


@dataclass
class LogitLensResult:
    """Tokens most boosted by a feature's decoder direction."""
    token: str
    logit_value: float          # Raw dot-product (logit boost)
    rank: int                   # 1-indexed rank


@dataclass
class POSDistribution:
    """Distribution of POS tags among a feature's top activations."""
    tag: str
    count: int
    fraction: float
    corpus_fraction: float      # Background POS rate in corpus
    enrichment: float           # fraction / corpus_fraction
    p_value: float              # Binomial test p-value


@dataclass
class FeatureInterpretation:
    """
    Complete interpretation report for one SAE feature.

    This is the main output of the pipeline, containing statistical,
    causal (logit-lens), and linguistic analyses.
    """
    feature_idx: int

    # --- Activation summary ---
    activation_rate: float              # Fraction of corpus positions where feature fires
    mean_activation: float              # Mean activation when active
    max_activation: float

    # --- Statistical: PMI + Chi-squared ---
    top_token_associations: List[TokenAssociation]
    summary_tokens: List[str]           # Top-5 most distinctive tokens (quick glance)

    # --- Causal: Logit Lens ---
    logit_lens_results: List[LogitLensResult]
    logit_lens_summary: str             # One-line summary

    # --- Linguistic: POS tags ---
    pos_distribution: List[POSDistribution]
    dominant_pos: str                   # Most enriched POS tag

    # --- Composite interpretation ---
    interpretation: str                 # Human-readable paragraph
    confidence: str                     # "high" / "medium" / "low"

    # --- Top activating contexts ---
    top_contexts: List[Dict[str, Any]]  # [{token, context, activation_value}, ...]


# ============================================================================
# Core: FeatureInterpreter
# ============================================================================

class FeatureInterpreter:
    """
    Interprets SAE features through statistical, causal, and linguistic
    analysis.

    Parameters
    ----------
    sae : SparseAutoencoder
        Trained SAE.
    gpt2_model : GPT2LMHeadModel
        GPT-2 model (needed for logit lens — the unembedding matrix).
    tokenizer : GPT2Tokenizer
        Tokenizer used during activation collection.
    cfg : InterpretationConfig
        Pipeline configuration.
    device : str
        Torch device.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        gpt2_model,
        tokenizer,
        cfg: InterpretationConfig = InterpretationConfig(),
        device: Optional[str] = None,
    ):
        self.sae = sae
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae.to(self.device).eval()

        # Cache the unembedding matrix (logit lens)
        if gpt2_model is not None:
            # GPT-2's lm_head projects residual stream → vocab logits
            # For GPT-2 specifically, lm_head.weight is tied to wte.weight
            self._W_U = gpt2_model.lm_head.weight.detach().float()  # (vocab, d_model)
        else:
            self._W_U = None

    # ------------------------------------------------------------------
    # 1. STATISTICAL INFERENCE: PMI + Chi-squared
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_token_associations(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        token_ids_flat: List[int],
    ) -> Tuple[List[TokenAssociation], Dict[str, Any]]:
        """
        Compute PMI and Chi-squared test for every token w.r.t. a feature.

        Theory
        ------
        Let F be the event "feature fires" and T be the event "token = t".

        PMI(F, T) = log2( P(T | F) / P(T) )

        - P(T | F) = (# times token t appears when feature active) /
                      (# total active positions)
        - P(T)     = (# times token t appears in corpus) /
                      (# total positions in corpus)

        High PMI means the token is *much more likely* to co-occur with the
        feature than expected by chance.

        The Chi-squared contingency test then evaluates whether the observed
        co-occurrence table could have arisen by chance under the null
        hypothesis of independence.

        Parameters
        ----------
        feature_idx : int
        activations : torch.Tensor
            Shape (N, d_model).
        token_ids_flat : List[int]
            Token ID for each of the N activation rows.

        Returns
        -------
        associations : List[TokenAssociation]
            Sorted by PMI descending (only significant ones first).
        stats_summary : dict
            Aggregate statistics (total active, etc.).
        """
        cfg = self.cfg
        N = activations.shape[0]
        assert len(token_ids_flat) == N, (
            f"token_ids_flat length ({len(token_ids_flat)}) must match "
            f"activations rows ({N})"
        )

        # --- Step 1: Get feature activation values for all positions ---
        feat_vals = torch.zeros(N, dtype=torch.float32)
        for start in range(0, N, cfg.batch_size):
            end = min(start + cfg.batch_size, N)
            batch = activations[start:end].to(self.device)
            encoded = self.sae.encode(batch)
            feat_vals[start:end] = encoded[:, feature_idx].cpu()

        active_mask = feat_vals > cfg.activation_threshold
        n_active = int(active_mask.sum().item())
        n_inactive = N - n_active

        if n_active == 0:
            return [], {"n_active": 0, "n_total": N, "activation_rate": 0.0}

        # --- Step 2: Count token occurrences in active vs corpus ---
        corpus_counts: Counter = Counter(token_ids_flat)
        active_counts: Counter = Counter()
        for i in range(N):
            if active_mask[i]:
                active_counts[token_ids_flat[i]] += 1

        # --- Step 3: Compute PMI and Chi-squared for each token ---
        associations: List[TokenAssociation] = []
        p_active = n_active / N  # P(feature active)

        for tok_id, count_corpus in corpus_counts.items():
            if count_corpus < cfg.min_token_count:
                continue

            count_active = active_counts.get(tok_id, 0)
            count_inactive_with_tok = count_corpus - count_active
            count_active_without_tok = n_active - count_active
            count_neither = n_inactive - count_inactive_with_tok

            # Probabilities
            p_token = count_corpus / N
            p_token_given_active = count_active / n_active if n_active > 0 else 0.0

            # PMI (in bits)
            if p_token_given_active > 0 and p_token > 0:
                pmi = math.log2(p_token_given_active / p_token)
            else:
                pmi = float("-inf")

            # Chi-squared test on 2x2 contingency table:
            #                    token=t    token≠t
            # feature active  |  a         |  b     | n_active
            # feature inactive|  c         |  d     | n_inactive
            #                    count_corpus  N-count_corpus
            observed = np.array([
                [count_active, count_active_without_tok],
                [count_inactive_with_tok, count_neither],
            ])

            if HAS_SCIPY:
                try:
                    chi2, p_value, _, _ = sp_stats.chi2_contingency(
                        observed, correction=True
                    )
                except ValueError:
                    chi2, p_value = 0.0, 1.0
            else:
                # Fallback: manual chi-squared (no Yates correction)
                expected = np.outer(
                    observed.sum(axis=1), observed.sum(axis=0)
                ) / N
                expected = np.maximum(expected, 1e-10)
                chi2 = float(((observed - expected) ** 2 / expected).sum())
                # Approximate p-value using survival function (1 dof)
                # Without scipy we just flag large chi2 values
                p_value = math.exp(-chi2 / 2) if chi2 < 700 else 0.0

            tok_str = self.tokenizer.decode([tok_id]).strip()
            if not tok_str:
                tok_str = f"<id:{tok_id}>"

            associations.append(TokenAssociation(
                token=tok_str,
                pmi=pmi,
                chi2_stat=chi2,
                p_value=p_value,
                significant=p_value < cfg.significance_level,
                count_active=count_active,
                count_corpus=count_corpus,
                prob_given_active=p_token_given_active,
                prob_corpus=p_token,
            ))

        # Sort: significant associations first (by PMI desc), then non-sig
        associations.sort(
            key=lambda a: (-int(a.significant), -a.pmi)
        )

        stats_summary = {
            "n_active": n_active,
            "n_total": N,
            "activation_rate": n_active / N,
            "n_unique_active_tokens": len(active_counts),
            "n_significant_associations": sum(
                1 for a in associations if a.significant
            ),
        }

        return associations, stats_summary

    # ------------------------------------------------------------------
    # 2. LOGIT LENS: Decoder weight → vocabulary projection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def logit_lens(self, feature_idx: int) -> List[LogitLensResult]:
        """
        Project the feature's decoder direction onto the unembedding matrix.

        Theory
        ------
        The decoder column d_f = W_dec[:, f] is the feature's direction in
        the residual stream.  When the feature activates with magnitude a,
        it adds a * d_f to the residual stream.  The model's prediction
        layer computes logits = W_U @ residual, so the feature's *marginal
        contribution* to each token's logit is:

            Δlogit_t = W_U[t, :] · d_f

        Tokens with the highest Δlogit are what the feature pushes the model
        toward predicting.  This is a *causal* interpretation: this feature
        *makes the model more likely to output these tokens*.

        Returns
        -------
        List[LogitLensResult]  sorted by logit value descending.
        """
        if self._W_U is None:
            warnings.warn(
                "GPT-2 model not provided — logit lens analysis skipped."
            )
            return []

        # Get decoder column for this feature
        if self.sae.use_tied_weights:
            d_f = self.sae.W_enc[feature_idx, :].detach().float()  # (d_model,)
        else:
            d_f = self.sae.W_dec[:, feature_idx].detach().float()  # (d_model,)

        # Project onto unembedding: (vocab, d_model) @ (d_model,) → (vocab,)
        logits = self._W_U @ d_f.to(self._W_U.device)

        # Top-k
        k = min(self.cfg.logit_lens_top_k, logits.shape[0])
        top_vals, top_ids = torch.topk(logits, k)

        results = []
        for rank, (val, tok_id) in enumerate(
            zip(top_vals.tolist(), top_ids.tolist()), start=1
        ):
            tok_str = self.tokenizer.decode([tok_id]).strip()
            if not tok_str:
                tok_str = f"<id:{tok_id}>"
            results.append(LogitLensResult(
                token=tok_str, logit_value=val, rank=rank
            ))

        return results

    # ------------------------------------------------------------------
    # 3. POS TAG ANALYSIS
    # ------------------------------------------------------------------

    def pos_tag_analysis(
        self,
        active_tokens: List[str],
        corpus_tokens: List[str],
    ) -> List[POSDistribution]:
        """
        Compute POS-tag distribution for the feature's active tokens and
        compare against the corpus background.

        Uses NLTK's averaged perceptron tagger.  Falls back gracefully if
        NLTK is not installed.

        Theory
        ------
        For each POS tag T, we compute:
        - fraction_active  = P(T | feature active)
        - fraction_corpus  = P(T | corpus)
        - enrichment       = fraction_active / fraction_corpus

        A binomial test checks whether the observed count of tag T among
        n_active tokens is significantly higher than expected under the
        corpus base rate.

        Returns
        -------
        List[POSDistribution]  sorted by enrichment descending.
        """
        if not self.cfg.pos_tag:
            return []

        try:
            import nltk
            # Ensure the POS tagger data is available
            try:
                nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            except LookupError:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger", quiet=True)
        except ImportError:
            warnings.warn("NLTK not installed — POS analysis skipped.")
            return []

        # Tag tokens
        active_tagged = nltk.pos_tag(active_tokens)
        corpus_tagged = nltk.pos_tag(corpus_tokens)

        # Count POS tags
        active_pos_counts: Counter = Counter(tag for _, tag in active_tagged)
        corpus_pos_counts: Counter = Counter(tag for _, tag in corpus_tagged)

        n_active = len(active_tokens)
        n_corpus = len(corpus_tokens)

        if n_active == 0 or n_corpus == 0:
            return []

        results: List[POSDistribution] = []
        all_tags = set(active_pos_counts.keys()) | set(corpus_pos_counts.keys())

        for tag in all_tags:
            count = active_pos_counts.get(tag, 0)
            frac = count / n_active
            corpus_frac = corpus_pos_counts.get(tag, 0) / n_corpus
            enrichment = frac / corpus_frac if corpus_frac > 0 else float("inf")

            # Binomial test: is count/n_active significantly greater than
            # corpus_frac?
            if HAS_SCIPY and n_active > 0:
                binom_result = sp_stats.binomtest(
                    count, n_active, corpus_frac, alternative="greater"
                )
                p_val = binom_result.pvalue
            else:
                p_val = 1.0

            results.append(POSDistribution(
                tag=tag,
                count=count,
                fraction=frac,
                corpus_fraction=corpus_frac,
                enrichment=enrichment,
                p_value=p_val,
            ))

        results.sort(key=lambda x: -x.enrichment)
        return results

    # ------------------------------------------------------------------
    # 4. Context extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_top_contexts(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        token_ids_flat: List[int],
        token_doc_ids: Optional[List[int]] = None,
        all_token_ids: Optional[List[List[int]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """
        Extract top-k activating contexts plus lists of active/corpus tokens.

        Returns
        -------
        contexts : list of dicts with keys {token, context, activation_value}
        active_tokens : decoded token strings for positions where feature fires
        corpus_tokens : decoded token strings for ALL positions (for POS baseline)
        """
        cfg = self.cfg
        N = activations.shape[0]

        # Encode
        feat_vals = torch.zeros(N, dtype=torch.float32)
        for start in range(0, N, cfg.batch_size):
            end = min(start + cfg.batch_size, N)
            batch = activations[start:end].to(self.device)
            encoded = self.sae.encode(batch)
            feat_vals[start:end] = encoded[:, feature_idx].cpu()

        # Top-k activating positions
        k = min(cfg.top_k, N)
        top_vals, top_idxs = torch.topk(feat_vals, k)

        contexts: List[Dict[str, Any]] = []
        active_tokens_list: List[str] = []
        cw = cfg.context_window

        for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            tok_id = token_ids_flat[idx]
            tok_str = self.tokenizer.decode([tok_id]).strip()
            active_tokens_list.append(tok_str)

            # Build context string
            if token_doc_ids is not None and all_token_ids is not None:
                # We have per-document token lists
                doc_id = token_doc_ids[idx]
                # Find position within document
                # Count how many tokens from this doc came before idx
                pos_in_doc = 0
                count = 0
                for i in range(idx + 1):
                    if token_doc_ids[i] == doc_id:
                        if i == idx:
                            pos_in_doc = count
                        count += 1

                toks = all_token_ids[doc_id]
                lo = max(0, pos_in_doc - cw)
                hi = min(len(toks), pos_in_doc + cw + 1)
                prefix = self.tokenizer.decode(toks[lo:pos_in_doc])
                target = self.tokenizer.decode(toks[pos_in_doc:pos_in_doc + 1])
                suffix = self.tokenizer.decode(toks[pos_in_doc + 1:hi])
                context_str = f"{prefix}>>>{target.strip()}<<<{suffix}"
            else:
                # No doc structure — just show the token
                lo = max(0, idx - cw)
                hi = min(N, idx + cw + 1)
                prefix_ids = token_ids_flat[lo:idx]
                suffix_ids = token_ids_flat[idx + 1:hi]
                prefix = self.tokenizer.decode(prefix_ids)
                target = self.tokenizer.decode([tok_id])
                suffix = self.tokenizer.decode(suffix_ids)
                context_str = f"{prefix}>>>{target.strip()}<<<{suffix}"

            contexts.append({
                "token": tok_str,
                "context": context_str,
                "activation_value": val,
            })

        # All active tokens (for POS analysis)
        active_mask = feat_vals > cfg.activation_threshold
        all_active_tokens = [
            self.tokenizer.decode([token_ids_flat[i]]).strip()
            for i in range(N) if active_mask[i]
        ]

        # Corpus tokens (subsample for efficiency)
        max_corpus_sample = min(N, 5000)
        step = max(1, N // max_corpus_sample)
        corpus_tokens = [
            self.tokenizer.decode([token_ids_flat[i]]).strip()
            for i in range(0, N, step)
        ]

        return contexts, all_active_tokens, corpus_tokens

    # ------------------------------------------------------------------
    # 5. Composite interpretation generation
    # ------------------------------------------------------------------

    def _generate_interpretation(
        self,
        feature_idx: int,
        associations: List[TokenAssociation],
        logit_results: List[LogitLensResult],
        pos_dist: List[POSDistribution],
        activation_rate: float,
    ) -> Tuple[str, str]:
        """
        Synthesize a human-readable interpretation from all analyses.

        Returns
        -------
        interpretation : str
        confidence : "high" | "medium" | "low"
        """
        lines = []
        confidence_score = 0  # accumulate evidence

        # --- From PMI ---
        sig_assoc = [a for a in associations if a.significant]
        if sig_assoc:
            top5 = [a.token for a in sig_assoc[:5]]
            lines.append(
                f"Statistically associated tokens (PMI): "
                f"{', '.join(repr(t) for t in top5)}"
            )
            # Check if the top tokens share a theme
            if len(sig_assoc) >= 3:
                confidence_score += 1

        # --- From Logit Lens ---
        if logit_results:
            top5_logit = [r.token for r in logit_results[:5]]
            lines.append(
                f"Decoder projection (logit lens) boosts: "
                f"{', '.join(repr(t) for t in top5_logit)}"
            )
            confidence_score += 1

        # --- From POS ---
        enriched_pos = [
            p for p in pos_dist
            if p.enrichment > 1.5 and p.p_value < 0.05 and p.count >= 3
        ]
        if enriched_pos:
            dominant = enriched_pos[0]
            tag_name = _POS_TAG_NAMES.get(dominant.tag, dominant.tag)
            lines.append(
                f"Dominant POS: {tag_name} ({dominant.tag}) — "
                f"{dominant.fraction:.0%} of active tokens vs "
                f"{dominant.corpus_fraction:.0%} corpus baseline "
                f"({dominant.enrichment:.1f}x enrichment, p={dominant.p_value:.2e})"
            )
            confidence_score += 1

        # --- Activation rate context ---
        if activation_rate < 0.01:
            lines.append(f"Very sparse feature (fires on {activation_rate:.2%} of tokens)")
        elif activation_rate > 0.20:
            lines.append(f"Broadly active feature ({activation_rate:.2%} activation rate)")

        interpretation = " | ".join(lines) if lines else "Insufficient evidence."

        if confidence_score >= 3:
            confidence = "high"
        elif confidence_score >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return interpretation, confidence

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def interpret_feature(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        token_ids_flat: Optional[List[int]] = None,
        token_doc_ids: Optional[List[int]] = None,
        all_token_ids: Optional[List[List[int]]] = None,
    ) -> FeatureInterpretation:
        """
        Run the full interpretation pipeline on one feature.

        Parameters
        ----------
        feature_idx : int
            Index of the SAE feature to interpret.
        activations : torch.Tensor
            Shape (N, d_model).  GPT-2 residual-stream activations.
        token_ids_flat : List[int], optional
            Token ID for each row of ``activations``.  If not provided,
            statistical analysis is skipped and only logit lens runs.
        token_doc_ids : List[int], optional
            Document index per activation row (for context strings).
        all_token_ids : List[List[int]], optional
            Per-document token ID lists (for context strings).

        Returns
        -------
        FeatureInterpretation
        """
        N = activations.shape[0]

        # --- Basic activation statistics ---
        feat_vals = torch.zeros(N, dtype=torch.float32)
        for start in range(0, N, self.cfg.batch_size):
            end = min(start + self.cfg.batch_size, N)
            batch = activations[start:end].to(self.device)
            encoded = self.sae.encode(batch)
            feat_vals[start:end] = encoded[:, feature_idx].cpu()

        active_mask = feat_vals > self.cfg.activation_threshold
        n_active = int(active_mask.sum().item())
        activation_rate = n_active / N if N > 0 else 0.0
        mean_act = feat_vals[active_mask].mean().item() if n_active > 0 else 0.0
        max_act = feat_vals.max().item()

        # --- 1. Statistical Inference (PMI + Chi-squared) ---
        associations: List[TokenAssociation] = []
        if token_ids_flat is not None:
            associations, _ = self.compute_token_associations(
                feature_idx, activations, token_ids_flat
            )

        # --- 2. Logit Lens ---
        logit_results = self.logit_lens(feature_idx)

        # --- 3. Context extraction + POS analysis ---
        contexts: List[Dict[str, Any]] = []
        pos_dist: List[POSDistribution] = []
        if token_ids_flat is not None:
            contexts, active_tokens, corpus_tokens = self._get_top_contexts(
                feature_idx, activations, token_ids_flat,
                token_doc_ids, all_token_ids,
            )
            if self.cfg.pos_tag and active_tokens:
                pos_dist = self.pos_tag_analysis(active_tokens, corpus_tokens)

        # --- 4. Composite interpretation ---
        interpretation, confidence = self._generate_interpretation(
            feature_idx, associations, logit_results, pos_dist, activation_rate
        )

        # --- Logit lens summary ---
        if logit_results:
            top3 = ", ".join(repr(r.token) for r in logit_results[:3])
            logit_summary = f"Boosts prediction of: {top3}"
        else:
            logit_summary = "N/A (no GPT-2 model)"

        # --- Dominant POS ---
        enriched_pos = [
            p for p in pos_dist if p.enrichment > 1.5 and p.count >= 2
        ]
        dominant_pos = enriched_pos[0].tag if enriched_pos else "N/A"

        # --- Summary tokens ---
        sig_tokens = [a.token for a in associations if a.significant][:5]

        return FeatureInterpretation(
            feature_idx=feature_idx,
            activation_rate=activation_rate,
            mean_activation=mean_act,
            max_activation=max_act,
            top_token_associations=associations[:30],  # keep top 30
            summary_tokens=sig_tokens,
            logit_lens_results=logit_results,
            logit_lens_summary=logit_summary,
            pos_distribution=pos_dist[:15],
            dominant_pos=dominant_pos,
            interpretation=interpretation,
            confidence=confidence,
            top_contexts=contexts[:15],
        )

    def interpret_features(
        self,
        feature_indices: List[int],
        activations: torch.Tensor,
        token_ids_flat: Optional[List[int]] = None,
        token_doc_ids: Optional[List[int]] = None,
        all_token_ids: Optional[List[List[int]]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[int, FeatureInterpretation]:
        """
        Interpret multiple features and optionally save results.

        Parameters
        ----------
        feature_indices : List[int]
        activations, token_ids_flat, token_doc_ids, all_token_ids
            Same as :meth:`interpret_feature`.
        save_path : Optional[str]
            If provided, save JSON report here.

        Returns
        -------
        Dict[int, FeatureInterpretation]
        """
        results: Dict[int, FeatureInterpretation] = {}

        for feat_idx in tqdm(feature_indices, desc="Interpreting features"):
            report = self.interpret_feature(
                feat_idx, activations, token_ids_flat,
                token_doc_ids, all_token_ids,
            )
            results[feat_idx] = report

        if save_path:
            self.save_reports(results, save_path)

        return results

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: FeatureInterpretation) -> None:
        """Pretty-print a single interpretation report."""
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  FEATURE {report.feature_idx} INTERPRETATION")
        print(sep)
        print(f"  Confidence  : {report.confidence}")
        print(f"  Activation  : rate={report.activation_rate:.2%}, "
              f"mean={report.mean_activation:.4f}, max={report.max_activation:.4f}")
        print()

        # Interpretation
        print(f"  [INTERPRETATION]")
        print(f"  {report.interpretation}")
        print()

        # Statistical
        if report.top_token_associations:
            print(f"  [STATISTICAL: PMI + Chi-squared]")
            print(f"  {'Token':<20} {'PMI':>8} {'χ²':>10} {'p-value':>12} "
                  f"{'Sig':>4} {'P(t|f)':>8} {'P(t)':>8}")
            print(f"  {'-'*70}")
            for a in report.top_token_associations[:15]:
                sig_mark = "***" if a.significant else ""
                print(
                    f"  {a.token:<20} {a.pmi:>8.3f} {a.chi2_stat:>10.2f} "
                    f"{a.p_value:>12.2e} {sig_mark:>4} "
                    f"{a.prob_given_active:>8.4f} {a.prob_corpus:>8.4f}"
                )
            print()

        # Logit Lens
        if report.logit_lens_results:
            print(f"  [LOGIT LENS: Decoder Weight → Vocabulary]")
            print(f"  {report.logit_lens_summary}")
            print(f"  {'Rank':>4} {'Token':<20} {'Logit boost':>12}")
            print(f"  {'-'*40}")
            for r in report.logit_lens_results[:10]:
                print(f"  {r.rank:>4} {r.token:<20} {r.logit_value:>12.4f}")
            print()

        # POS
        if report.pos_distribution:
            print(f"  [POS TAG DISTRIBUTION]")
            print(f"  {'Tag':<6} {'Name':<20} {'Count':>5} {'Frac':>8} "
                  f"{'Corpus':>8} {'Enrich':>8} {'p-value':>12}")
            print(f"  {'-'*72}")
            for p in report.pos_distribution[:10]:
                name = _POS_TAG_NAMES.get(p.tag, "")
                print(
                    f"  {p.tag:<6} {name:<20} {p.count:>5} "
                    f"{p.fraction:>8.2%} {p.corpus_fraction:>8.2%} "
                    f"{p.enrichment:>8.1f}x {p.p_value:>12.2e}"
                )
            print()

        # Top contexts
        if report.top_contexts:
            print(f"  [TOP ACTIVATING CONTEXTS]")
            for i, ctx in enumerate(report.top_contexts[:8], 1):
                print(f"  {i:>3}. [{ctx['activation_value']:.3f}] {ctx['context']}")
            print()

        print(sep)

    @staticmethod
    def print_summary(results: Dict[int, FeatureInterpretation]) -> None:
        """Print compact summary table of all interpreted features."""
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"{'IDX':>6}  {'CONF':^8}  {'ACT RATE':>8}  "
              f"{'DOM POS':>8}  SUMMARY TOKENS / INTERPRETATION")
        print(sep)
        for feat_idx in sorted(results.keys()):
            r = results[feat_idx]
            toks = ", ".join(r.summary_tokens[:4]) if r.summary_tokens else "-"
            print(
                f"{feat_idx:>6}  {r.confidence:^8}  "
                f"{r.activation_rate:>8.2%}  {r.dominant_pos:>8}  {toks}"
            )
        print(sep)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_reports(
        results: Dict[int, FeatureInterpretation],
        path: str,
    ) -> None:
        """Save interpretation reports to JSON."""
        serialisable = {}
        for feat_idx, report in results.items():
            d = asdict(report)
            serialisable[str(feat_idx)] = d
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False, default=str)
        print(f"Interpretation reports saved to {path}")

    @staticmethod
    def load_reports(path: str) -> Dict[int, dict]:
        """Load interpretation reports from JSON (as plain dicts)."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}


# ============================================================================
# Utility: Build flat token ID list from activations + texts
# ============================================================================

def build_flat_token_ids(
    texts: List[str],
    tokenizer,
    max_length: int = 128,
) -> Tuple[List[int], List[int], List[List[int]]]:
    """
    Tokenize texts and return flat lists suitable for FeatureInterpreter.

    Returns
    -------
    token_ids_flat : List[int]
        Token ID for each activation row (matches activations 1-to-1).
    token_doc_ids : List[int]
        Document index for each activation row.
    all_token_ids : List[List[int]]
        Per-document token ID lists.
    """
    all_token_ids: List[List[int]] = []
    token_ids_flat: List[int] = []
    token_doc_ids: List[int] = []

    for doc_idx, text in enumerate(texts):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = enc["input_ids"][0].tolist()
        all_token_ids.append(ids)
        for tok_id in ids:
            token_ids_flat.append(tok_id)
            token_doc_ids.append(doc_idx)

    return token_ids_flat, token_doc_ids, all_token_ids


# ============================================================================
# POS tag name lookup
# ============================================================================

_POS_TAG_NAMES = {
    "CC": "Coordinating conj.",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition/subord.",
    "JJ": "Adjective",
    "JJR": "Adj. comparative",
    "JJS": "Adj. superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun (singular)",
    "NNS": "Noun (plural)",
    "NNP": "Proper noun (sg.)",
    "NNPS": "Proper noun (pl.)",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb comparative",
    "RBS": "Adverb superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb (base)",
    "VBD": "Verb (past tense)",
    "VBG": "Verb (gerund)",
    "VBN": "Verb (past part.)",
    "VBP": "Verb (non-3sg pres)",
    "VBZ": "Verb (3sg present)",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Wh-possessive",
    "WRB": "Wh-adverb",
    ".": "Sentence-final punct.",
    ",": "Comma",
    ":": "Colon/semicolon",
    "(": "Left bracket",
    ")": "Right bracket",
    "``": "Open quotation",
    "''": "Close quotation",
    "#": "Pound sign",
    "$": "Dollar sign",
}


# ============================================================================
# __main__: End-to-end demo
# ============================================================================

if __name__ == "__main__":
    """
    Standalone demo: interpret the top-5 most active features from a
    trained SAE + GPT-2 activations.

    Requires:
        - checkpoints/best_model.pt  (trained SAE)
        - activations.pt             (GPT-2 hidden states)

    Run:
        cd <project_root>
        python src/feature_interpretation.py
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from sae_model import SparseAutoencoder
    from data_collection import GPT2ActivationCollector

    # ------------------------------------------------------------------ #
    # 1.  Load SAE
    # ------------------------------------------------------------------ #
    print("Loading SAE checkpoint...")
    ckpt_path = Path("checkpoints/best_model.pt")
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found. Train a model first with run_sae.py")
        sys.exit(1)

    payload = torch.load(ckpt_path, map_location="cpu")
    hp = payload.get("hyperparameters", {})
    state = payload["model_state_dict"]
    d_model = hp.get("d_model", state["W_enc"].shape[1])
    d_hidden = hp.get("d_hidden", state["W_enc"].shape[0])
    l1_coeff = hp.get("l1_coeff", 3e-4)
    layer_index = hp.get("layer_index", 8)

    sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden, l1_coeff=l1_coeff)
    sae.load_state_dict(state)
    sae.eval()
    print(f"  SAE: d_model={d_model}, d_hidden={d_hidden}")

    # ------------------------------------------------------------------ #
    # 2.  Load GPT-2 (for logit lens)
    # ------------------------------------------------------------------ #
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2 for logit lens...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()

    # ------------------------------------------------------------------ #
    # 3.  Sample corpus + activations
    # ------------------------------------------------------------------ #
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Einstein's field equations G_uv + Lambda*g_uv = 8*pi*G*T_uv.",
        "Heavy rain and strong winds are expected for the coastal regions.",
        "Voltage is equal to current multiplied by resistance (Ohm's law).",
        "1, 1, 2, 3, 5, 8, 13, 21 — the Fibonacci sequence.",
        "def quicksort(arr): return arr if len(arr) <= 1 else ...",
        "import torch; x = torch.randn(10, 768)",
        "print('Hello World'); for i in range(10): print(i)",
        "React.useEffect(() => { console.log('mounted'); }, []);",
        "She whispered, 'I have a secret for you,' as the clock struck midnight.",
        "Why does the sun rise in the east and set in the west?",
        "Blueberry pancakes with maple syrup are a classic breakfast choice.",
        "Under the bridge Downtown, is where I drew some blood.",
        "Wait! Before you go, make sure you have your passport.",
        "To be, or not to be, that is the question.",
        "What is the airspeed velocity of an unladen swallow?",
        "JSON output format: { 'status': 'success', 'data': [] }",
        "0.003, 1.45, -0.72, 3.14159 are floating point numbers.",
        "The fundamental theorem of calculus relates antiderivatives to integrals.",
        "Blue whales are the largest animals ever known to have lived on Earth.",
        "Machine learning models can classify images with superhuman accuracy.",
        "The cat sat on the mat and stared at the bird outside.",
        "Parliament voted to approve the new climate change legislation.",
        "Neurons in the brain communicate through electrical and chemical signals.",
        "The stock market crashed after the unexpected interest rate hike.",
    ]

    print(f"Collecting GPT-2 activations from {len(sample_texts)} texts (layer {layer_index})...")
    collector = GPT2ActivationCollector(
        model_name="gpt2", layer_index=layer_index
    )
    activations = collector.collect_activations(
        texts=sample_texts, batch_size=4, max_length=128
    )
    print(f"  Collected {activations.shape[0]} activation vectors")

    # Build flat token IDs
    token_ids_flat, token_doc_ids, all_token_ids = build_flat_token_ids(
        sample_texts, tokenizer, max_length=128
    )
    assert len(token_ids_flat) == activations.shape[0], (
        f"Token map ({len(token_ids_flat)}) != activations ({activations.shape[0]})"
    )

    # ------------------------------------------------------------------ #
    # 4.  Find top features to interpret
    # ------------------------------------------------------------------ #
    print("Finding most active features...")
    mean_acts = torch.zeros(sae.d_hidden)
    with torch.no_grad():
        for start in range(0, activations.shape[0], 256):
            batch = activations[start:start + 256]
            enc = sae.encode(batch)
            mean_acts += enc.sum(dim=0).cpu()
    mean_acts /= activations.shape[0]

    top_feats = mean_acts.argsort(descending=True)[500:600].tolist()
    print(f"  Top-5 features by mean activation: {top_feats}")

    # ------------------------------------------------------------------ #
    # 5.  Run interpretation
    # ------------------------------------------------------------------ #
    cfg = InterpretationConfig(
        top_k=30,
        logit_lens_top_k=15,
        pos_tag=True,
    )
    interp = FeatureInterpreter(sae, gpt2_model, tokenizer, cfg)

    reports = interp.interpret_features(
        feature_indices=top_feats,
        activations=activations,
        token_ids_flat=token_ids_flat,
        token_doc_ids=token_doc_ids,
        all_token_ids=all_token_ids,
        save_path="interpretation_reports.json",
    )

    # Print all reports
    for feat_idx in sorted(reports.keys()):
        FeatureInterpreter.print_report(reports[feat_idx])

    FeatureInterpreter.print_summary(reports)
    print("\nDone! Reports saved to interpretation_reports.json")
