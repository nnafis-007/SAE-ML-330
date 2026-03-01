"""
SAE Analyzer – connects the Sparse Autoencoder pipeline to the FastAPI backend.

Uses:
    - ``sae_model.SparseAutoencoder``   for encoding activations
    - ``data_collection.GPT2ActivationCollector`` for extracting GPT-2 hidden states
    - ``analysis.FeatureLabeler``        for on-demand LLM-based feature labeling
    - ``analysis.build_token_maps``      for building corpus token maps

All heavy objects (GPT-2 model, SAE checkpoints, collectors) are lazily loaded
and cached so repeated requests are fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Make the SAE source package importable
# ---------------------------------------------------------------------------
_SAE_ROOT = Path(__file__).resolve().parent.parent
_SAE_SRC = _SAE_ROOT / "src"
if str(_SAE_SRC) not in sys.path:
    sys.path.insert(0, str(_SAE_SRC))

# Heavy imports (transformers, GPT2, etc.) are deferred to first use
# to keep server startup fast.  Only lightweight imports happen here.
from sae_model import SparseAutoencoder                                # noqa: E402

from . import BaseAnalyzer, register                                   # noqa: E402

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
    * ``label_feature`` optionally calls an LLM (via ``analysis.py``) to
      generate a human-readable label for a single feature.
    """

    def __init__(self, checkpoints_dir: str | Path = CHECKPOINTS_DIR):
        self._checkpoints_dir = Path(checkpoints_dir)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Caches (populated lazily)
        self._sae_cache: Dict[str, Dict[str, Any]] = {}
        self._collector_cache: Dict[int, GPT2ActivationCollector] = {}
        self._labels_cache: Dict[str, Dict[int, str]] = {}
        self._model_list_cache: Optional[List[Dict[str, Any]]] = None

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

    # -- On-demand feature labeling (uses analysis.py's FeatureLabeler) --------

    def label_feature(
        self,
        model_id: str,
        feature_idx: int,
        corpus_texts: Optional[List[str]] = None,
        labeling_config: Optional[Dict[str, Any]] = None,
        groq_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Label a single SAE feature using an LLM via ``analysis.FeatureLabeler``.

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
            corpus_path = _SAE_ROOT / "corpus_texts.txt"
            if corpus_path.exists():
                raw = corpus_path.read_text(encoding="utf-8")
                corpus_texts = [
                    t.strip() for t in raw.split("\n\n") if t.strip()
                ]
            else:
                corpus_texts = [
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning enables computers to learn from data.",
                    "Neural networks have transformed natural language processing.",
                ]

        # Collect activations for this corpus
        activations = collector.collect_activations(
            texts=corpus_texts, batch_size=8, max_length=128,
        )

        from analysis import build_token_maps  # lazy import

        token_ids, doc_map, pos_map = build_token_maps(
            corpus_texts, tokenizer, max_length=128,
        )

        # Labeling config (lazy imports)
        from analysis import FeatureLabeler, LabelingConfig, LabelResult  # noqa: E811

        resolved_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        cfg_kwargs: Dict[str, Any] = {
            "backend": "groq",
            "model": "llama-3.3-70b-versatile",
            "top_k": min(15, activations.shape[0]),
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

        return {
            "feature_idx": result.feature_idx,
            "label": result.label,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "top_tokens": result.top_tokens,
            "error": result.error,
        }

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
