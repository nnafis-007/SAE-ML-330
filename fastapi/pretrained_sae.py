from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


DEFAULT_PRETRAINED_MODELS: List[Tuple[str, str]] = [
    ("gpt2-small-res-jb", "blocks.8.hook_resid_pre"),
]
DEFAULT_PRETRAINED_MODEL_ID = "gpt2-small-res-jb:blocks.8.hook_resid_pre"

_MODEL_ID_PATTERN = re.compile(r"^[^:]+:[^:]+$")
_LAYER_PATTERN = re.compile(r"blocks\.(\d+)\.")


@dataclass(frozen=True)
class SAEMetadata:
    model_id: str
    release: str
    sae_id: str
    hook_name: str
    model_name: str
    layer_index: int
    d_model: int
    d_hidden: int
    context_size: int
    prepend_bos: bool


@dataclass
class SAEBundle:
    metadata: SAEMetadata
    sae: Any
    model: HookedTransformer
    tokenizer: Any


class PretrainedSAEStore:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._bundle_cache: Dict[str, SAEBundle] = {}
        self._meta_cache: Dict[str, SAEMetadata] = {}

    @staticmethod
    def parse_model_id(model_id: str) -> Tuple[str, str]:
        model_id = model_id.strip()
        if not _MODEL_ID_PATTERN.match(model_id):
            raise ValueError("model_id must be in 'release:sae_id' format")
        if model_id != DEFAULT_PRETRAINED_MODEL_ID:
            raise ValueError(
                "Only the default pretrained model is currently supported: "
                f"{DEFAULT_PRETRAINED_MODEL_ID}"
            )
        release, sae_id = model_id.split(":", 1)
        return release, sae_id

    @staticmethod
    def allowed_model_ids() -> List[str]:
        return [DEFAULT_PRETRAINED_MODEL_ID]

    @staticmethod
    def is_allowed_model_id(model_id: str) -> bool:
        return model_id.strip() == DEFAULT_PRETRAINED_MODEL_ID

    @staticmethod
    def build_model_id(release: str, sae_id: str) -> str:
        return f"{release}:{sae_id}"

    def list_default_models(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for release, sae_id in DEFAULT_PRETRAINED_MODELS:
            model_id = self.build_model_id(release, sae_id)
            meta = self._meta_cache.get(model_id)
            if meta is not None:
                rows.append(
                    {
                        "id": meta.model_id,
                        "name": meta.model_id,
                        "release": meta.release,
                        "sae_id": meta.sae_id,
                        "hook_name": meta.hook_name,
                        "layer_index": meta.layer_index,
                        "d_model": meta.d_model,
                        "d_hidden": meta.d_hidden,
                        "context_size": meta.context_size,
                        "prepend_bos": meta.prepend_bos,
                        "model_name": meta.model_name,
                        "source": "sae-lens",
                    }
                )
            else:
                rows.append(
                    {
                        "id": model_id,
                        "name": model_id,
                        "release": release,
                        "sae_id": sae_id,
                        "hook_name": sae_id,
                        "layer_index": self._infer_layer_index(sae_id),
                        "d_model": None,
                        "d_hidden": None,
                        "context_size": None,
                        "prepend_bos": None,
                        "model_name": self._infer_release_model_name(release),
                        "source": "sae-lens",
                    }
                )
        return rows

    def get_bundle(self, model_id: str) -> SAEBundle:
        if model_id in self._bundle_cache:
            return self._bundle_cache[model_id]

        release, sae_id = self.parse_model_id(model_id)
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=self.device)
        if isinstance(sae, tuple):
            sae = sae[0]

        # Keep compatibility with code that expects these attrs.
        d_model = int(getattr(sae.cfg, "d_in", getattr(sae.cfg, "d_model", sae.W_enc.shape[1])))
        d_hidden = int(getattr(sae.cfg, "d_sae", getattr(sae.cfg, "d_hidden", sae.W_enc.shape[0])))
        setattr(sae, "d_model", d_model)
        setattr(sae, "d_hidden", d_hidden)

        hook_name = str(getattr(getattr(sae.cfg, "metadata", object()), "hook_name", sae_id))
        context_size = int(getattr(getattr(sae.cfg, "metadata", object()), "context_size", 128))
        prepend_bos = bool(getattr(getattr(sae.cfg, "metadata", object()), "prepend_bos", False))

        model_name = self._infer_model_name(release=release, sae=sae)
        model = HookedTransformer.from_pretrained(model_name, device=self.device)

        layer_match = _LAYER_PATTERN.search(hook_name)
        layer_index = int(layer_match.group(1)) if layer_match else self._infer_layer_index(sae_id)

        meta = SAEMetadata(
            model_id=model_id,
            release=release,
            sae_id=sae_id,
            hook_name=hook_name,
            model_name=model_name,
            layer_index=layer_index,
            d_model=d_model,
            d_hidden=d_hidden,
            context_size=context_size,
            prepend_bos=prepend_bos,
        )
        bundle = SAEBundle(metadata=meta, sae=sae, model=model, tokenizer=model.tokenizer)
        self._meta_cache[model_id] = meta
        self._bundle_cache[model_id] = bundle
        return bundle

    def get_metadata(self, model_id: str) -> SAEMetadata:
        if model_id in self._meta_cache:
            return self._meta_cache[model_id]
        bundle = self.get_bundle(model_id)
        return bundle.metadata

    @staticmethod
    def _infer_model_name(release: str, sae: Any) -> str:
        cfg_model_name = getattr(sae.cfg, "model_name", None)
        if cfg_model_name:
            return str(cfg_model_name)

        metadata = getattr(sae.cfg, "metadata", None)
        if metadata is not None:
            meta_model_name = getattr(metadata, "model_name", None)
            if meta_model_name:
                return str(meta_model_name)

        # Fallback for common SAE-Lens release naming scheme.
        if release.startswith("gpt2-small"):
            return "gpt2-small"
        return "gpt2-small"

    @staticmethod
    def _infer_layer_index(sae_id: str) -> int:
        match = _LAYER_PATTERN.search(sae_id)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _infer_release_model_name(release: str) -> str:
        if release.startswith("gpt2-small"):
            return "gpt2-small"
        return "gpt2-small"
