"""
FastAPI backend for the SAE Interpretability UI.

Provides:
    GET  /models           – list available checkpoints
    GET  /analyzers        – list registered analysis methods
    POST /analyze          – run analysis on input text
    POST /label-feature    – on-demand LLM-based feature labeling (via analysis.py)
"""

import sys
import json
import threading
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Ensure project-root packages (e.g., analyzers/) are importable regardless
# of whether uvicorn is launched from project root or fastapi/.
_SAE_ROOT = Path(__file__).resolve().parent.parent
if str(_SAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAE_ROOT))

app = FastAPI(title="SAE Interpretability API")

_INTERPRETED_FEATURES_LOCK = threading.Lock()


def _safe_model_file_stem(model_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._")
    return stem or "unknown_model"


def _append_interpreted_feature_log(model_id: str, entry: Dict[str, Any]) -> None:
    """Append one interpreted-feature record to a JSON list on disk."""
    log_path = _SAE_ROOT / "logs" / f"ui_llm_interpreted_features_{_safe_model_file_stem(model_id)}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with _INTERPRETED_FEATURES_LOCK:
        payload: List[Dict[str, Any]] = []
        if log_path.exists():
            try:
                existing = json.loads(log_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    payload = existing
            except Exception:
                payload = []

        payload.append(entry)
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _load_labeled_features_for_model(model_id: str) -> List[Dict[str, Any]]:
    """Load persisted per-model feature labels from logs/feature_labels_<model>.json."""
    labels_path = _SAE_ROOT / "logs" / f"feature_labels_{_safe_model_file_stem(model_id)}.json"
    if not labels_path.exists():
        return []

    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse feature labels JSON: {exc}") from exc

    rows: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = enumerate(payload)
    else:
        return []

    for key, value in iterable:
        if not isinstance(value, dict):
            continue

        raw_feature_id = value.get("feature_idx", key)
        try:
            feature_id = int(raw_feature_id)
        except (TypeError, ValueError):
            continue

        rows.append({
            "feature_id": feature_id,
            "label": value.get("label") or f"Feature {feature_id}",
            "description": value.get("explanation") or value.get("description") or "No description available.",
            "confidence": value.get("confidence"),
            "updated_at": value.get("updated_at"),
        })

    rows.sort(key=lambda item: item["feature_id"])
    return rows

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Import analyzers – each module auto-registers on import
# ---------------------------------------------------------------------------
import sae_analyzer as _sae_reg  # noqa: F401, E402
import synonym_analyzer as _syn_reg  # noqa: F401, E402
import caps_analyzer as _caps_reg  # noqa: F401, E402
from analyzers import get_analyzer, list_analyzers, get_all_models  # noqa: E402


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_id: str
    analyzer: str = "sae"
    top_k: int = 10


class SynonymTestRequest(BaseModel):
    model_id: str
    clusters: Optional[List[str]] = None  # None = all clusters
    custom_words: Optional[List[str]] = None  # user-typed words to compare
    top_k: int = 1000


class CapsTestRequest(BaseModel):
    model_id: str
    words: Optional[List[str]] = None  # None = all words
    top_k: int = 30


class LabelFeatureRequest(BaseModel):
    model_id: str
    feature_idx: int
    analyzer: str = "sae"
    corpus_texts: Optional[List[str]] = None
    corpus_path: Optional[str] = None
    labeling_config: Optional[Dict[str, Any]] = None
    groq_api_key: Optional[str] = None  # falls back to GROQ_API_KEY env var


class BulkLabelFeaturesRequest(BaseModel):
    model_id: str
    feature_start: Optional[int] = Field(None, ge=0)
    feature_end: Optional[int] = Field(None, ge=0)
    feature_ids: Optional[List[int]] = None
    num_sentences: int = Field(200, ge=1)
    llm_top_k: int = Field(25, ge=1)
    min_activation: float = Field(0.0, ge=0.0)
    analyzer: str = "sae"
    corpus_path: Optional[str] = None
    labeling_config: Optional[Dict[str, Any]] = None
    groq_api_key: Optional[str] = None


class FeatureActivationsRequest(BaseModel):
    model_id: str
    feature_id: int
    corpus_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config: str = "local"
    split: str = "local"
    max_sentences: Optional[int] = None
    target_activating_examples: Optional[int] = None
    page: int = 1
    page_size: int = 25
    min_activation: float = 0.0
    text_field: Optional[str] = None
    max_length: int = 128
    seed: int = 0


class FeatureInfoRequest(BaseModel):
    model_id: str
    feature_id: int = Field(..., ge=0)
    analyzer: str = "sae"


class SentenceFeatureTraceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_id: str
    feature_id: int = Field(..., ge=0)
    min_activation: float = Field(0.0, ge=0.0)
    max_length: int = Field(512, ge=1, le=1024)


class SentenceFeatureToken(BaseModel):
    index: int
    text: str
    activation: float
    is_active: bool


class SentenceFeatureTraceResponse(BaseModel):
    model: str
    layer_index: int
    d_hidden: int
    feature_id: int
    feature_name: str
    feature_description: str
    active_token_count: int
    token_count: int
    max_activation: float
    min_activation: float
    tokens: List[SentenceFeatureToken]


class FeatureInfoResponse(BaseModel):
    model: str
    layer_index: int
    d_hidden: int
    feature_id: int
    feature_name: str
    feature_description: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/models")
def get_models(analyzer: Optional[str] = None):
    """Return checkpoint models, optionally filtered by analyzer name."""
    if analyzer:
        try:
            a = get_analyzer(analyzer)
        except KeyError:
            raise HTTPException(404, f"Analyzer '{analyzer}' not found")
        return {"models": a.list_models()}
    return {"models": get_all_models()}


@app.get("/labeled-features")
def get_labeled_features(model_id: str):
    """Return all persisted labeled features for a given model id."""
    model_id = (model_id or "").strip()
    if not model_id:
        raise HTTPException(400, "model_id is required")

    try:
        features = _load_labeled_features_for_model(model_id)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))

    return {
        "model_id": model_id,
        "count": len(features),
        "features": features,
    }


@app.get("/analyzers")
def get_available_analyzers():
    """Return registered analyzer names."""
    return {"analyzers": list_analyzers()}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Run analysis on the input text using the selected model."""
    try:
        a = get_analyzer(request.analyzer)
    except KeyError:
        raise HTTPException(404, f"Analyzer '{request.analyzer}' not found")

    if not request.text.strip():
        raise HTTPException(400, "Input text is empty. Please provide non-empty text for analysis.")

    try:
        result = a.analyze(
            text=request.text,
            model_id=request.model_id,
            top_k=request.top_k,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model not found: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")


@app.post("/synonym-test")
def synonym_test(request: SynonymTestRequest):
    """Run the synonym feature-overlap test on selected clusters or custom words."""
    try:
        a = get_analyzer("synonym")
    except KeyError:
        raise HTTPException(404, "Synonym analyzer not registered")

    # Custom-word mode takes priority
    if request.custom_words and len(request.custom_words) >= 2:
        try:
            result = a.analyze(
                text="",
                model_id=request.model_id,
                top_k=request.top_k,
                custom_words=request.custom_words,
            )
            return result
        except FileNotFoundError as exc:
            raise HTTPException(404, f"Model not found: {exc}")
        except Exception as exc:
            raise HTTPException(500, f"Synonym test failed: {exc}")

    # Predefined cluster mode
    clusters_text = ",".join(request.clusters) if request.clusters else "all"
    try:
        result = a.analyze(
            text=clusters_text,
            model_id=request.model_id,
            top_k=request.top_k,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model not found: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Synonym test failed: {exc}")


@app.get("/synonym-clusters")
def get_synonym_clusters():
    """Return available synonym cluster names."""
    try:
        a = get_analyzer("synonym")
        # Perform a dummy analyze to get cluster names, or just return them directly
        from synonym_analyzer import SYNONYM_CLUSTERS
        return {
            "clusters": list(SYNONYM_CLUSTERS.keys()),
            "details": {
                name: list(words.keys())
                for name, words in SYNONYM_CLUSTERS.items()
            },
        }
    except Exception as exc:
        raise HTTPException(500, f"Failed to list clusters: {exc}")


@app.post("/caps-test")
def caps_test(request: CapsTestRequest):
    """Run the capitalisation invariance test on selected words."""
    try:
        a = get_analyzer("caps")
    except KeyError:
        raise HTTPException(404, "Caps analyzer not registered")

    words_text = ",".join(request.words) if request.words else "all"
    try:
        result = a.analyze(
            text=words_text,
            model_id=request.model_id,
            top_k=request.top_k,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model not found: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Caps test failed: {exc}")


@app.get("/caps-words")
def get_caps_words():
    """Return available test words for the caps invariance test."""
    try:
        from caps_analyzer import WORD_TEMPLATES
        return {"words": list(WORD_TEMPLATES.keys())}
    except Exception as exc:
        raise HTTPException(500, f"Failed to list words: {exc}")


@app.post("/label-feature")
def label_feature(request: LabelFeatureRequest):
    """
    On-demand LLM-based labeling for a single SAE feature.

    Requires a valid LLM API key (GROQ_API_KEY or OPENAI_API_KEY in env).
    Uses ``analysis.py``'s ``FeatureLabeler`` under the hood.
    """
    try:
        a = get_analyzer(request.analyzer)
    except KeyError:
        raise HTTPException(404, f"Analyzer '{request.analyzer}' not found")

    if not hasattr(a, "label_feature"):
        raise HTTPException(400, f"Analyzer '{request.analyzer}' does not support feature labeling")

    try:
        result = a.label_feature(
            model_id=request.model_id,
            feature_idx=request.feature_idx,
            corpus_texts=request.corpus_texts,
            corpus_path=request.corpus_path,
            labeling_config=request.labeling_config,
            groq_api_key=request.groq_api_key,
        )

        # Persist successful UI-triggered interpretation output for audit/history.
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "ui",
            "analyzer": request.analyzer,
            "model_id": request.model_id,
            "feature_idx": request.feature_idx,
            "label": result.get("label"),
            "explanation": result.get("explanation"),
            "confidence": result.get("confidence"),
            "request_id": result.get("request_id"),
            "llm_activation_mode": result.get("llm_activation_mode"),
            "top_tokens": result.get("top_tokens") or [],
            "prompt_examples_count": len(result.get("llm_prompt_examples") or []),
            "corpus_texts_count": len(request.corpus_texts or []),
            "labeling_config": request.labeling_config or {},
            "error": result.get("error"),
        }
        try:
            _append_interpreted_feature_log(request.model_id, log_entry)
        except Exception as log_exc:
            print(f"[label_feature] Failed to append interpreted feature log: {log_exc}")

        return result
    except Exception as exc:
        raise HTTPException(500, f"Labeling failed: {exc}")


@app.post("/bulk-label-features")
def bulk_label_features(request: BulkLabelFeaturesRequest):
    """Bulk label features using either an explicit feature-id list or a contiguous range."""
    try:
        a = get_analyzer(request.analyzer)
    except KeyError:
        raise HTTPException(404, f"Analyzer '{request.analyzer}' not found")

    if not hasattr(a, "bulk_label_features"):
        raise HTTPException(400, f"Analyzer '{request.analyzer}' does not support bulk feature labeling")

    if request.feature_ids:
        parsed_feature_ids = [int(x) for x in request.feature_ids if int(x) >= 0]
        if not parsed_feature_ids:
            raise HTTPException(400, "feature_ids must contain at least one non-negative integer.")
        feature_start = None
        feature_end = None
    else:
        if request.feature_start is None or request.feature_end is None:
            raise HTTPException(400, "Provide either feature_ids or both feature_start and feature_end.")
        feature_start = int(request.feature_start)
        feature_end = int(request.feature_end)
        parsed_feature_ids = None

    try:
        result = a.bulk_label_features(
            model_id=request.model_id,
            feature_start=feature_start,
            feature_end=feature_end,
            feature_ids=parsed_feature_ids,
            num_sentences=request.num_sentences,
            llm_top_k=request.llm_top_k,
            min_activation=request.min_activation,
            corpus_path=request.corpus_path,
            labeling_config=request.labeling_config,
            groq_api_key=request.groq_api_key,
        )

        for item in result.get("labeled", []):
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "ui-bulk",
                "analyzer": request.analyzer,
                "model_id": request.model_id,
                "feature_idx": item.get("feature_id"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "request_id": item.get("request_id") or result.get("request_id"),
                "labeling_config": {
                    "num_sentences": request.num_sentences,
                    "top_k": request.llm_top_k,
                    "min_activation": request.min_activation,
                },
            }
            try:
                _append_interpreted_feature_log(request.model_id, entry)
            except Exception as log_exc:
                print(f"[bulk_label_features] Failed to append interpreted feature log: {log_exc}")

        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Bulk labeling failed: {exc}")


@app.post("/feature-activations")
def feature_activations(request: FeatureActivationsRequest):
    """
    Fetch sentence/token examples where a selected SAE feature activates.
    """
    try:
        a = get_analyzer("sae")
    except KeyError:
        raise HTTPException(404, "SAE analyzer not registered")

    if not hasattr(a, "find_feature_activations"):
        raise HTTPException(400, "SAE analyzer does not support feature activation search")

    try:
        result = a.find_feature_activations(
            model_id=request.model_id,
            feature_id=request.feature_id,
            dataset_name=request.dataset_name,
            dataset_config=request.dataset_config,
            split=request.split,
            corpus_path=request.corpus_path,
            max_sentences=request.max_sentences,
            target_activating_examples=request.target_activating_examples,
            page=request.page,
            page_size=request.page_size,
            min_activation=request.min_activation,
            text_field=request.text_field,
            max_length=request.max_length,
            seed=request.seed,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model not found: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Feature activation lookup failed: {exc}")


@app.post('/sentence-feature-trace', response_model=SentenceFeatureTraceResponse)
def sentence_feature_trace(req: SentenceFeatureTraceRequest):
    """Trace a single feature across all tokens in an input sentence."""
    try:
        a = get_analyzer("sae")
    except KeyError:
        raise HTTPException(404, "SAE analyzer not registered")

    if not hasattr(a, "trace_feature_in_sentence"):
        raise HTTPException(400, "SAE analyzer does not support sentence feature tracing")

    try:
        result = a.trace_feature_in_sentence(
            text=req.text,
            model_id=req.model_id,
            feature_id=req.feature_id,
            min_activation=req.min_activation,
            max_length=req.max_length,
        )
        return SentenceFeatureTraceResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.post('/feature-info', response_model=FeatureInfoResponse)
def feature_info(req: FeatureInfoRequest):
    """Return label/description metadata for one feature ID."""
    try:
        a = get_analyzer(req.analyzer)
    except KeyError:
        raise HTTPException(404, f"Analyzer '{req.analyzer}' not found")

    if not hasattr(a, "get_feature_info"):
        raise HTTPException(400, f"Analyzer '{req.analyzer}' does not support feature metadata lookup")

    try:
        result = a.get_feature_info(model_id=req.model_id, feature_id=req.feature_id)
        return FeatureInfoResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
