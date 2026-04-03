"""
FastAPI backend for the SAE Interpretability UI.

Provides:
    GET  /models           – list available checkpoints
    GET  /analyzers        – list registered analysis methods
    POST /analyze          – run analysis on input text
    POST /label-feature    – on-demand LLM-based feature labeling (via analysis.py)
"""

import sys
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
    text: str
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
    labeling_config: Optional[Dict[str, Any]] = None
    groq_api_key: Optional[str] = None  # falls back to GROQ_API_KEY env var


class FeatureActivationsRequest(BaseModel):
    model_id: str
    feature_id: int
    dataset_name: str = "MLCommons/peoples_speech"
    dataset_config: str = "validation"
    split: str = "validation"
    max_sentences: Optional[int] = None
    target_activating_examples: Optional[int] = None
    page: int = 1
    page_size: int = 25
    min_activation: float = 0.0
    text_field: Optional[str] = None
    max_length: int = 128
    seed: int = 0


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

    try:
        result = a.analyze(
            text=request.text,
            model_id=request.model_id,
            top_k=request.top_k,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, f"Model not found: {exc}")
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
            labeling_config=request.labeling_config,
            groq_api_key=request.groq_api_key,
        )
        return result
    except Exception as exc:
        raise HTTPException(500, f"Labeling failed: {exc}")


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
