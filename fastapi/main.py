"""
FastAPI backend for the SAE Interpretability UI.

Provides:
    GET  /models           – list available checkpoints
    GET  /analyzers        – list registered analysis methods
    POST /analyze          – run analysis on input text
    POST /label-feature    – on-demand LLM-based feature labeling (via analysis.py)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

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
from analyzers import sae_analyzer as _sae_reg  # noqa: F401, E402
from analyzers import synonym_analyzer as _syn_reg  # noqa: F401, E402
from analyzers import caps_analyzer as _caps_reg  # noqa: F401, E402
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
    top_k: int = 30


class CapsTestRequest(BaseModel):
    model_id: str
    words: Optional[List[str]] = None  # None = all words
    top_k: int = 30


class LabelFeatureRequest(BaseModel):
    model_id: str
    feature_idx: int
    analyzer: str = "sae"
    corpus_texts: Optional[List[str]] = None
    groq_api_key: Optional[str] = None  # falls back to GROQ_API_KEY env var


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
        from analyzers.synonym_analyzer import SYNONYM_CLUSTERS
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
        from analyzers.caps_analyzer import WORD_TEMPLATES
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
            groq_api_key=request.groq_api_key,
        )
        return result
    except Exception as exc:
        raise HTTPException(500, f"Labeling failed: {exc}")
