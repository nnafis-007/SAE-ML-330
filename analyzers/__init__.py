"""
Modular analyzer framework.

To add a new analysis method:
1. Subclass ``BaseAnalyzer``
2. Implement ``name``, ``list_models``, and ``analyze``
3. Call ``register(instance)`` at module level

The FastAPI routes in main.py use the registry to dispatch requests.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseAnalyzer(ABC):
    """Interface every analyzer must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short name (used as query param / path segment)."""
        ...

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """Return available models / checkpoints for this analyzer."""
        ...

    @abstractmethod
    def analyze(self, text: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """
        Run analysis on *text* using the model identified by *model_id*.

        Must return a dict with at least::

            {
                "model": <model_id>,
                "tokens": [
                    {"text": "...", "features": [{"id": ..., "activation": ..., "description": ...}, ...]},
                    ...
                ]
            }
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: Dict[str, BaseAnalyzer] = {}


def register(analyzer: BaseAnalyzer) -> None:
    """Register an analyzer instance so the API can discover it."""
    _registry[analyzer.name] = analyzer


def get_analyzer(name: str) -> BaseAnalyzer:
    """Look up a registered analyzer by name. Raises KeyError if missing."""
    return _registry[name]


def list_analyzers() -> List[str]:
    """Return names of all registered analyzers."""
    return list(_registry.keys())


def get_all_models() -> List[Dict[str, Any]]:
    """Aggregate models from every registered analyzer."""
    models: List[Dict[str, Any]] = []
    for name, analyzer in _registry.items():
        for model in analyzer.list_models():
            model["analyzer"] = name
            models.append(model)
    return models
