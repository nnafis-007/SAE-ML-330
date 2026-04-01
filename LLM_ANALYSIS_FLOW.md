# `llm_analysis.py` Flow Explained

This document explains how `analyzers/llm_analysis.py` works, step by step, with key code snippets.

## 1) What the module does

At a high level, `llm_analysis.py`:
1. Collects GPT-2 activations + token mappings.
2. Encodes activations with the trained SAE.
3. Finds top activating contexts per feature.
4. Builds a structured LLM prompt.
5. Calls the selected backend (`openai`, `groq`, or `ollama`).
6. Parses and saves labels.

Core class:

```python
class FeatureLabeler:
    ...
```

## 2) Runtime configuration (`LabelingConfig`)

`LabelingConfig` controls LLM behavior and the new activation handling logic:

```python
@dataclass
class LabelingConfig:
    backend: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    top_k: int = 20
    context_window: int = 10
    batch_size: int = 512
    request_delay: float = 0.5
    prompt_log_path: Optional[str] = "llm_prompts.log"

    normalize_mode: str = "standardize"  # standardize | center | none
    std_floor: float = 1e-3
    skip_first_token: bool = True
    global_top_features_k: int = 10
```

Meaning of the new fields:
- `normalize_mode` and `std_floor`: normalize activation vectors before SAE encoding.
- `skip_first_token=True`: excludes first token in each sequence (GPT-2 attention sink mitigation).
- `global_top_features_k=10`: include top globally active feature IDs/values in the prompt.

## 3) Prompt templates

Two templates are used:
- `_SYSTEM_PROMPT`: strict JSON output contract.
- `_USER_PROMPT_TEMPLATE`: provides top contexts and token stats.

Key part of user prompt now:

```python
_USER_PROMPT_TEMPLATE = """\
Feature index: {feature_idx}
...
Most frequent activating tokens (top 10): {top_tokens}

Top activated features across the analyzed corpus (top 10, feature_idx:mean_activation):
{global_top_features}

Based on these examples, what concept does this SAE feature detect?
"""
```

## 4) Backend creation and LLM call path

Backend factory:

```python
def _build_backend(cfg: LabelingConfig):
    if cfg.backend == "openai":
        return _OpenAIBackend(cfg)
    elif cfg.backend == "groq":
        return _GroqBackend(cfg)
    elif cfg.backend == "ollama":
        return _OllamaBackend(cfg)
    else:
        raise ValueError(...)
```

Inside `FeatureLabeler.__init__`, the backend is created once:

```python
self._backend = _build_backend(cfg)
```

## 5) Activation normalization before SAE encoding

Normalization stats are computed from the provided activation corpus:

```python
def _compute_normalization_stats(self, activations: torch.Tensor) -> Dict[str, Any]:
    mean = activations.mean(dim=0, keepdim=True)
    std = activations.std(dim=0, keepdim=True)
    return {
        "mean": mean,
        "std": std,
        "normalize_mode": self.cfg.normalize_mode,
        "std_floor": float(self.cfg.std_floor),
    }
```

Normalization is applied before `sae.encode(...)`:

```python
def _apply_activation_normalization(self, x, norm_stats):
    mode = str(norm_stats.get("normalize_mode", "standardize")).lower().strip()
    ...
    if mode in {"center", "center_only", "mean"}:
        return x - mean
    if mode in {"standardize", "zscore", "z-score"}:
        std = std.clamp_min(std_floor)
        return (x - mean) / (std + 1e-8)
```

Where this is used:
- `_collect_token_contexts(...)` before feature extraction.
- `_compute_mean_feature_activations(...)` for global ranking.

## 6) First-token filtering (attention sink skip)

Row filtering logic:

```python
def _build_valid_row_indices(self, token_pos_map: List[int]) -> torch.Tensor:
    if self.cfg.skip_first_token:
        idxs = [i for i, pos in enumerate(token_pos_map) if pos > 0]
        return torch.tensor(idxs, dtype=torch.long)
    return torch.arange(len(token_pos_map), dtype=torch.long)
```

This means when `skip_first_token=True`, only positions `1..N-1` are used per sequence.

## 7) Collecting top activating contexts for one feature

Main context extraction path:

```python
def _collect_token_contexts(...):
    feat_vals = torch.zeros(n, dtype=torch.float32)

    for start in range(0, n, cfg.batch_size):
        batch = activations[start:end].to(self.device)
        batch = self._apply_activation_normalization(batch, norm_stats)
        encoded = self.sae.encode(batch)
        feat_vals[start:end] = encoded[:, feature_idx].cpu()

    candidate_rows = valid_row_indices
    if candidate_rows is None:
        candidate_rows = self._build_valid_row_indices(token_pos_map)
    candidate_vals = feat_vals[candidate_rows]
    top_vals, rel_top_idxs = torch.topk(candidate_vals, k)
    top_idxs = candidate_rows[rel_top_idxs]
```

Then each selected row is converted into readable context:

```python
context_str = f"{prefix}>>>{token_s}<<<{suffix}"
contexts.append(TokenContext(token=token_s.strip(), context=context_str, activation_value=val))
```

## 8) Global top-10 feature list for prompt context

Global mean activation per feature is computed over valid rows:

```python
def _compute_mean_feature_activations(self, activations, valid_row_indices, norm_stats):
    ...
    enc = self.sae.encode(batch)
    mean_acts += enc.sum(dim=0).cpu()
    mean_acts /= max(n_valid, 1)
```

Formatted for prompt:

```python
def _format_global_top_features(self, mean_acts, top_n=10) -> str:
    vals, idxs = torch.topk(mean_acts, n)
    parts = [f"{int(i)}:{float(v):.3f}" for v, i in zip(vals.tolist(), idxs.tolist())]
    return ", ".join(parts)
```

## 9) Prompt creation, logging, and LLM call

`_call_llm(...)` builds prompt fields and writes logs when enabled:

```python
user_prompt = _USER_PROMPT_TEMPLATE.format(
    feature_idx=feature_idx,
    n_examples=len(contexts),
    examples_block=examples_block,
    top_tokens=top_tokens_str,
    global_top_features=global_top_features,
)

if self.cfg.prompt_log_path:
    with open(self.cfg.prompt_log_path, "a", encoding="utf-8") as f:
        f.write("SYSTEM PROMPT:\n")
        f.write(f"{_SYSTEM_PROMPT}\n")
        f.write("USER PROMPT:\n")
        f.write(f"{user_prompt}\n")

raw = self._backend.call(_SYSTEM_PROMPT, user_prompt)
```

Then response is parsed by `_parse_response(...)` with multiple fallbacks:
- direct JSON,
- fenced JSON,
- first `{...}` block,
- regex fallback.

## 10) Single-feature and multi-feature APIs

Single feature (recommended with pre-collected activations):

```python
def label_feature_from_activations(...):
    norm_stats = norm_stats or self._compute_normalization_stats(activations)
    if valid_row_indices is None:
        valid_row_indices = self._build_valid_row_indices(token_pos_map)
    if global_top_features is None:
        mean_acts = self._compute_mean_feature_activations(...)
        global_top_features = self._format_global_top_features(mean_acts, ...)

    contexts = self._collect_token_contexts(...)
    return self._call_llm(feature_idx, contexts, global_top_features)
```

Batch labeling:

```python
def label_features_from_activations(...):
    norm_stats = self._compute_normalization_stats(activations)
    valid_row_indices = self._build_valid_row_indices(token_pos_map)
    mean_acts = self._compute_mean_feature_activations(...)
    global_top_features = self._format_global_top_features(...)

    for feat_idx in feature_indices:
        result = self.label_feature_from_activations(
            feat_idx, activations, token_ids, token_doc_map, token_pos_map,
            norm_stats=norm_stats,
            valid_row_indices=valid_row_indices,
            global_top_features=global_top_features,
        )
```

So normalization/filtering/global-top-features are computed once and reused.

## 11) Saving and resuming

Results are serialized as JSON with `asdict(...)`:

```python
with open(path, "w", encoding="utf-8") as f:
    json.dump(serialisable, f, indent=2, ensure_ascii=False)
```

Resume logic in `label_features_from_activations(...)` skips already successful features when `resume=True`.

## 12) CLI flow in `__main__`

Execution order in script mode:
1. Parse CLI args (backend/model/dataset/cache/logging/normalization flags).
2. Load SAE checkpoint.
3. Load cached activations + token maps (or recollect + rebuild maps).
4. Build `LabelingConfig` and `FeatureLabeler`.
5. Compute alive/ranked features from normalized encoded activations.
6. Label features and save output JSON.

Config wiring in `__main__`:

```python
cfg = LabelingConfig(
    backend=args.backend,
    model=args.model,
    top_k=min(15, activations.shape[0]),
    request_delay=args.request_delay,
    prompt_log_path=args.prompt_log_path,
    normalize_mode=args.normalize_mode,
    std_floor=args.std_floor,
    skip_first_token=not args.include_first_token,
    global_top_features_k=10,
)
```

## 13) Minimal mental model

If you need a compact memory:
- `activations + token maps` -> `normalize` -> `SAE encode`.
- For each feature: top token rows -> context snippets -> prompt.
- Prompt also includes global top-10 features and top tokens.
- LLM returns JSON label/explanation/confidence.
- Results + raw responses + contexts are saved.
