# SAE-OOM-fixed: Analyzer, Backend, and UI Documentation

This document explains the three practical layers you interact with in this project:

1. `analyzers/` (research scripts and analysis utilities)
2. `fastapi/` (backend API that exposes analysis to the frontend)
3. `ui/` (Expo React Native Web interface)

It is written to help you understand each component's role, how they connect, and how to use them correctly.

---

## 1. Big Picture Architecture

The project has two usage modes:

- Offline research mode: run scripts in `analyzers/` to generate reports and plots.
- Interactive app mode: start the FastAPI server and use the web UI.

### Runtime flow (interactive app)

1. UI sends requests to FastAPI (`http://localhost:8000`).
2. FastAPI routes requests to a registered analyzer (`sae`, `synonym`, `caps`).
3. Analyzer loads SAE checkpoint + GPT-2 internals as needed.
4. Analyzer returns structured JSON.
5. UI renders tokens/features/metrics and allows deeper inspection.

### Core dependency direction

- `ui/` depends on `fastapi/`
- `fastapi/` depends on `analyzers/` and `src/`
- `analyzers/` depends on `src/` (SAE model and data collection code)

---

## 2. `analyzers/` Folder: What It Is and How to Use It

This folder is both:

- A plugin registry system (for backend analyzers)
- A collection of standalone experiment scripts and plotting tools

Think of it as your "analysis lab".

## 2.1 Registry and Plugin Contract

### `analyzers/__init__.py`

Defines the analyzer interface and global registry:

- `BaseAnalyzer`: abstract interface every analyzer must implement
	- `name`
	- `list_models()`
	- `analyze(text, model_id, **kwargs)`
- `register(analyzer)`: adds analyzer instance to registry
- `get_analyzer(name)`: lookup by analyzer name
- `list_analyzers()`: list all registered analyzers
- `get_all_models()`: aggregate models from all registered analyzers

How it is used:

- FastAPI imports analyzer modules.
- At module import time, each analyzer creates an instance and calls `register(...)`.
- API routes call `get_analyzer(...)` dynamically.

---

## 2.2 Standalone Analysis Scripts

These scripts are meant for experiments/report generation, not direct UI rendering.

### `analyzers/run_synonym_test.py`

Purpose:

- Tests semantic consistency: do synonyms activate overlapping SAE features?

How it works:

- Uses predefined synonym clusters (e.g., `happy`, `large`, `fast`, `angry`).
- Extracts activations at exact token positions for each word.
- Computes top-K features per word.
- Compares words using:
	- Jaccard overlap on top-K feature sets
	- Cosine similarity on mean feature vectors

Outputs:

- JSON report (default `synonym_test_report.json`)
- Rich terminal summary

CLI example:

```bash
python analyzers/run_synonym_test.py --top-k 30 --checkpoint checkpoints/best_model.pt
```

### `analyzers/run_caps_test.py`

Purpose:

- Tests case invariance: does `cat` behave similarly to `Cat` / `CAT` / alternating-case forms?

How it works:

- Generates capitalization variants for each word.
- Uses template sentences to isolate each variant in controlled contexts.
- Compares feature overlap/similarity across variants.

Outputs:

- JSON report (default `caps_test_report.json`)

CLI example:

```bash
python analyzers/run_caps_test.py --top-k 30 --words cat king happy
```

### `analyzers/run_bias_test.py`

Purpose:

- Tests representational bias using subject-role sentence templates.
- Includes both stereotype-targeted groups and neutral baseline groups.

Groups in script:

- `gender`
- `racial`
- `gender_neutral` (baseline)
- `racial_neutral` (baseline)

Metrics include:

- Role-position Jaccard
- Subject-position Jaccard
- Normalized L2 distance
- MRAD (mean relative activation difference)
- Subject-axis projection
- Bias delta

Output:

- JSON report (default `bias_test_report.json`)

CLI example:

```bash
python analyzers/run_bias_test.py --groups gender gender_neutral --top-k 30
```

### `analyzers/run_interpretation.py`

Purpose:

- End-to-end feature interpretation pipeline runner.

Pipeline:

1. Load SAE checkpoint.
2. Load GPT-2.
3. Collect activations from built-in sample or HuggingFace dataset.
4. Select features (manual or auto with activation-rate filtering).
5. Run interpretation and save full report.

Output:

- JSON report (default `interpretation_reports.json`)

CLI example:

```bash
python analyzers/run_interpretation.py --dataset openwebtext --num-texts 500
```

---

## 2.3 Interpretation and Labeling Modules

### `analyzers/pmi_feature_interpretation.py`

Purpose:

- Implements reusable `FeatureInterpreter` logic for feature meaning analysis.

Main methods/ideas:

- Token association analysis (PMI + Chi-square)
- Logit-lens projection from decoder direction to token logits
- POS-tag enrichment analysis
- Top-context extraction
- Report generation/printing/saving
- Utility: `build_flat_token_ids(...)`

Use this module when you need programmatic interpretation from Python code.

### `analyzers/llm_analysis.py`

Purpose:

- Uses LLMs to auto-label features from top-activating contexts.

Backends:

- `groq` (default in config)
- `openai`
- `ollama`

Important components:

- `LabelingConfig`
- `FeatureLabeler`
- `label_feature_from_activations(...)` (recommended API)
- `label_features_from_activations(...)`
- `build_token_maps(...)` helper

Expected output:

- Label JSON such as `feature_labels.json`

How backend uses this:

- FastAPI `/label-feature` route calls analyzer method that internally uses this module.

---

## 2.4 Plotting Scripts

### `analyzers/plot_synonym_results.py`

Reads `synonym_test_report.json` and generates multi-view plots:

- Cluster overview
- Pairwise heatmaps
- Feature breakdown
- Jaccard-vs-cosine scatter
- Universal shared features

### `analyzers/plot_caps_results.py`

Reads `caps_test_report.json` and generates plots:

- Word overview
- Pairwise heatmaps
- lower-vs-UPPER comparison
- Feature breakdown per variant
- Jaccard-vs-cosine scatter

CLI examples:

```bash
python analyzers/plot_synonym_results.py --report synonym_test_report.json --output-dir plots
python analyzers/plot_caps_results.py --report caps_test_report.json --output-dir plots
```

---

## 2.5 Practical guidance for `analyzers/`

- Use `run_*` scripts when you want reproducible experiment artifacts.
- Use `plot_*` scripts after reports are generated.
- Use `llm_analysis.py` only when API key/backend is configured.
- Keep checkpoint path consistent across scripts for comparable results.
- For heavy runs, prefer GPU (`--device auto` will select CUDA when available).

---

## 3. `fastapi/` Folder: Backend API Layer

This folder turns model analysis into API endpoints consumed by the UI.

Files:

- `fastapi/main.py`
- `fastapi/sae_analyzer.py`
- `fastapi/synonym_analyzer.py`
- `fastapi/caps_analyzer.py`

## 3.1 `fastapi/main.py`

Responsibilities:

- Creates FastAPI app.
- Enables CORS (`allow_origins=["*"]`) for local frontend access.
- Ensures project root is importable via `sys.path`.
- Imports analyzer modules so they auto-register.
- Exposes API routes and request schemas.

### Endpoints

#### `GET /models`

- Optional query: `analyzer`
- Returns model checkpoints.
- If analyzer provided, returns that analyzer's model list only.

#### `GET /analyzers`

- Returns registered analyzers, e.g., `sae`, `synonym`, `caps`.

#### `POST /analyze`

Body:

```json
{
	"text": "...",
	"model_id": "...",
	"analyzer": "sae",
	"top_k": 10
}
```

Returns token-level feature activations from selected analyzer.

#### `POST /synonym-test`

Body (preset mode):

```json
{
	"model_id": "...",
	"clusters": ["happy", "fast"],
	"top_k": 30
}
```

Body (custom words mode):

```json
{
	"model_id": "...",
	"custom_words": ["happy", "joyful", "cheerful"],
	"top_k": 30
}
```

#### `GET /synonym-clusters`

- Returns available cluster names and word details.

#### `POST /caps-test`

Body:

```json
{
	"model_id": "...",
	"words": ["cat", "happy"],
	"top_k": 30
}
```

#### `GET /caps-words`

- Returns selectable words for caps test.

#### `POST /label-feature`

Body:

```json
{
	"model_id": "...",
	"feature_idx": 42,
	"analyzer": "sae",
	"corpus_texts": null,
	"groq_api_key": null
}
```

Notes:

- Analyzer must support `label_feature` method.
- Typically requires `GROQ_API_KEY` or provided key.

---

## 3.2 `fastapi/sae_analyzer.py`

Purpose:

- Core analyzer for token-wise SAE feature activations used in the SAE UI tab.

Key behavior:

- Lazy-load and cache SAE checkpoints.
- Lazy-load and cache GPT-2 activation collectors per layer.
- Build token responses with top active features and optional label descriptions.
- Load existing `feature_labels.json` beside checkpoint if present.
- Supports on-demand LLM labeling via `label_feature(...)`.

Key output shape for UI:

```json
{
	"model": "...",
	"layer_index": 8,
	"d_hidden": 12288,
	"tokens": [
		{
			"text": " token",
			"features": [
				{"id": 123, "activation": 4.56, "description": "Feature 123"}
			]
		}
	]
}
```

---

## 3.3 `fastapi/synonym_analyzer.py`

Purpose:

- Backend adapter for synonym overlap tests.

Important details:

- Reuses helpers from `analyzers/run_synonym_test.py`.
- Supports two modes:
	- preset clusters
	- custom words (neutral generic sentence templates)
- Returns cluster-level metrics and pairwise detail.

---

## 3.4 `fastapi/caps_analyzer.py`

Purpose:

- Backend adapter for capitalization invariance tests.

Important details:

- Reuses helpers from `analyzers/run_caps_test.py`.
- Tests selected words or all predefined words.
- Returns per-word metrics, variant pairwise stats, and interpretation labels.

---

## 3.5 Backend startup and expected behavior

From project root:

```bash
source venv/bin/activate
uvicorn main:app --app-dir fastapi --host 127.0.0.1 --port 8000
```

The first heavy request may be slower because GPT-2/checkpoints are loaded lazily.

Health checks you can do quickly:

- `GET http://127.0.0.1:8000/analyzers`
- `GET http://127.0.0.1:8000/models?analyzer=sae`

---

## 4. `ui/` Folder: Frontend Layer

This is an Expo React Native Web app that provides three tabs over the API.

Files:

- `ui/index.js`
- `ui/App.js`
- `ui/FeatureDetails.js`
- `ui/package.json`
- `ui/app.json`

## 4.1 `ui/index.js`

- Entry point.
- Registers the root component (`App`).

## 4.2 `ui/App.js`

This is the main app logic.

### Key states

- Active tab (`sae`, `synonym`, `caps`)
- Selected model/checkpoint
- Token results and selected token
- Synonym test mode/settings/results
- Caps test settings/results
- Feature modal selection

### Key UI tabs

#### SAE Analysis tab

- Loads models from `/models?analyzer=sae`
- Sends input text to `/analyze`
- Renders tokenized text interactively
- On token click, shows top features and activation values
- Opens feature modal (`FeatureDetails`) for deeper inspection

#### Context-Similarity Test tab

- Fetches cluster metadata from `/synonym-clusters`
- Supports custom words and preset clusters
- Sends requests to `/synonym-test`
- Displays cluster cards with Jaccard/cosine/pairwise stats

#### Caps Test tab

- Fetches words from `/caps-words`
- Sends selected words to `/caps-test`
- Displays per-word invariance metrics and pairwise results

### Important config

- API base URL is hardcoded:

```js
const API_BASE = 'http://localhost:8000';
```

If backend is running elsewhere, update this value.

## 4.3 `ui/FeatureDetails.js`

Purpose:

- Feature detail modal view (opened from SAE tab).

Current behavior:

- Shows rich mock visualization sections (logits/activations cards).
- Supports live LLM label fetch from `/label-feature` for selected feature.

Note:

- Much of this component is currently mock/demo style; the LLM label section is wired to backend.

## 4.4 `ui/package.json`

Useful scripts:

- `npm run start`
- `npm run web`
- `npm run android`
- `npm run ios`

Main stack:

- Expo 50
- React 18
- React Native 0.73
- React Native Web
- `@react-native-picker/picker`

## 4.5 `ui/app.json`

- Expo app metadata/config (name, splash, icon, web settings, etc.)

---

## 5. End-to-End Usage Recipes

## 5.1 Run full interactive app

Terminal 1 (backend):

```bash
source venv/bin/activate
uvicorn main:app --app-dir fastapi --host 127.0.0.1 --port 8000
```

Terminal 2 (frontend):

```bash
cd ui
npx expo start --web
```

Then open the local web URL Expo provides.

## 5.2 Generate research report + plots (offline flow)

Example (synonym):

```bash
python analyzers/run_synonym_test.py --checkpoint checkpoints/best_model.pt --top-k 30
python analyzers/plot_synonym_results.py --report synonym_test_report.json --output-dir plots
```

Example (caps):

```bash
python analyzers/run_caps_test.py --checkpoint checkpoints/best_model.pt --top-k 30
python analyzers/plot_caps_results.py --report caps_test_report.json --output-dir plots
```

---

## 6. Common Pitfalls and Fixes

## Backend returns "model not found"

- Ensure selected `model_id` exists under `checkpoints/`.
- Verify `/models?analyzer=sae` returns it.

## UI cannot connect to backend

- Verify backend is running at `127.0.0.1:8000`.
- If using remote/devcontainer host, update `API_BASE` in `ui/App.js`.

## LLM labeling fails

- Set `GROQ_API_KEY` (or pass `groq_api_key` in request).
- Ensure outbound internet/API access.

## First request is very slow

- Expected: analyzer lazy-loads GPT-2 + checkpoint on first heavy request.

---

## 7. Extending the System Correctly

To add a new analyzer end-to-end:

1. Create new analyzer class (implement `BaseAnalyzer`) in `fastapi/` or reusable logic in `analyzers/`.
2. Register instance with `register(...)`.
3. Import module in `fastapi/main.py` so it auto-registers.
4. Add endpoint(s) if needed in `main.py`.
5. Add UI tab or controls in `ui/App.js`.
6. Verify with `/analyzers`, `/models`, and a test POST route call.

This ensures consistency with existing plugin-style design.

---

## 8. Quick Component Responsibility Summary

- `analyzers/`: research methods, metric logic, report generation, plotting, optional LLM labeling utilities.
- `fastapi/`: stable API contract layer + runtime caching + model dispatch.
- `ui/`: user interaction, API orchestration, and visualization of analysis results.

If you remember only one thing: the backend is the bridge between heavy SAE logic and the interactive frontend, while `analyzers/` is the reusable analysis engine.
