# Dataset Config Support - Fix Summary

## What Was Fixed

Added support for datasets that require a configuration name (like `wikitext`, `c4`, etc.) in all data collection scripts.

## Files Modified

### 1. `/home/nn/buet-classes/330-ML-lab/SAE-OOM-fixed/src/data_collection.py`
- ✅ Added `dataset_config` parameter to `collect_from_dataset()` method
- ✅ Added `dataset_config` parameter to `collect_from_dataset_with_texts()` method
- ✅ Both methods now conditionally pass config to `load_dataset()`

### 2. `/home/nn/buet-classes/330-ML-lab/SAE-OOM-fixed/collect_activation.py`
- ✅ Added `--dataset-config` CLI argument
- ✅ Pass `dataset_config` to both collector methods
- ✅ Display config in output when specified

### 3. `/home/nn/buet-classes/330-ML-lab/SAE-OOM-fixed/run_sae.py`
- ✅ Added `--dataset-config` CLI argument
- ✅ Pass `dataset_config` to `collect_from_dataset()`

### 4. `/home/nn/buet-classes/330-ML-lab/SAE-OOM-fixed/run_FC_sae.py`
- ✅ Added `--dataset-config` CLI argument
- ✅ Pass `dataset_config` to `collect_from_dataset()`

---

## Usage Examples

### Using Wikitext Dataset

```bash
# Standalone collection script
python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --output wikitext_activations.pt

# With standard SAE training
python run_sae.py \
    --layer 8 \
    --samples 50000 \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --epochs 30

# With Feature Choice SAE training
python run_FC_sae.py \
    --layer 8 \
    --samples 50000 \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --m 32 \
    --epochs 30
```

### Using C4 Dataset

```bash
python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --dataset c4 \
    --dataset-config en \
    --output c4_activations.pt
```

### Using Datasets Without Config (No change needed)

```bash
# OpenWebText doesn't need config - works as before
python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --dataset openwebtext
```

---

## Common Datasets That Need Config

| Dataset | Config Example | Description |
|---------|---------------|-------------|
| `wikitext` | `wikitext-103-v1` or `wikitext-2-v1` | Wikipedia text |
| `c4` | `en` or `es` | Web crawl text |
| `bookcorpus` | (no config needed) | Book text |
| `openwebtext` | (no config needed) | Reddit submissions |
| `pile` | (subset names) | Large diverse dataset |

---

## Implementation Details

### Before (Would Fail)
```python
# This would fail for wikitext:
dataset = load_dataset("wikitext", split="train", streaming=True)
# Error: wikitext requires a config name
```

### After (Works Correctly)
```python
# Now handles both cases:
if dataset_config:
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
else:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
```

---

## Testing

To verify the fix works:

```bash
# Test with wikitext (requires config)
python collect_activation.py \
    --layer 8 \
    --samples 1000 \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --output test_activations.pt

# Should output:
# Loading dataset: wikitext (config: wikitext-103-v1)
# ✓ Successfully collected activations
```

---

## Kaggle Usage

When using on Kaggle, add the `--dataset-config` parameter:

```python
# In Kaggle notebook
!python collect_activation.py \
    --layer 8 \
    --samples 50000 \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --batch-size 32

# Or in training scripts
!python run_FC_sae.py \
    --skip-collection \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --epochs 30
```

---

## Backward Compatibility

✅ Fully backward compatible - all existing scripts continue to work:

```bash
# Old usage (no config) - still works
python collect_activation.py --layer 8 --samples 50000 --dataset openwebtext

# New usage (with config) - now works
python collect_activation.py --layer 8 --samples 50000 --dataset wikitext --dataset-config wikitext-103-v1
```

The `dataset_config` parameter defaults to `None`, so datasets that don't need a config work exactly as before.

---

## Error Handling

If you forget the config for a dataset that needs it, HuggingFace will provide a helpful error:

```
ValueError: Config name is missing. Please select one of: ['wikitext-2-v1', 'wikitext-103-v1']
```

Then simply add `--dataset-config wikitext-103-v1` to your command.
