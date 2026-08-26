# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPADE is an unsupervised visual anomaly detection framework. It trains exclusively on normal images and detects defects at inference time using Mahalanobis distance scoring on patch-level features from a frozen BLIP-2 vision encoder. The MVTec Anomaly Detection dataset is the primary benchmark.

**One model per MVTec category.** Everything (training, checkpoints, thresholds) is category-scoped: the category comes from `dataset.category` in `config/data.yaml`, and checkpoints land in `checkpoints/<category>/spade_best.pt`. Training all categories means re-running `train.py` once per category with that field changed.

## Commands

```bash
# Setup
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"   # Required for BLIP-2 download
python test_setup.py                        # Verify installation, GPU, config, dataset

# Training (category comes from config/data.yaml → dataset.category)
python train.py

# Evaluation — checkpoint must match the category in config/data.yaml
python eval.py --checkpoint checkpoints/bottle/spade_best.pt
python eval.py --checkpoint checkpoints/bottle/spade_best.pt --save_visualizations --log_wandb
python eval.py --checkpoint checkpoints/bottle/spade_best.pt --image path/to/image.png   # single image

# Inference on a single image (adds LLM explanation unless --no_llm)
python infer.py --image path/to/image.png --checkpoint checkpoints/bottle/spade_best.pt --output_dir ./output

# Demo web app (FastAPI + static frontend); reads root config.yaml
python app.py

# Tests
pytest tests/
pytest tests/test_mahalanobis_scoring.py::test_mahalanobis_forward   # single test
```

## Configuration layers

Two independent config layers — do not confuse them:

- `config/*.yaml` — the pipeline. Each entry point has its **own** `load_config()` that merges a different subset into one flat dict: `train.py` and `eval.py` merge `model + data + train`; `infer.py` merges `model + data + llm + train` (the last for `training.checkpoint_dir`). Top-level keys are flattened, so a key added to two files silently overwrites. Adding a config key that a script needs means checking that script's own tuple.
- `config.yaml` (repo root) — the demo app only. Holds `dataset_root`, `checkpoints_root`, per-category curated image lists, per-category verdict `thresholds`, and display-only `hyperparameters`. The `hyperparameters` block is a UI label set and can drift from `config/model.yaml` — `config/model.yaml` is the truth for anything that affects computation.

Nothing is hard-coded in the training scripts; all hyperparameters live in these files.

## Architecture

**Data flow:**
```
Normal images (MVTec, train/good) at 448x448
  → frozen ViT-G (BLIP-2), position embeddings interpolated 224→448   [models/vit.py]
       taps intermediate blocks 20 and 30 (config: vit.feature_layers)
  → 32x32 = 1024 patches per block, 1408-d
  → MultiLayerPatchFusion: per-block LayerNorm + Linear + concat   [models/feature_fusion.py]
  → Q-Former (trainable, 32 query tokens) over the final block      [models/qformer.py]
  → QueryPatchContextualizer: every patch cross-attends to the 32 queries
  → 512-d CONTEXTUALISED per-patch descriptors
  → Mahalanobis over descriptors                          [models/mahalanobis_scoring.py]
  + parallel Fourier frequency stream                     [models/frequency_features.py]
  → per-patch score → top-k mean → image score
  → 32x32 map → heatmap [utils/heatmap.py]; LLM explanation [models/llm.py]
```

**Why this descriptor space** (all measured on MVTec, see `docs/`):
- **448 not 224**: at 224 one patch covers 196 px = 0.39% of the image, and capsule
  `crack` (0.30%), `poke` (0.23%) and `faulty_imprint` (0.28%) are each *smaller than
  one patch* — averaged away before scoring. That is 94% of capsule's AUROC deficit.
- **mid-level blocks not just the final one**: the final ViT-G block is BLIP-2's most
  semantic representation and is invariant to sub-semantic damage. Defect elevation
  was 1.5–3x on weak classes against 8–75x on classes that work.
- **contextualised not independent**: role violations (`cable_swap` = ordinary blue
  insulation in the wrong position, `cut_lead` = ordinary board where a lead belongs)
  are individually normal patches. `cut_lead` defect patches scored *below* their own
  image's clean patches (within-image AUROC 0.436). Only conditioning on the rest of
  the image separates them.

**Trainable vs frozen:** trainable = fusion, contextualizer, Q-Former (projection is
frozen unless `projection.trainable`, since no loss supervises the text path).
Frozen = ViT-G, LLM. Mahalanobis μ/Σ are *fitted in closed form*, not learned.
Because fusion/contextualizer are trainable and sit before the scorer, the Mahalanobis
term now has a gradient path — it had none before (the frozen ViT runs under `no_grad`).

**Construction is centralized** in `models/builder.py` (`build_spade`, `load_spade`,
`describe`). Every entry point — train/eval/infer/app/demo/scripts — goes through it;
do not re-create `SPADE(...)` argument lists.

**Checkpoints:** `train.py` saves everything except `vision_encoder.` (restored from the
BLIP-2 download). Loading goes through `utils/checkpoint.py`, which drops
shape-mismatched tensors with a warning instead of raising — pre-redesign checkpoints
have 1408-d/256-patch statistics that cannot fit the 512-d/1024-patch model and must be
refitted.

**HPA has been removed.** It was measured to be a literal no-op: with it enabled and
disabled the eval produced bit-identical image/pixel/PRO numbers, because its only
surviving output fed an attention term worth ~1e-5 of the score. The additive
attention-importance and cross terms were removed with it; the Q-Former now enters
through the descriptor instead.

## Critical Evaluation Note

Per-image normalization of anomaly scores **must not** be used when computing pixel-level AUROC. It inflates metrics by removing global score calibration. See `EVALUATION_FIX.md` for full context. `patches_to_heatmap(..., normalize=False)` is the metric path; `normalize=True` is for visualization only (`utils/heatmap.py`).

## Demo app notes

`app.py` is a single-file FastAPI server: it lazy-loads one SPADE model per category and **evicts every other cached model before loading a new one** to keep VRAM free, so switching categories in the UI is slow by design. Images are returned to the frontend as base64 (224px JPEG thumbnails for the slot-machine picker) to avoid per-image round trips through a tunnel. Ground-truth masks are found by rewriting `test/` → `ground_truth/` in the image path and appending `_mask.png`.
