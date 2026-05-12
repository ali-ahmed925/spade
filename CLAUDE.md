# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPADE is an unsupervised visual anomaly detection framework. It trains exclusively on normal images and detects defects at inference time using Mahalanobis distance scoring on patch-level features from a frozen BLIP-2 vision encoder. The MVTec Anomaly Detection dataset is the primary benchmark.

## Commands

```bash
# Setup
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"   # Required for BLIP-2 download
python test_setup.py                        # Verify installation, GPU, config, dataset

# Training
python train.py                             # Uses config/model.yaml, config/train.yaml, config/data.yaml

# Evaluation
python eval.py --checkpoint checkpoints/spade_best.pt

# Inference on a single image
python infer.py --image path/to/image.png --checkpoint checkpoints/spade_best.pt --output_dir ./output

# Tests
pytest tests/
```

## Architecture

**Data flow:**
```
Normal images (MVTec)
  → Synthetic anomaly generation (CutPaste / Crack) [data/synthetic.py]
  → ViT-G encoder (frozen BLIP-2) [models/vit.py]
  → Q-Former (learnable, 32 query tokens) [models/qformer.py]
  → Hierarchical Patch Annealing (HPA) — progressive patch pruning over T steps [models/hierarchical_patch_refinement.py]
  → Mahalanobis scoring + Fourier frequency features [models/mahalanobis_scoring.py, models/frequency_features.py]
  → Patch-level anomaly heatmap; optional LLM explanation [models/llm.py]
```

**Trainable vs frozen:** Only Q-Former and the LLM projection layer are trained. ViT-G and the LLM are frozen throughout.

**Normal statistics:** `models/normal_statistics.py` tracks running mean/covariance over normal-image patch embeddings; this is what the Mahalanobis scorer uses at inference.

**Config-driven:** All hyperparameters live in `config/` YAML files (`model.yaml`, `train.yaml`, `data.yaml`). Nothing is hard-coded in the training scripts.

## Critical Evaluation Note

Per-image normalization of anomaly scores **must not** be used when computing pixel-level AUROC. It inflates metrics by removing global score calibration. See `EVALUATION_FIX.md` for full context. Normalization is only appropriate for visualization (`utils/heatmap.py`).
