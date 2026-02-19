# Critical Fix: Pixel AUROC Evaluation

## Problem

The evaluation code was **artificially inflating pixel AUROC** by applying per-image normalization before computing the metric.

### What Was Wrong

```python
# OLD CODE (WRONG):
# Per-image percentile normalization
p5, p95 = np.percentile(patch_scores_np, [5, 95])
patch_scores_normalized = (patch_scores_clipped - p5) / (p95 - p5)

# Then compute pixel AUROC on normalized scores
hmap = patches_to_heatmap(patch_scores_normalized, ...)
pixel_auroc = compute_pixel_auroc(masks_arr, heatmaps_arr)
```

**Why this is wrong:**
- Each image's scores are independently stretched to [0, 1]
- A completely normal image will still have pixels near 1.0
- A weak anomaly gets boosted to maximum score
- Global calibration is removed
- Makes pixel AUROC artificially easier to achieve

## Solution

### Golden Rule

**Normalization is ONLY for visualization. NEVER for metrics.**

### Fixed Code

```python
# NEW CODE (CORRECT):
# Use RAW scores for pixel AUROC computation
hmap_raw = patches_to_heatmap(
    patch_scores_raw,
    normalize=False,  # NO normalization for metrics
)

# Only normalize for visualization
if save_dir is not None:
    patch_scores_normalized = normalize_per_image(patch_scores_raw)
    hmap_viz = patches_to_heatmap(
        patch_scores_normalized,
        normalize=True,  # Normalize for visualization only
    )
    save_heatmap(hmap_viz, ...)  # Save normalized version
```

## Changes Made

1. **`utils/heatmap.py`**:
   - Added `normalize` parameter to `patches_to_heatmap()`
   - When `normalize=False`, returns raw scores (no percentile clipping/normalization)
   - When `normalize=True`, applies percentile-based normalization (for visualization)

2. **`eval.py`**:
   - Use **raw scores** (`normalize=False`) for pixel AUROC computation
   - Create **normalized version** (`normalize=True`) only for saving visualizations
   - Clear comments explaining why normalization is NOT used for metrics

3. **`infer.py`**:
   - Updated to use consistent normalization approach
   - Added comment clarifying normalization is for visualization only

## Impact

- **Before**: Pixel AUROC was artificially inflated due to per-image normalization
- **After**: Pixel AUROC uses raw scores, providing honest evaluation

## Expected Behavior

After this fix:
- Pixel AUROC scores will likely be **lower** (more honest)
- This is **correct** - the previous scores were inflated
- Visualization heatmaps remain unchanged (still normalized for better visual contrast)

## Testing

Run evaluation to verify:
```bash
python eval.py --checkpoint checkpoints/spade_best.pt --save_heatmaps eval_output/heatmaps
```

The pixel AUROC will now reflect true localization performance without artificial inflation.



