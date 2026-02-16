"""Visual explanation of heatmap generation and pixel-level AUROC calculation.

This script demonstrates:
1. How 256 patch scores are converted to pixel-level heatmaps
2. How pixel-level AUROC is computed from heatmaps vs ground truth masks
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.heatmap import patches_to_heatmap


def explain_heatmap_generation():
    """Step-by-step explanation of patch-to-pixel heatmap conversion."""
    print("=" * 80)
    print("HEATMAP GENERATION: From Patch Scores to Pixel-Level Visualization")
    print("=" * 80)
    
    # Simulate 256 patch scores (one per patch)
    np.random.seed(42)
    # Create some high scores in a region (simulating an anomaly)
    patch_scores = np.random.rand(256) * 0.3  # Low scores everywhere
    # Add a high-score region (anomaly)
    anomaly_patches = [100, 101, 102, 116, 117, 118, 132, 133, 134]
    for idx in anomaly_patches:
        patch_scores[idx] = 0.8 + np.random.rand() * 0.2  # High scores
    
    print(f"\n1. INPUT: {len(patch_scores)} patch scores (one per ViT patch)")
    print(f"   Shape: ({len(patch_scores)},)")
    print(f"   Score range: [{patch_scores.min():.3f}, {patch_scores.max():.3f}]")
    print(f"   Each patch represents a {14}x{14} pixel region in the original image")
    
    # Step 1: Reshape to grid
    grid_size = int(np.sqrt(len(patch_scores)))  # 16
    patch_grid = patch_scores.reshape(grid_size, grid_size)
    print(f"\n2. RESHAPE: Convert to {grid_size}x{grid_size} spatial grid")
    print(f"   Shape: ({grid_size}, {grid_size})")
    print(f"   This represents the spatial arrangement of patches")
    
    # Step 2: Upsample to image resolution
    image_size = 224
    heatmap = cv2.resize(patch_grid, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    print(f"\n3. UPSAMPLE: Bilinear interpolation to {image_size}x{image_size}")
    print(f"   Shape: ({image_size}, {image_size})")
    print(f"   Method: cv2.INTER_LINEAR (bilinear interpolation)")
    print(f"   Each pixel now has a score interpolated from nearby patches")
    print(f"   Pixels near high-score patches get higher scores")
    print(f"   Pixels near low-score patches get lower scores")
    
    # Step 3: Normalize using percentiles
    p5, p95 = np.percentile(heatmap, [5, 95])
    heatmap_clipped = np.clip(heatmap, p5, p95)
    if p95 - p5 > 1e-8:
        heatmap_normalized = (heatmap_clipped - p5) / (p95 - p5)
    else:
        heatmap_normalized = np.zeros_like(heatmap_clipped)
    
    print(f"\n4. NORMALIZE: Percentile-based normalization (5th-95th percentile)")
    print(f"   Percentiles: [{p5:.3f}, {p95:.3f}]")
    print(f"   Clips extreme outliers to prevent saturation")
    print(f"   Final range: [0.0, 1.0]")
    print(f"   Shape: ({image_size}, {image_size})")
    
    # Step 4: Apply colormap
    print(f"\n5. COLORMAP: Apply 'hot' colormap for visualization")
    print(f"   Low scores (0.0) → Black (no anomaly)")
    print(f"   Medium scores (0.5) → Red (possible anomaly)")
    print(f"   High scores (1.0) → Yellow/White (strong anomaly)")
    print(f"   This creates the visual heatmap you see")
    
    return patch_scores, patch_grid, heatmap_normalized


def explain_pixel_auroc():
    """Explain how pixel-level AUROC is computed."""
    print("\n" + "=" * 80)
    print("PIXEL-LEVEL AUROC: How We Evaluate Localization")
    print("=" * 80)
    
    # Simulate ground truth masks and heatmaps for multiple images
    n_images = 3
    image_size = 224
    
    print(f"\n1. INPUT: {n_images} images with ground truth masks and predicted heatmaps")
    print(f"   Each mask: ({image_size}, {image_size}) binary (0=normal, 1=anomaly)")
    print(f"   Each heatmap: ({image_size}, {image_size}) float scores [0, 1]")
    
    # Create example masks and heatmaps
    masks = []
    heatmaps = []
    
    for i in range(n_images):
        # Ground truth mask (binary)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        # Add some anomaly regions
        if i > 0:  # First image is normal
            y1, y2 = 50 + i*20, 100 + i*20
            x1, x2 = 50 + i*20, 100 + i*20
            mask[y1:y2, x1:x2] = 1
        
        # Predicted heatmap (scores)
        heatmap = np.random.rand(image_size, image_size) * 0.3
        if i > 0:
            # Make heatmap high in anomaly region (good prediction)
            heatmap[y1:y2, x1:x2] = 0.7 + np.random.rand(y2-y1, x2-x1) * 0.3
        
        masks.append(mask)
        heatmaps.append(heatmap)
    
    masks_arr = np.stack(masks)  # (N, H, W)
    heatmaps_arr = np.stack(heatmaps)  # (N, H, W)
    
    print(f"\n2. STACK: Combine all images")
    print(f"   Masks shape: ({n_images}, {image_size}, {image_size})")
    print(f"   Heatmaps shape: ({n_images}, {image_size}, {image_size})")
    
    # Flatten
    masks_flat = masks_arr.ravel()  # (N*H*W,)
    heatmaps_flat = heatmaps_arr.ravel()  # (N*H*W,)
    
    print(f"\n3. FLATTEN: Convert to 1D arrays")
    print(f"   Masks flattened: {masks_flat.shape} = {n_images * image_size * image_size} pixels")
    print(f"   Heatmaps flattened: {heatmaps_flat.shape}")
    print(f"   Each pixel is now treated as an independent prediction")
    
    # Compute AUROC
    from sklearn.metrics import roc_auc_score
    pixel_auroc = roc_auc_score(masks_flat, heatmaps_flat)
    
    print(f"\n4. COMPUTE AUROC: Compare flattened masks vs heatmaps")
    print(f"   Method: sklearn.metrics.roc_auc_score")
    print(f"   Input: {len(masks_flat)} pixel predictions")
    print(f"   Each pixel: (ground_truth_label, predicted_score)")
    print(f"   Result: Pixel-level AUROC = {pixel_auroc:.4f}")
    
    print(f"\n5. INTERPRETATION:")
    print(f"   AUROC = 1.0: Perfect pixel-level localization")
    print(f"   AUROC = 0.5: Random guessing")
    print(f"   AUROC > 0.7: Good localization")
    print(f"   AUROC > 0.9: Excellent localization")
    
    return masks_arr, heatmaps_arr, pixel_auroc


def visualize_process():
    """Create a visual diagram of the process."""
    print("\n" + "=" * 80)
    print("VISUALIZATION: Creating step-by-step diagram")
    print("=" * 80)
    
    # Generate example data
    patch_scores, patch_grid, heatmap = explain_heatmap_generation()
    masks_arr, heatmaps_arr, auroc = explain_pixel_auroc()
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Row 1: Patch scores → Heatmap
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(patch_grid, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f'1. Patch Grid\n({patch_grid.shape[0]}×{patch_grid.shape[1]})')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f'2. Upsampled Heatmap\n({heatmap.shape[0]}×{heatmap.shape[1]})')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(3, 3, 3)
    # Show colormap legend
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax3.imshow(gradient, cmap='hot', aspect='auto')
    ax3.set_title('3. Colormap\n(Black→Red→Yellow)')
    ax3.set_xlabel('Low Score → High Score')
    ax3.set_yticks([])
    
    # Row 2: Ground truth vs prediction
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(masks_arr[1], cmap='gray')
    ax4.set_title('Ground Truth Mask\n(White = Anomaly)')
    ax4.axis('off')
    
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(heatmaps_arr[1], cmap='hot', vmin=0, vmax=1)
    ax5.set_title('Predicted Heatmap\n(Red/Yellow = High Score)')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = plt.subplot(3, 3, 6)
    # Show pixel-level comparison
    mask_flat = masks_arr.ravel()
    heatmap_flat = heatmaps_arr.ravel()
    # Sample for visualization
    sample_idx = np.random.choice(len(mask_flat), size=1000, replace=False)
    ax6.scatter(heatmap_flat[sample_idx], mask_flat[sample_idx], 
                alpha=0.1, s=1)
    ax6.set_xlabel('Predicted Score')
    ax6.set_ylabel('Ground Truth (0/1)')
    ax6.set_title(f'Pixel-Level Comparison\n(AUROC = {auroc:.3f})')
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Process summary
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    summary_text = """
    PATCH → PIXEL PROCESS:
    
    1. 256 patch scores (one per patch)
    2. Reshape to 16×16 grid
    3. Upsample to 224×224 (bilinear)
    4. Normalize (percentile-based)
    5. Apply colormap (hot)
    """
    ax7.text(0.1, 0.5, summary_text, fontsize=10, 
             verticalalignment='center', family='monospace')
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    auroc_text = f"""
    PIXEL AUROC CALCULATION:
    
    1. Stack all masks: (N, H, W)
    2. Stack all heatmaps: (N, H, W)
    3. Flatten both: (N×H×W,)
    4. Compute AUROC on flattened arrays
    
    Result: {auroc:.4f}
    """
    ax8.text(0.1, 0.5, auroc_text, fontsize=10,
             verticalalignment='center', family='monospace')
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    interpretation_text = """
    INTERPRETATION:
    
    • Each pixel gets a score from nearby patches
    • High scores → Red/Yellow (anomaly)
    • Low scores → Black (normal)
    • Pixel AUROC measures how well scores
      match ground truth at pixel level
    """
    ax9.text(0.1, 0.5, interpretation_text, fontsize=10,
             verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig('heatmap_auroc_explanation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: heatmap_auroc_explanation.png")
    
    return fig


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HEATMAP & PIXEL AUROC EXPLANATION")
    print("=" * 80)
    
    fig = visualize_process()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
    HEATMAP GENERATION:
    ───────────────────
    1. Model outputs 256 patch scores (one per ViT patch)
    2. Scores are reshaped into a 16×16 spatial grid
    3. Grid is upsampled to 224×224 using bilinear interpolation
       → Each pixel gets a score interpolated from nearby patches
    4. Scores are normalized using percentiles (5th-95th) to handle outliers
    5. 'hot' colormap is applied: black (low) → red (medium) → yellow (high)
    
    PIXEL-LEVEL AUROC:
    ──────────────────
    1. For each test image, we have:
       - Ground truth mask: (H, W) binary (0=normal, 1=anomaly)
       - Predicted heatmap: (H, W) float scores [0, 1]
    2. Stack all masks: (N, H, W) → Flatten: (N×H×W,)
    3. Stack all heatmaps: (N, H, W) → Flatten: (N×H×W,)
    4. Compute AUROC between flattened arrays
       → Each pixel is treated as an independent prediction
    5. Result: Pixel-level AUROC (0.0 = worst, 1.0 = perfect)
    
    KEY INSIGHT:
    ────────────
    We don't have "patch AUROC" separately - we convert patch scores to
    pixel-level heatmaps first, then compute pixel AUROC. This allows us to
    evaluate localization at the pixel level, which is more fine-grained than
    patch-level evaluation.
    """)
    
    plt.show()

