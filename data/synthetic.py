"""Synthetic anomaly generation for self-supervised training.

Generates realistic fake defects on normal images and produces patch-level masks.
Methods: Enhanced CutPaste and Crack generation.
"""

import math
import random

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# ──────────────────────────────────────────────
# Enhanced CutPaste with Realistic Blending
# ──────────────────────────────────────────────


def cutpaste(
    image: np.ndarray,
    area_ratio: tuple[float, float] = (0.02, 0.15),
    blend_alpha: float = 0.7,
    color_jitter: float = 0.1,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advanced CutPaste with:
    - Clean binary masks (white=anomaly, black=normal)
    - Smart region matching (cut from similar brightness/texture areas)
    - Seamless Poisson-like blending
    - Realistic defect characteristics
    """
    rng = rng or random
    np_rng = np_rng or np.random.default_rng()

    h, w = image.shape[:2]
    area = h * w

    # Choose random patch size (more variation)
    target_area = rng.uniform(*area_ratio) * area
    aspect_ratio = rng.uniform(0.5, 2.0)
    patch_h = int(math.sqrt(target_area * aspect_ratio))
    patch_w = int(math.sqrt(target_area / aspect_ratio))
    patch_h, patch_w = min(patch_h, h // 2), min(patch_w, w // 2)
    patch_h, patch_w = max(patch_h, 30), max(patch_w, 30)  # Minimum size for visibility

    # ═══════════════════════════════════════════════
    # SMART SOURCE-DESTINATION MATCHING
    # ═══════════════════════════════════════════════
    
    # First, select destination
    dx, dy = rng.randint(0, w - patch_w), rng.randint(0, h - patch_h)
    dest_region = image[dy:dy + patch_h, dx:dx + patch_w].astype(np.float32)
    dest_brightness = dest_region.mean()
    dest_std = dest_region.std()
    
    # Find source region with SIMILAR characteristics
    # Try multiple candidates and pick best match
    best_source = None
    best_score = float('inf')
    num_attempts = 10
    
    for _ in range(num_attempts):
        sx = rng.randint(0, w - patch_w)
        sy = rng.randint(0, h - patch_h)
        
        # Don't overlap with destination (at least 30% separation)
        if abs(sx - dx) < patch_w * 0.7 and abs(sy - dy) < patch_h * 0.7:
            continue
            
        candidate = image[sy:sy + patch_h, sx:sx + patch_w].astype(np.float32)
        cand_brightness = candidate.mean()
        cand_std = candidate.std()
        
        # Score based on brightness and texture similarity
        brightness_diff = abs(cand_brightness - dest_brightness)
        std_diff = abs(cand_std - dest_std)
        score = brightness_diff + std_diff * 0.5
        
        if score < best_score:
            best_score = score
            best_source = (sx, sy, candidate.copy())
    
    # Use best matching source or fallback to random
    if best_source is not None:
        sx, sy, patch = best_source
    else:
        sx, sy = rng.randint(0, w - patch_w), rng.randint(0, h - patch_h)
        patch = image[sy:sy + patch_h, sx:sx + patch_w].copy().astype(np.float32)

    # ═══════════════════════════════════════════════
    # CREATE CLEAN BINARY MASK (no Perlin noise)
    # ═══════════════════════════════════════════════
    
    mask_patch = np.zeros((patch_h, patch_w), dtype=np.uint8)
    
    # Choose shape type
    shape_type = rng.choice(['ellipse', 'rectangle', 'polygon', 'irregular'])
    
    if shape_type == 'ellipse':
        center = (patch_w // 2, patch_h // 2)
        axes = (int(patch_w * rng.uniform(0.35, 0.48)), int(patch_h * rng.uniform(0.35, 0.48)))
        cv2.ellipse(mask_patch, center, axes, rng.uniform(0, 180), 0, 360, 1, -1)
    
    elif shape_type == 'rectangle':
        margin = 0.1
        x1 = int(patch_w * rng.uniform(margin, 0.2))
        y1 = int(patch_h * rng.uniform(margin, 0.2))
        x2 = int(patch_w * rng.uniform(0.8, 1-margin))
        y2 = int(patch_h * rng.uniform(0.8, 1-margin))
        cv2.rectangle(mask_patch, (x1, y1), (x2, y2), 1, -1)
    
    elif shape_type == 'polygon':
        num_pts = rng.randint(5, 8)
        pts = []
        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts + rng.uniform(-0.4, 0.4)
            r = min(patch_w, patch_h) * rng.uniform(0.35, 0.48)
            x = int(patch_w/2 + r * math.cos(angle))
            y = int(patch_h/2 + r * math.sin(angle))
            pts.append([x, y])
        cv2.fillPoly(mask_patch, [np.array(pts, np.int32)], 1)
    
    else:  # irregular blob
        # Create organic shape with overlapping circles
        num_circles = rng.randint(3, 6)
        for _ in range(num_circles):
            cx = rng.randint(patch_w//4, 3*patch_w//4)
            cy = rng.randint(patch_h//4, 3*patch_h//4)
            radius = rng.randint(min(patch_h, patch_w)//6, min(patch_h, patch_w)//3)
            cv2.circle(mask_patch, (cx, cy), radius, 1, -1)
    
    # Clean binary mask - just smooth edges slightly
    mask_patch = cv2.GaussianBlur(mask_patch.astype(np.float32), (5, 5), 1)
    mask_patch = (mask_patch > 0.5).astype(np.uint8)  # Clean threshold

    # ═══════════════════════════════════════════════
    # APPLY TRANSFORMATIONS TO PATCH
    # ═══════════════════════════════════════════════
    
    # Random rotation and scale
    angle = rng.uniform(-30, 30)
    scale = rng.uniform(0.9, 1.1)
    center_rot = (patch_w // 2, patch_h // 2)
    M = cv2.getRotationMatrix2D(center_rot, angle, scale)
    patch = cv2.warpAffine(patch, M, (patch_w, patch_h), borderMode=cv2.BORDER_REFLECT)
    mask_patch = cv2.warpAffine(mask_patch, M, (patch_w, patch_h), borderMode=cv2.BORDER_CONSTANT)
    mask_patch = (mask_patch > 0.5).astype(np.uint8)

    # ═══════════════════════════════════════════════
    # ADD DEFECT CHARACTERISTICS
    # ═══════════════════════════════════════════════
    
    defect_type = rng.choice(['discolor', 'contamination', 'scratch', 'texture_change', 'brightness'])
    
    # Create defect mask (only where mask_patch is 1)
    defect_mask = mask_patch.astype(np.float32)
    
    if defect_type == 'discolor':
        # Color shift in one or more channels
        num_channels = rng.randint(1, 3)
        for _ in range(num_channels):
            channel = rng.randint(0, 2)
            shift = rng.uniform(-40, 40)
            patch[:, :, channel] += shift * defect_mask
    
    elif defect_type == 'contamination':
        # Dark spots/stains
        num_spots = rng.randint(3, 8)
        for _ in range(num_spots):
            spot_x = rng.randint(0, patch_w-1)
            spot_y = rng.randint(0, patch_h-1)
            if mask_patch[spot_y, spot_x] > 0:  # Only in defect region
                spot_size = rng.randint(8, 20)
                spot_mask = create_blob_mask(patch_h, patch_w, spot_x, spot_y, spot_size, np_rng)
                spot_mask *= defect_mask  # Constrain to defect region
                contamination_color = np_rng.uniform(0, 80, 3)
                patch = patch * (1 - spot_mask[:, :, None] * 0.7) + contamination_color * spot_mask[:, :, None] * 0.7
    
    elif defect_type == 'scratch':
        # Visible scratches
        num_scratches = rng.randint(2, 5)
        for _ in range(num_scratches):
            scratch_mask = create_scratch_pattern(patch_h, patch_w, rng, np_rng)
            scratch_mask *= defect_mask  # Constrain to defect region
            patch = patch * (1 - scratch_mask[:, :, None] * 0.6)
    
    elif defect_type == 'texture_change':
        # High-frequency texture noise
        noise = np_rng.normal(0, 25, (patch_h, patch_w, 3))
        patch = patch + noise * defect_mask[:, :, None]
    
    elif defect_type == 'brightness':
        # Overall brightness change
        factor = rng.uniform(0.6, 1.4)
        patch = patch * (1 - defect_mask[:, :, None]) + patch * factor * defect_mask[:, :, None]

    patch = np.clip(patch, 0, 255)

    # ═══════════════════════════════════════════════
    # SEAMLESS POISSON-LIKE BLENDING
    # ═══════════════════════════════════════════════
    
    # Create smooth blending mask with proper feathering
    blend_mask = mask_patch.astype(np.float32)
    feather_size = max(7, min(15, min(patch_h, patch_w) // 8))
    blend_mask = cv2.GaussianBlur(blend_mask, (feather_size*2+1, feather_size*2+1), feather_size/3)
    
    # Multi-band blending for seamless integration
    result = seamless_blend(patch, dest_region, blend_mask, blend_alpha)
    
    # ═══════════════════════════════════════════════
    # APPLY TO IMAGE
    # ═══════════════════════════════════════════════
    
    augmented = image.copy()
    augmented[dy:dy + patch_h, dx:dx + patch_w] = np.clip(result, 0, 255).astype(np.uint8)

    # Create final CLEAN binary mask (white=anomaly, black=normal)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy:dy + patch_h, dx:dx + patch_w] = mask_patch * 255  # 255 for anomaly

    return augmented, mask


# ──────────────────────────────────────────────
# Helper Functions for Advanced CutPaste
# ──────────────────────────────────────────────

def seamless_blend(patch, dest, blend_mask, alpha):
    """
    Advanced seamless blending using gradient domain techniques.
    Combines patch and destination while preserving gradients.
    """
    mask_3d = blend_mask[:, :, None]
    
    # Method 1: Direct alpha blending with gradient preservation
    result = patch * mask_3d * alpha + dest * (1 - mask_3d * alpha)
    
    # Method 2: Add gradient matching at boundaries
    # Get boundary region (where mask transitions)
    kernel = np.ones((5, 5), np.uint8)
    mask_binary = (blend_mask > 0.5).astype(np.uint8)
    mask_dilated = cv2.dilate(mask_binary, kernel, iterations=2)
    mask_eroded = cv2.erode(mask_binary, kernel, iterations=2)
    boundary = (mask_dilated - mask_eroded).astype(np.float32)
    
    if boundary.sum() > 0:
        # Convert to uint8 for Sobel (OpenCV requirement)
        patch_uint8 = np.clip(patch, 0, 255).astype(np.uint8)
        dest_uint8 = np.clip(dest, 0, 255).astype(np.uint8)
        
        # Compute gradients on each channel separately
        patch_grad_x = np.zeros_like(patch)
        patch_grad_y = np.zeros_like(patch)
        dest_grad_x = np.zeros_like(dest)
        dest_grad_y = np.zeros_like(dest)
        
        for c in range(3):
            patch_grad_x[:, :, c] = cv2.Sobel(patch_uint8[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
            patch_grad_y[:, :, c] = cv2.Sobel(patch_uint8[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
            dest_grad_x[:, :, c] = cv2.Sobel(dest_uint8[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
            dest_grad_y[:, :, c] = cv2.Sobel(dest_uint8[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        
        # Blend gradients in boundary region
        boundary_3d = boundary[:, :, None]
        blended_grad_x = patch_grad_x * (1 - boundary_3d) + dest_grad_x * boundary_3d
        blended_grad_y = patch_grad_y * (1 - boundary_3d) + dest_grad_y * boundary_3d
        
        # Apply gradient correction
        correction = (blended_grad_x - patch_grad_x + blended_grad_y - patch_grad_y) * boundary_3d * 0.3
        result = result + correction
    
    return result


def create_blob_mask(h, w, cx, cy, size, np_rng):
    """Create irregular blob for contamination - more visible."""
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    blob = np.exp(-(dist**2) / (2 * size**2))
    # Add noise to make irregular
    noise = np_rng.uniform(0.8, 1.2, (h, w))
    blob = blob * noise
    return np.clip(blob, 0, 1)


def create_scratch_pattern(h, w, rng, np_rng):
    """Create highly visible scratch-like linear patterns."""
    mask = np.zeros((h, w), dtype=np.float32)
    num_scratches = rng.randint(2, 5)
    
    for _ in range(num_scratches):
        # Random line with better visibility
        x1, y1 = rng.randint(0, w-1), rng.randint(0, h-1)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.randint(min(h, w)//3, min(h, w))
        x2 = int(np.clip(x1 + length * math.cos(angle), 0, w-1))
        y2 = int(np.clip(y1 + length * math.sin(angle), 0, h-1))
        
        # Thicker scratches for visibility
        thickness = rng.randint(2, 5)
        cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness)
        
        # Add jagged edges for realism
        num_points = max(3, int(length / 20))
        for i in range(1, num_points):
            t = i / num_points
            px = int(x1 + (x2 - x1) * t)
            py = int(y1 + (y2 - y1) * t)
            # Random jitter
            jitter_x = rng.randint(-3, 3)
            jitter_y = rng.randint(-3, 3)
            px = np.clip(px + jitter_x, 0, w-1)
            py = np.clip(py + jitter_y, 0, h-1)
            cv2.circle(mask, (px, py), thickness//2, 1.0, -1)
    
    # Light blur for natural look but keep visible
    mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
    mask = np.clip(mask, 0, 1)
    return mask


# ──────────────────────────────────────────────
# Realistic Crack Generation
# ──────────────────────────────────────────────

def crack_anomaly(
    image: np.ndarray,
    num_cracks: int = 3,
    min_length: float = 0.1,
    max_length: float = 0.4,
    min_width: int = 1,
    max_width: int = 3,
    darkness: float = 0.3,
    branching_prob: float = 0.3,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Highly realistic and VISIBLE crack generator with:
    - Much more visible cracks (thicker, darker, more prominent)
    - Clean binary masks (white=anomaly, black=normal)
    - Natural fractal branching patterns
    - Variable width and intensity
    - Realistic crack characteristics (edges, depth, color changes)
    """
    rng = rng or random
    np_rng = np_rng or np.random.default_rng()
    h, w = image.shape[:2]
    diag = math.sqrt(h**2 + w**2)

    augmented = image.copy().astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)  # Binary mask from start

    def draw_visible_crack(start, length, width, angle_init, depth=0, parent_intensity=1.0):
        """Draw highly visible crack with clean masking."""
        if depth > 3 or length < 15:
            return

        # More segments for smoother curves
        num_segments = max(8, int(length / 6))
        angle = angle_init
        pts = [start]
        widths = []

        # Generate smooth path
        for i in range(1, num_segments + 1):
            progress = i / num_segments
            
            # Gentler angle changes for natural look
            angle_change = np_rng.normal(0, 0.1)
            angle += angle_change
            
            # Variable segment length
            segment_length = (length / num_segments) * rng.uniform(0.85, 1.15)
            
            dx = segment_length * math.cos(angle)
            dy = segment_length * math.sin(angle)
            
            x, y = pts[-1]
            nx = int(np.clip(x + dx, 0, w - 1))
            ny = int(np.clip(y + dy, 0, h - 1))
            pts.append((nx, ny))
            
            # Width varies but stays visible (minimum 2 pixels)
            base_width = max(2, width * (1 - abs(progress - 0.5) * 0.3))  # Less taper
            local_width = max(2, int(base_width * rng.uniform(0.9, 1.1)))
            widths.append(local_width)

        # Draw crack with high visibility
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            w_crack = widths[i] if i < len(widths) else widths[-1]
            
            # Ensure minimum width for visibility
            w_crack = max(3, w_crack)
            
            # ═══════════════════════════════════════════
            # DRAW ON BINARY MASK (clean white line)
            # ═══════════════════════════════════════════
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=w_crack)
            
            # ═══════════════════════════════════════════
            # DARKEN IMAGE (make crack VISIBLE)
            # ═══════════════════════════════════════════
            
            # Get local region
            pad = w_crack * 3
            region_y1 = max(0, min(y1, y2) - pad)
            region_y2 = min(h, max(y1, y2) + pad)
            region_x1 = max(0, min(x1, x2) - pad)
            region_x2 = min(w, max(x1, x2) + pad)
            
            if region_y2 > region_y1 and region_x2 > region_x1:
                region = augmented[region_y1:region_y2, region_x1:region_x2]
                local_brightness = region.mean()
                
                # MUCH stronger darkening for visibility
                # Darker on bright surfaces, still visible on dark surfaces
                if local_brightness > 100:
                    adaptive_darkness = darkness * 1.5  # Even darker on bright surfaces
                else:
                    adaptive_darkness = darkness * 0.8  # But still visible on dark
                
                # Apply strong darkening to crack pixels
                for dy in range(region_y2 - region_y1):
                    for dx in range(region_x2 - region_x1):
                        global_y, global_x = region_y1 + dy, region_x1 + dx
                        
                        # Distance to crack line
                        dist = point_to_segment_distance((global_x, global_y), (x1, y1), (x2, y2))
                        
                        if dist <= w_crack:
                            # Core crack - very dark
                            falloff = 1.0 - (dist / w_crack) ** 0.5  # Slower falloff
                            augmented[global_y, global_x] *= (1 - adaptive_darkness * falloff * 0.9)
                        elif dist <= w_crack * 2:
                            # Outer shadow - lighter but still visible
                            falloff = 1.0 - ((dist - w_crack) / w_crack)
                            augmented[global_y, global_x] *= (1 - adaptive_darkness * falloff * 0.4)
            
            # Add depth effect (dark core + lighter edges)
            # Draw darker core
            core_width = max(1, w_crack - 1)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=core_width)
            
            # Darken core even more
            for dy in range(-core_width, core_width + 1):
                for dx in range(-core_width, core_width + 1):
                    if math.sqrt(dx**2 + dy**2) <= core_width:
                        py, px = y1 + dy, x1 + dx
                        if 0 <= py < h and 0 <= px < w:
                            augmented[py, px] *= 0.5  # Very dark core
        
        # Add edge highlights for 3D effect (makes it more visible)
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            w_crack = widths[i] if i < len(widths) else widths[-1]
            
            # Add subtle highlight on one side
            if rng.random() < 0.4:
                angle_line = math.atan2(y2 - y1, x2 - x1)
                angle_perp = angle_line + math.pi / 2
                offset_dir = rng.choice([-1, 1])
                
                ox = int((w_crack + 2) * offset_dir * math.cos(angle_perp))
                oy = int((w_crack + 2) * offset_dir * math.sin(angle_perp))
                
                hx1, hy1 = x1 + ox, y1 + oy
                hx2, hy2 = x2 + ox, y2 + oy
                
                if 0 <= hx1 < w and 0 <= hy1 < h and 0 <= hx2 < w and 0 <= hy2 < h:
                    # Slight brightening for 3D effect
                    # augmented is float32, so use float color values
                    max_val = float(np.max(augmented))
                    highlight_val = float(min(255.0, max_val * 0.2))
                    cv2.line(augmented, (int(hx1), int(hy1)), (int(hx2), int(hy2)),
                    color=(highlight_val, highlight_val, highlight_val), thickness=1)


        # Micro-cracks for realism (but still visible)
        if depth < 2 and rng.random() < 0.3:
            num_micro = rng.randint(1, 2)
            for _ in range(num_micro):
                if len(pts) > 2:
                    micro_idx = rng.randint(1, len(pts) - 2)
                    micro_start = pts[micro_idx]
                    micro_length = length * rng.uniform(0.15, 0.35)
                    micro_angle = angle_init + rng.uniform(-math.pi/3, math.pi/3)
                    micro_width = max(2, width - 1)  # Still visible
                    draw_visible_crack(
                        micro_start, micro_length, micro_width,
                        micro_angle, depth + 2, parent_intensity * 0.7
                    )

        # Branching
        if rng.random() < branching_prob and depth < 2:
            if len(pts) > 3:
                branch_idx = rng.randint(len(pts) // 4, 3 * len(pts) // 4)
                branch_start = pts[branch_idx]
                branch_length = length * rng.uniform(0.35, 0.65)
                branch_angle = angle + rng.uniform(-math.pi/4, math.pi/4)
                branch_width = max(2, width - 1)  # Keep visible
                draw_visible_crack(
                    branch_start, branch_length, branch_width,
                    branch_angle, depth + 1, parent_intensity * 0.85
                )

    # Generate main cracks with better parameters for visibility
    for _ in range(num_cracks):
        # Start position
        start_x = rng.randint(int(w*0.1), int(w*0.9))
        start_y = rng.randint(int(h*0.1), int(h*0.9))
        start = (start_x, start_y)
        
        # Crack properties - ensure visibility
        length = diag * rng.uniform(min_length, max_length)
        width = rng.randint(max(3, min_width), max(5, max_width))  # Minimum 3 pixels wide
        initial_angle = rng.uniform(0, 2 * math.pi)
        
        draw_visible_crack(start, length, width, initial_angle)

    # ═══════════════════════════════════════════════
    # POST-PROCESSING FOR REALISM
    # ═══════════════════════════════════════════════

    # 1. Add subtle discoloration around cracks
    mask_float = mask.astype(np.float32) / 255.0
    mask_dilated = cv2.dilate((mask_float * 255).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
    mask_dilated = mask_dilated.astype(np.float32) / 255.0
    
    stain_mask = mask_dilated - mask_float
    stain_mask = cv2.GaussianBlur(stain_mask, (11, 11), 3)
    
    # Subtle color shift
    for c in range(3):
        color_shift = rng.uniform(-10, 10)
        augmented[:, :, c] += stain_mask * color_shift

    # 2. Add texture variation along cracks
    texture_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    texture_noise = np_rng.normal(0, 3, (h, w, 3))
    augmented += texture_noise * (texture_mask[:, :, None] / 255.0)

    # 3. Very light blur to remove digital artifacts (but keep visibility)
    augmented = cv2.GaussianBlur(augmented, (3, 3), 0.3)

    # Final conversion
    augmented = np.clip(augmented, 0, 255).astype(np.uint8)

    # Clean binary mask - already have it (255 = crack, 0 = normal)
    # Just do minimal cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return augmented, mask


def point_to_segment_distance(point, seg_start, seg_end):
    """Calculate minimum distance from point to line segment."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


# ──────────────────────────────────────────────
# Category-specific method mapping
# ──────────────────────────────────────────────

# Define which synthetic methods are appropriate for each MVTec category
CATEGORY_METHODS = {
    "bottle": ["cutpaste", "crack"],
    "cable": ["cutpaste", "crack"],
    "capsule": ["cutpaste", "crack"],
    "carpet": ["cutpaste"],  # Cracks don't make sense for soft materials
    "grid": ["cutpaste", "crack"],
    "hazelnut": ["cutpaste", "crack"],
    "leather": ["cutpaste"],  # Cracks don't make sense for leather
    "metal_nut": ["cutpaste", "crack"],
    "pill": ["cutpaste", "crack"],
    "screw": ["cutpaste", "crack"],
    "tile": ["cutpaste", "crack"],
    "toothbrush": ["cutpaste", "crack"],
    "transistor": ["cutpaste", "crack"],
    "wood": ["cutpaste", "crack"],
    "zipper": ["cutpaste"],  # Cracks don't make sense for zippers
    # Default fallback
    "default": ["cutpaste"],
}


def get_methods_for_category(category: str) -> list[str]:
    """Get list of synthetic methods appropriate for a category."""
    return CATEGORY_METHODS.get(category.lower(), CATEGORY_METHODS["default"])


# ──────────────────────────────────────────────
# Patch-level mask
# ──────────────────────────────────────────────

def mask_to_patch_labels(
    mask: np.ndarray,
    patch_size: int = 14,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Convert a pixel mask to patch-level binary labels.

    A patch is labelled anomalous if more than `threshold` fraction of its
    pixels are anomalous.

    Args:
        mask: (H, W) binary mask.
        patch_size: ViT patch size.
        threshold: fraction of anomalous pixels to mark a patch as positive.

    Returns:
        (N_patches,) float tensor of patch labels in {0, 1}.
    """
    h, w = mask.shape
    gh, gw = h // patch_size, w // patch_size
    labels = []
    for i in range(gh):
        for j in range(gw):
            patch = mask[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            ratio = patch.mean()
            labels.append(1.0 if ratio > threshold else 0.0)
    return torch.tensor(labels, dtype=torch.float32)  # (N_patches,)