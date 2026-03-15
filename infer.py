"""SPADE inference script.

Given a single image, produces:
  1. Image-level anomaly score
  2. Patch-level heatmap
  3. Natural language explanation (via frozen LLM)

Usage:
    python infer.py --image path/to/image.png --checkpoint checkpoints/spade_best.pt
"""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from data.transforms import get_eval_transforms
from models.spade import SPADE
from models.llm import FrozenLLM
from utils.heatmap import patches_to_heatmap, overlay_heatmap, save_heatmap
from utils.logging import get_logger


def load_config() -> dict:
    cfg = {}
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    for name in ("model", "data", "llm"):
        with open(os.path.join(config_dir, f"{name}.yaml")) as f:
            cfg.update(yaml.safe_load(f))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Dir to save results")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM text generation")
    args = parser.parse_args()

    cfg = load_config()
    logger = get_logger("infer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["vit"]["image_size"]
    patch_size = cfg["vit"]["patch_size"]

    # ── Load model ──
    model = SPADE(
        blip2_model_name=cfg["blip2"]["model_name"],
        llm_embed_dim=cfg["projection"]["output_dim"],
        # HPA parameters
        hpa_n_max=cfg["hpa"]["n_max"],
        hpa_n_min=cfg["hpa"]["n_min"],
        hpa_t_steps=cfg["hpa"]["t_steps"],
        hpa_w=cfg["hpa"]["w"],
        hpa_p1=cfg["hpa"]["p1"],
        hpa_p2=cfg["hpa"]["p2"],
        # Scoring parameters
        score_alpha=cfg["scoring"]["alpha"],
        score_beta=cfg["scoring"]["beta"],
        score_gamma=cfg["scoring"].get("gamma", 0.25),  # Frequency weight
        score_lambda=cfg["scoring"]["lambda"],
        mahalanobis_gamma=cfg["scoring"]["mahalanobis_gamma"],
        mahalanobis_reg=cfg["scoring"]["mahalanobis_reg"],
        use_mahalanobis=cfg["scoring"].get("use_mahalanobis", True),
        # Normal statistics parameters
        normal_stats_buffer_size=cfg["normal_stats"]["buffer_size"],
        normal_stats_update_frequency=cfg["normal_stats"]["update_frequency"],
        # Attention verification
        verify_attention=cfg["scoring"].get("verify_attention", False),
        # Raw attention mode
        use_raw_attention=cfg["scoring"].get("use_raw_attention", True),
    ).to(device)

    # Enable/disable HPA based on config
    use_hpa = cfg.get("hpa", {}).get("enabled", True)
    model.use_hpa = use_hpa
    if use_hpa:
        logger.info("HPA enabled - queries will be refined through hierarchical patch annealing")
    else:
        logger.info("HPA disabled - queries will attend to all patches directly (no refinement)")
    
    # Log Mahalanobis status
    if model.use_mahalanobis:
        logger.info("✅ Mahalanobis scoring enabled")
    else:
        logger.info("❌ Mahalanobis scoring DISABLED - using only attention scores")
    
    # Enable frequency features if configured
    if cfg.get("frequency", {}).get("enabled", False):
        model.enable_frequency_features(
            freq_num_bands=cfg["frequency"].get("num_bands", 6),
            freq_use_phase=cfg["frequency"].get("use_phase", True),
            freq_feature_dim=cfg["frequency"].get("feature_dim", 32),
            score_gamma=cfg["scoring"].get("gamma", 0.25),
        )
        logger.info("Frequency features enabled")
    else:
        logger.info("Frequency features disabled")

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        # Load only trainable parameters (Q-Former + custom heads)
        # Vision encoder will remain frozen from BLIP-2 initialization
        model.load_state_dict(state["model_state_dict"], strict=False)
        checkpoint_epoch = state.get("epoch", "unknown")
        logger.info(f"Loaded checkpoint: {args.checkpoint} (epoch: {checkpoint_epoch})")
        if "config" in state:
            logger.info(f"Checkpoint config: {state['config']}")
    else:
        # Legacy format: assume full state_dict
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint} (legacy format)")
    model.eval()

    # ── Load and preprocess image ──
    image_np = cv2.imread(args.image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (image_size, image_size))
    image_pil = Image.fromarray(image_resized)

    transform = get_eval_transforms(image_size)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # (1, 3, H, W)

    # ── Forward pass ──
    with torch.no_grad():
        outputs = model(image_tensor)  # No update_stats in inference

    patch_scores = outputs["patch_scores"]    # (1, N)
    visual_tokens = outputs["visual_tokens"]  # (1, Q, D_llm)

    # ── Image-level score ──
    image_score = model.get_image_score(patch_scores).item()
    logger.info(f"Image anomaly score: {image_score:.4f}")

    # ── Heatmap (normalize scores for visualization only) ──
    # NOTE: Normalization is ONLY for visualization. For metric computation (pixel AUROC),
    # we would use raw scores without per-image normalization.
    patch_scores_np = patch_scores[0].detach().cpu().numpy()
    p5, p95 = np.percentile(patch_scores_np, [5, 95])
    patch_scores_clipped = np.clip(patch_scores_np, p5, p95)
    if p95 - p5 > 1e-8:
        patch_scores_normalized = (patch_scores_clipped - p5) / (p95 - p5)
    else:
        patch_scores_normalized = np.zeros_like(patch_scores_clipped)
    
    heatmap = patches_to_heatmap(
        torch.from_numpy(patch_scores_normalized),
        image_size=image_size,
        patch_size=patch_size,
        normalize=True,  # Normalize for visualization
    )
    overlay = overlay_heatmap(image_resized, heatmap)

    os.makedirs(args.output_dir, exist_ok=True)
    save_heatmap(heatmap, os.path.join(args.output_dir, "heatmap.png"))
    cv2.imwrite(
        os.path.join(args.output_dir, "overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    logger.info(f"Saved heatmap and overlay to {args.output_dir}/")

    # ── LLM explanation ──
    if not args.no_llm:
        logger.info("Freeing SPADE model from GPU to make room for LLM...")
        # Move visual_tokens to CPU before freeing GPU memory
        visual_tokens_cpu = visual_tokens.cpu()
        # Move SPADE model to CPU and clear GPU cache
        model = model.cpu()
        del image_tensor, patch_scores, visual_tokens
        torch.cuda.empty_cache()
        
        logger.info("Loading frozen LLM for text generation...")
        llm_cfg = cfg["llm"]
        llm = FrozenLLM(
            model_name=llm_cfg["model_name"],
            device_map=llm_cfg["device_map"],
            max_new_tokens=llm_cfg["max_new_tokens"],
            temperature=llm_cfg["temperature"],
            top_p=llm_cfg["top_p"],
        )

        prompt_cfg = cfg["prompt"]
        prompt = prompt_cfg["system"] + "\n" + prompt_cfg["prefix"] + prompt_cfg["suffix"]

        explanations = llm.generate(visual_tokens_cpu, prompt)
        logger.info(f"LLM Explanation:\n{explanations[0]}")

        with open(os.path.join(args.output_dir, "explanation.txt"), "w") as f:
            f.write(explanations[0])
    else:
        logger.info("Skipped LLM generation (--no_llm flag).")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
