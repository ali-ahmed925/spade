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
from models.builder import describe, load_spade
from models.llm import FrozenLLM
from utils.heatmap import patches_to_heatmap, overlay_heatmap, save_heatmap
from utils.logging import get_logger


def load_config() -> dict:
    cfg = {}
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    # "train" is included for training.checkpoint_dir so checkpoint paths resolve
    # identically here and in train.py/eval.py; its other keys are unused.
    for name in ("model", "data", "llm", "train"):
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
    model, ckpt_meta = load_spade(cfg, args.checkpoint, device=device, logger=logger)
    logger.info(f"model: {describe(model)}")


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
