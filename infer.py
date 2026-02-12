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
        patch_head_hidden=cfg["patch_head"]["hidden_dim"],
        patch_head_dropout=cfg["patch_head"]["dropout"],
        llm_embed_dim=cfg["projection"]["output_dim"],
    ).to(device)

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
        outputs = model(image_tensor)

    patch_logits = outputs["patch_logits"]    # (1, N)
    visual_tokens = outputs["visual_tokens"]  # (1, Q, D_llm)

    # ── Image-level score ──
    image_score = model.get_image_score(patch_logits).item()
    logger.info(f"Image anomaly score: {image_score:.4f}")

    # ── Heatmap ──
    heatmap = patches_to_heatmap(patch_logits[0], image_size=image_size, patch_size=patch_size)
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
        del image_tensor, patch_logits, visual_tokens
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
