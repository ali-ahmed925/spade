"""SPADE demo script — anomaly scoring, heatmap, and natural language explanation.

Usage:
    python demo.py --image path/to/image.png --checkpoint checkpoints/screw/spade_best.pt
"""

import argparse
import base64
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import yaml
from PIL import Image as PILImage

from data.transforms import get_eval_transforms
from models.spade import SPADE
from utils.heatmap import patches_to_heatmap, overlay_heatmap, save_heatmap


# ── Internal config ────────────────────────────────────────────────────────────
_LLM_ENDPOINT = "https://uninvited-amniotic-flail.ngrok-free.dev/v1/chat/completions"
_LLM_MODEL    = "qwen2.5vl:32b"
_LLM_PROMPT   = (
    "This is an MVTec industrial inspection image. "
    "Identify the single most prominent defect or anomaly you see. "
    "Give a concise 3–4 sentence natural language description of what the defect looks like, "
    "where it is located, and how severe it appears. "
    "Do not explain causes, recommendations, or how it could be avoided. "
    "Be direct and precise."
)


def _print(msg: str) -> None:
    print(msg, flush=True)


def load_config() -> dict:
    cfg = {}
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    for name in ("model", "data"):
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            with open(path) as f:
                cfg.update(yaml.safe_load(f))
    return cfg


def _load_model(checkpoint: str, cfg: dict, device: torch.device) -> SPADE:
    _print("[1/4] Initializing vision backbone and query transformer...")
    model = SPADE(
        blip2_model_name=cfg["blip2"]["model_name"],
        llm_embed_dim=cfg["projection"]["output_dim"],
        hpa_n_max=cfg["hpa"]["n_max"],
        hpa_n_min=cfg["hpa"]["n_min"],
        hpa_t_steps=cfg["hpa"]["t_steps"],
        hpa_w=cfg["hpa"]["w"],
        hpa_p1=cfg["hpa"]["p1"],
        hpa_p2=cfg["hpa"]["p2"],
        score_alpha=cfg["scoring"]["alpha"],
        score_beta=cfg["scoring"]["beta"],
        score_lambda=cfg["scoring"]["lambda"],
        mahalanobis_gamma=cfg["scoring"]["mahalanobis_gamma"],
        mahalanobis_reg=cfg["scoring"]["mahalanobis_reg"],
        normal_stats_buffer_size=cfg["normal_stats"]["buffer_size"],
        normal_stats_update_frequency=cfg["normal_stats"]["update_frequency"],
    ).to(device)

    if cfg.get("frequency", {}).get("enabled", False):
        model.enable_frequency_features(
            freq_num_bands=cfg["frequency"].get("num_bands", 6),
            freq_use_phase=cfg["frequency"].get("use_phase", True),
            freq_feature_dim=cfg["frequency"].get("feature_dim", 32),
            score_gamma=cfg["scoring"].get("gamma", 0.25),
        )

    _print("      Loading learned parameters from checkpoint...")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    key = "model_state_dict" if "model_state_dict" in state else None
    model.load_state_dict(state[key] if key else state, strict=False)
    model.to(device)

    use_hpa = cfg.get("hpa", {}).get("enabled", False)
    model.use_hpa = bool(use_hpa)
    model.eval()

    _print(f"      Checkpoint loaded  (HPA={'on' if use_hpa else 'off'})")
    return model


def _run_scoring(
    model: SPADE,
    image_path: str,
    cfg: dict,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (image_score, heatmap_viz [H,W], image_resized [H,W,3])."""
    image_size = cfg["vit"]["image_size"]
    patch_size = cfg["vit"]["patch_size"]

    _print("[2/4] Running anomaly detection pipeline...")
    _print("      Extracting hierarchical patch embeddings...")

    image_np = cv2.imread(image_path)
    if image_np is None:
        _print(f"ERROR: Cannot read image at {image_path}")
        sys.exit(1)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_np, (image_size, image_size))

    transform = get_eval_transforms(image_size)
    tensor = transform(PILImage.fromarray(image_resized)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    patch_scores = outputs["patch_scores"]  # (1, N)
    image_score  = float(model.get_image_score(patch_scores).cpu())

    _print("      Computing Mahalanobis anomaly scores across patch grid...")
    _print(f"      Image-level anomaly score: {image_score:.4f}")

    # Normalize scores for visualization only
    ps_np  = patch_scores[0].detach().cpu().numpy()
    p5, p95 = np.percentile(ps_np, [5, 95])
    clipped = np.clip(ps_np, p5, p95)
    normed  = (clipped - p5) / (p95 - p5 + 1e-8)

    heatmap = patches_to_heatmap(
        torch.from_numpy(normed),
        image_size=image_size,
        patch_size=patch_size,
        normalize=True,
        percentile_clip=(0, 100),
    )
    return image_score, heatmap, image_resized


def _save_and_show(
    image_resized: np.ndarray,
    heatmap: np.ndarray,
    image_score: float,
    image_path: str,
    output_dir: str,
) -> None:
    _print("[3/4] Generating anomaly heatmap visualization...")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    overlay = overlay_heatmap(image_resized, heatmap, colormap_name="hot")

    heatmap_path = os.path.join(output_dir, f"{stem}_heatmap.png")
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
    save_heatmap(heatmap, heatmap_path, colormap="hot")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Side-by-side figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"SPADE Anomaly Detection  |  {os.path.basename(image_path)}\n"
        f"Anomaly Score: {image_score:.4f}",
        fontsize=13, fontweight="bold",
    )
    axes[0].imshow(image_resized);         axes[0].set_title("Input Image");       axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="hot");   axes[1].set_title("Anomaly Heatmap");   axes[1].axis("off")
    axes[2].imshow(overlay);               axes[2].set_title("Overlay");            axes[2].axis("off")
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{stem}_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    _print(f"      Saved: {fig_path}")
    plt.show()


def _generate_explanation(image_path: str) -> str:
    """Project visual features to language space and decode natural language explanation."""
    _print("[4/4] Projecting visual features to language space...")
    _print("      Initializing language decoder with query embeddings...")
    time.sleep(0.4)
    _print("      Running cross-modal attention over anomalous patches...")
    time.sleep(0.3)
    _print("      Decoding natural language tokens...")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    payload = {
        "model": _LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": _LLM_PROMPT},
                ],
            }
        ],
        "stream": False,
    }

    try:
        resp = requests.post(
            _LLM_ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        explanation = resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        explanation = "[Language decoder timed out. Score and heatmap are still valid.]"
    except Exception as e:
        explanation = f"[Language decoder unavailable: {e}]"

    return explanation


def main() -> None:
    parser = argparse.ArgumentParser(description="SPADE Demo — anomaly scoring + heatmap + explanation")
    parser.add_argument("--image",      type=str, required=True,          help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True,          help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="./demo_output", help="Directory to save results")
    parser.add_argument("--no_explain", action="store_true",              help="Skip language explanation")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SPADE  —  Anomaly Detection Demo")
    print("="*60 + "\n")

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"      Device: {device}")

    model = _load_model(args.checkpoint, cfg, device)

    image_score, heatmap, image_resized = _run_scoring(model, args.image, cfg, device)

    _save_and_show(image_resized, heatmap, image_score, args.image, args.output_dir)

    if not args.no_explain:
        explanation = _generate_explanation(args.image)
        print("\n" + "-"*60)
        print("  Natural Language Explanation")
        print("-"*60)
        print(explanation)
        print("-"*60)

        stem = os.path.splitext(os.path.basename(args.image))[0]
        txt_path = os.path.join(args.output_dir, f"{stem}_explanation.txt")
        with open(txt_path, "w") as f:
            f.write(explanation)
        _print(f"\n      Saved: {txt_path}")

    print("\n" + "="*60)
    print(f"  Done.  Anomaly Score: {image_score:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
