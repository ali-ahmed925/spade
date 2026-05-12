"""SPADE Demo Web Application.

Single-file FastAPI app that serves the frontend and runs SPADE inference.
Start with: python app.py
"""

import asyncio
import base64
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import requests
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from PIL import Image as PILImage
from pydantic import BaseModel

# ── Local imports (SPADE pipeline) ────────────────────────────────────────────
from data.transforms import get_eval_transforms
from models.spade import SPADE
from utils.heatmap import patches_to_heatmap, overlay_heatmap

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL_CFG_PATH = os.path.join(os.path.dirname(__file__), "config", "model.yaml")
with open(MODEL_CFG_PATH) as f:
    MODEL_CFG = yaml.safe_load(f)

DATASET_ROOT    = CFG["dataset_root"]
CHECKPOINTS_ROOT = CFG["checkpoints_root"]
CURATED         = CFG["curated_images"]
THRESHOLDS      = CFG.get("thresholds", {})
HYPERPARAMS     = CFG["hyperparameters"]
MODE            = CFG.get("mode", "localhost").upper()


def _build_image_pool(max_defect: int = 30) -> Dict[str, list]:
    """Scan test directories and build a pool of up to max_defect defect images.

    Excludes the 'good' subfolder. Falls back to curated list if the test dir
    doesn't exist or is empty.
    """
    pool: Dict[str, list] = {}
    for cat in CURATED.keys():
        test_dir = os.path.join(DATASET_ROOT, cat, "test")
        cat_root = os.path.join(DATASET_ROOT, cat)
        discovered: list = []
        if os.path.isdir(test_dir):
            defect_imgs = [
                p for p in Path(test_dir).rglob("*.png")
                if "good" not in p.parts
            ]
            random.shuffle(defect_imgs)
            discovered = [
                str(p.relative_to(cat_root))
                for p in defect_imgs[:max_defect]
            ]
        if discovered:
            pool[cat] = discovered
            print(f"[SPADE] Image pool: {cat} → {len(discovered)} images")
        else:
            pool[cat] = [p for p in CURATED.get(cat, []) if p]
            print(f"[SPADE] Image pool: {cat} → {len(pool[cat])} images (curated fallback)")
    return pool


_IMAGE_POOL = _build_image_pool(max_defect=30)

_LLM_ENDPOINT = "http://localhost:11434/v1/chat/completions"
_LLM_MODEL    = "llama3.2-vision:11b"   # switch: "qwen2.5vl:32b" | "llama3.2-vision:11b"

# ── Per-class visual context for the vision model ─────────────────────────────
# Each entry: (visual description, expected defect types)
_CLASS_CONTEXT: Dict[str, tuple] = {
    "bottle": (
        "A glass bottle with a cylindrical body, horizontal rings or ridges around its midsection, "
        "a narrow neck, and a flat base. The glass is typically transparent or amber-brown.",
        "Possible defects: large or small cracks and breaks in the glass body or neck, "
        "surface contamination such as dirt, dark stains, or foreign residue.",
    ),
    "cable": (
        "A multi-conductor electrical cable containing several individual wires with colored "
        "plastic insulation (commonly black, blue, red, white) bundled together.",
        "Possible defects: bent or kinked wires, swapped cable positions, cut or poked outer "
        "or inner insulation, missing wire segments, or combined damage.",
    ),
    "capsule": (
        "A small two-part pharmaceutical capsule — cylindrical, with two distinct color halves "
        "(e.g. white body and orange cap) that fit together at the middle seam.",
        "Possible defects: surface cracks, deep scratches, poke holes, faulty or missing "
        "printed text/markings, or squeezed/deformed shape.",
    ),
    "carpet": (
        "A flat woven carpet viewed from directly above, showing a regular repeating geometric "
        "or diamond-shaped weave pattern with uniform color.",
        "Possible defects: color discoloration or staining, cut or frayed fibers, punched holes, "
        "embedded metal contamination, or stray loose threads crossing the pattern.",
    ),
    "grid": (
        "A metal wire mesh or grid with uniform square or rectangular openings, "
        "viewed flat-on. The wires cross at right angles with consistent spacing.",
        "Possible defects: bent or deformed wires, broken strands leaving open gaps, "
        "glue residue on the surface, embedded metal contamination, or foreign thread material.",
    ),
    "hazelnut": (
        "A single hazelnut viewed close-up — round to slightly oval, with a hard brown shell "
        "that has a rough, mottled, natural texture.",
        "Possible defects: cracks or cuts running across the shell surface, drilled or punched "
        "holes, or printed markings/stamps that should not appear.",
    ),
    "leather": (
        "A flat piece of leather viewed from above, showing a natural fine-grain texture "
        "with a subtle sheen. The surface should be smooth and uniform in color.",
        "Possible defects: color discoloration or uneven patches, straight cut marks, "
        "fold creases, glue residue spots, or poke holes.",
    ),
    "metal_nut": (
        "A hexagonal metal nut viewed from above, with six flat sides and a threaded "
        "circular hole in the center. The metal surface appears machined and uniform.",
        "Possible defects: bent or deformed hex edges, color anomalies such as rust or "
        "dark discoloration, flipped orientation, or scratches across the flat faces.",
    ),
    "pill": (
        "A small round or oval pharmaceutical pill — typically white, smooth, and flat "
        "on both sides with a uniform surface and possibly an imprinted text or score line.",
        "Possible defects: color deviation (dark spots, staining), surface contamination, "
        "cracks, chipped edges, faulty or missing imprint text, scratches, or wrong pill type.",
    ),
    "screw": (
        "A metal screw with a circular head (Phillips cross or flat slot drive) and a "
        "cylindrical shank with helical threads running along its length.",
        "Possible defects: scratches on the head face or neck, damaged or irregular thread "
        "pattern on the shank, manipulated or bent tip, or a missing/deformed drive slot.",
    ),
    "tile": (
        "A flat ceramic or stone tile viewed from above with a uniform, slightly rough surface "
        "texture. The tile should have a consistent color and clean edges.",
        "Possible defects: cracks running across the surface, gray stroke marks, oil stains, "
        "rough or pitted patches, or dried glue strip residue.",
    ),
    "toothbrush": (
        "A manual toothbrush with a plastic handle and a rectangular bristle head containing "
        "parallel rows of white nylon bristles standing upright.",
        "Possible defects: missing, bent, flattened, or splayed bristles in one or more tufts, "
        "deformed or cracked handle, or irregular bristle height.",
    ),
    "transistor": (
        "An electronic transistor component — small rectangular black epoxy body with three "
        "metal leads (pins) extending from the base, mounted on a flat surface.",
        "Possible defects: bent or kinked leads, cut or missing leads, cracked or damaged "
        "epoxy casing, or a misplaced/rotated component.",
    ),
    "wood": (
        "A flat wood panel viewed from above showing natural wood grain lines running roughly "
        "parallel across the surface, with occasional knots.",
        "Possible defects: color stains or discoloration patches, drilled or punched holes, "
        "liquid spill marks, surface scratches, or combined multiple damage types.",
    ),
    "zipper": (
        "A zipper viewed flat-on, showing two parallel rows of interlocking plastic or metal "
        "teeth along a fabric tape, with a center spine where the teeth mesh together.",
        "Possible defects: broken or missing teeth, split teeth that no longer interlock, "
        "squeezed or deformed teeth, rough surface damage, or torn fabric at the border or interior.",
    ),
}

def _build_prompt(category: str) -> str:
    """Build a category-specific 3-sentence prompt for the vision model."""
    ctx = _CLASS_CONTEXT.get(category)
    if ctx:
        visual_desc, defect_types = ctx
        return (
            f"You are performing industrial quality inspection on a {category}. "
            f"Object appearance: {visual_desc} "
            f"{defect_types} "
            "Identify the single most prominent defect visible in this image. "
            "Respond in exactly 3 sentences: (1) describe what the defect looks like, "
            "(2) state where on the object it is located, "
            "(3) assess how severe it appears. "
            "Be direct and precise. Do not explain causes or recommendations."
        )
    # Fallback for unknown categories
    return (
        "This is an MVTec industrial inspection image. "
        "Identify the single most prominent defect visible. "
        "Respond in exactly 3 sentences: describe what it looks like, where it is located, "
        "and how severe it appears. Be direct. Do not explain causes or recommendations."
    )

# ── Model cache ────────────────────────────────────────────────────────────────
_model_cache: Dict[str, SPADE] = {}
_model_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SPADE] Using device: {_device}")


def _load_spade(category: str) -> SPADE:
    ckpt_path = os.path.join(CHECKPOINTS_ROOT, category, "spade_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint for category '{category}' at {ckpt_path}")

    model = SPADE(
        blip2_model_name=MODEL_CFG["blip2"]["model_name"],
        llm_embed_dim=MODEL_CFG["projection"]["output_dim"],
        hpa_n_max=MODEL_CFG["hpa"]["n_max"],
        hpa_n_min=MODEL_CFG["hpa"]["n_min"],
        hpa_t_steps=MODEL_CFG["hpa"]["t_steps"],
        hpa_w=MODEL_CFG["hpa"]["w"],
        hpa_p1=MODEL_CFG["hpa"]["p1"],
        hpa_p2=MODEL_CFG["hpa"]["p2"],
        score_alpha=MODEL_CFG["scoring"]["alpha"],
        score_beta=MODEL_CFG["scoring"]["beta"],
        score_lambda=MODEL_CFG["scoring"]["lambda"],
        mahalanobis_gamma=MODEL_CFG["scoring"]["mahalanobis_gamma"],
        mahalanobis_reg=MODEL_CFG["scoring"]["mahalanobis_reg"],
        normal_stats_buffer_size=MODEL_CFG["normal_stats"]["buffer_size"],
        normal_stats_update_frequency=MODEL_CFG["normal_stats"]["update_frequency"],
    ).to(_device)

    if MODEL_CFG.get("frequency", {}).get("enabled", False):
        model.enable_frequency_features(
            freq_num_bands=MODEL_CFG["frequency"].get("num_bands", 6),
            freq_use_phase=MODEL_CFG["frequency"].get("use_phase", True),
            freq_feature_dim=MODEL_CFG["frequency"].get("feature_dim", 32),
            score_gamma=MODEL_CFG["scoring"].get("gamma", 0.25),
        )

    state = torch.load(ckpt_path, map_location=_device, weights_only=True)
    key = "model_state_dict" if "model_state_dict" in state else None
    model.load_state_dict(state[key] if key else state, strict=False)
    model.to(_device)
    model.use_hpa = bool(MODEL_CFG.get("hpa", {}).get("enabled", False))
    model.eval()
    return model


def get_model(category: str) -> SPADE:
    with _model_lock:
        if category not in _model_cache:
            # Evict all cached models to free VRAM before loading a new one
            for key, old_model in list(_model_cache.items()):
                print(f"[SPADE] Evicting model for '{key}' to free VRAM")
                old_model.cpu()
                del _model_cache[key]
            torch.cuda.empty_cache()
            print(f"[SPADE] Loading model for '{category}'...")
            _model_cache[category] = _load_spade(category)
            print(f"[SPADE] Model ready for '{category}'")
        return _model_cache[category]


# ── Inference ──────────────────────────────────────────────────────────────────
def _get_ground_truth_path(image_path: str) -> Optional[str]:
    """Derive MVTec ground truth mask path from a test image path."""
    try:
        p = Path(image_path)
        parts = list(p.parts)
        test_idx = parts.index("test")
        gt_parts = parts.copy()
        gt_parts[test_idx] = "ground_truth"
        gt_parts[-1] = p.stem + "_mask.png"
        return str(Path(*gt_parts))
    except (ValueError, Exception):
        return None


def _encode_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(image_path: str, category: str) -> dict:
    model = get_model(category)
    image_size = MODEL_CFG["vit"]["image_size"]
    patch_size = MODEL_CFG["vit"]["patch_size"]

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))

    transform = get_eval_transforms(image_size)
    tensor = transform(PILImage.fromarray(img_resized)).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(tensor)

    patch_scores = outputs["patch_scores"]
    image_score  = float(model.get_image_score(patch_scores).cpu())

    # Normalize scores for visualization
    ps_np = patch_scores[0].detach().cpu().numpy()
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
    # Raw heatmap: pure colormap, no original image blended in
    raw_heatmap_img = overlay_heatmap(np.zeros_like(img_resized), heatmap, alpha=1.0, colormap_name="hot")
    # Overlay: heatmap blended onto original
    overlay_img = overlay_heatmap(img_resized, heatmap, colormap_name="hot")

    # Original image (resized to inference size)
    original_b64    = _encode_png(img_resized)
    heatmap_b64     = _encode_png(raw_heatmap_img)
    overlay_b64     = _encode_png(overlay_img)

    # Ground truth mask (optional)
    ground_truth_b64: Optional[str] = None
    gt_path = _get_ground_truth_path(image_path)
    if gt_path and os.path.exists(gt_path):
        try:
            gt_gray    = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_resized = cv2.resize(gt_gray, (image_size, image_size))
            gt_rgb     = cv2.cvtColor(gt_resized, cv2.COLOR_GRAY2RGB)
            ground_truth_b64 = _encode_png(gt_rgb)
        except Exception:
            pass

    # Verdict
    threshold = THRESHOLDS.get(category, 1.0)
    verdict = "ANOMALOUS" if image_score > threshold else "NORMAL"

    return {
        "anomaly_score":    image_score,
        "original_b64":     original_b64,
        "heatmap_b64":      heatmap_b64,
        "overlay_b64":      overlay_b64,
        "ground_truth_b64": ground_truth_b64,
        "verdict":          verdict,
    }


def call_vlm(image_path: str, category: str = "") -> str:
    try:
        prompt = _build_prompt(category)
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext  = Path(image_path).suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        resp = requests.post(
            _LLM_ENDPOINT,
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"},
            json={
                "model": _LLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "stream": False,
            },
            timeout=420,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        # Some VLMs return content as a list of parts
        if isinstance(raw, list):
            raw = " ".join(p.get("text", "") for p in raw if isinstance(p, dict))
        text = (raw or "").strip()
        print(f"[VLM] response length={len(text)} chars, preview={text[:80]!r}")
        return text if text else "Language decoder returned an empty response."
    except Exception as e:
        print(f"[VLM] error: {e}")
        return f"Language decoder output unavailable: {e}"



# ── FastAPI ────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
app = FastAPI(title="SPADE Demo")
app.mount("/static", StaticFiles(directory=os.path.join(_BASE, "static")), name="static")
_templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))


class SelectImageRequest(BaseModel):
    category: str

class AnalyzeRequest(BaseModel):
    image_path: str
    category: str


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(_device), "mode": MODE}


@app.get("/api/config")
def get_config():
    return {"hyperparameters": HYPERPARAMS, "mode": MODE}


@app.get("/api/categories")
def get_categories():
    cats = []
    for cat in CURATED.keys():
        ckpt = os.path.join(CHECKPOINTS_ROOT, cat, "spade_best.pt")
        cats.append({
            "name": cat,
            "count": len(_IMAGE_POOL.get(cat, [])),
            "has_checkpoint": os.path.exists(ckpt),
        })
    return cats


@app.get("/api/image/{category}/{image_path:path}")
def serve_image(category: str, image_path: str):
    full = os.path.normpath(os.path.join(DATASET_ROOT, category, image_path))
    root = os.path.normpath(DATASET_ROOT)
    if not full.startswith(root):
        raise HTTPException(403, "Access denied")
    if not os.path.exists(full):
        raise HTTPException(404, "Image not found")
    return FileResponse(full)


@app.get("/api/slot-images/{category}")
def slot_images(category: str):
    """Return up to 10 defect images as base64 data URIs for the slot-machine animation.

    Embeds images directly to avoid per-image round trips through the tunnel.
    """
    test_dir = os.path.join(DATASET_ROOT, category, "test")
    all_imgs = list(Path(test_dir).rglob("*.png"))
    defect = [p for p in all_imgs if "good" not in p.parts]
    random.shuffle(defect)
    pool = defect[:10] if len(defect) >= 10 else defect
    result = []
    for p in pool:
        try:
            img = PILImage.open(p).convert("RGB")
            img.thumbnail((224, 224), PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode()
            result.append(f"data:image/jpeg;base64,{b64}")
        except Exception:
            pass
    return result


@app.post("/api/select-image")
def select_image(req: SelectImageRequest):
    cat = req.category
    if cat not in _IMAGE_POOL:
        raise HTTPException(404, f"Category '{cat}' not found")
    imgs = [p for p in _IMAGE_POOL[cat] if p]
    if not imgs:
        raise HTTPException(404, "No images available for this category")

    # Pick a random image from the pool; verify it exists on disk
    candidates = imgs.copy()
    random.shuffle(candidates)
    rel_path = None
    full_path = None
    for candidate in candidates:
        fp = os.path.join(DATASET_ROOT, cat, candidate)
        if os.path.exists(fp):
            rel_path = candidate
            full_path = fp
            break

    if full_path is None:
        raise HTTPException(404, "No accessible test images found on disk")

    with open(full_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    defect_type = Path(rel_path).parent.name
    return {
        "image_path": full_path,
        "rel_path": rel_path,
        "defect_type": defect_type,
        "category": cat,
        "image_b64": b64,
    }


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, run_inference, req.image_path, req.category
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")


class ExplainRequest(BaseModel):
    image_path: str
    category: str = ""


@app.post("/api/explain")
async def explain(req: ExplainRequest):
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(
            _executor, call_vlm, req.image_path, req.category
        )
        print(f"[/api/explain] cat={req.category!r} len={len(text)} preview={text[:80]!r}")
        return {"explanation": text}
    except Exception as e:
        raise HTTPException(500, f"VLM error: {e}")


@app.get("/", response_class=HTMLResponse)
def serve_frontend(request: Request):
    return _templates.TemplateResponse(request, "index.html")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[SPADE] Starting — mode={MODE}, device={_device}")
    print(f"[SPADE] http://{CFG['host']}:{CFG['port']}")
    uvicorn.run(app, host=CFG["host"], port=CFG["port"], log_level="warning",
                timeout_keep_alive=600, timeout_graceful_shutdown=600)
