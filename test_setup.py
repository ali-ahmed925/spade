"""Quick setup verification script.

Checks if dataset, model, and dependencies are ready.
"""

import os
import sys
from pathlib import Path

import torch
import yaml

print("=" * 60)
print("SPADE Setup Verification")
print("=" * 60)

# ── Check dependencies ──
print("\n1. Checking dependencies...")
try:
    import transformers
    import wandb
    import cv2
    import numpy as np
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Missing dependency: {e}")
    sys.exit(1)

# ── Check GPU ──
print("\n2. Checking GPU...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
else:
    print("   ⚠  No GPU detected (will use CPU - very slow)")

# ── Check config files ──
print("\n3. Checking config files...")
config_dir = Path(__file__).parent / "config"
required_configs = ["model.yaml", "data.yaml", "train.yaml"]
for cfg in required_configs:
    if (config_dir / cfg).exists():
        print(f"   ✓ {cfg}")
    else:
        print(f"   ✗ Missing: {cfg}")
        sys.exit(1)

# ── Check dataset ──
print("\n4. Checking dataset...")
cfg = {}
with open(config_dir / "data.yaml") as f:
    cfg.update(yaml.safe_load(f))

dataset_root = Path(cfg["dataset"]["root"])
category = cfg["dataset"]["category"]
category_path = dataset_root / category

if category_path.exists():
    train_good = category_path / "train" / "good"
    test_good = category_path / "test" / "good"
    
    if train_good.exists():
        num_train = len(list(train_good.glob("*.png")))
        print(f"   ✓ Training samples: {num_train}")
    else:
        print(f"   ✗ Missing: {train_good}")
        sys.exit(1)
    
    if test_good.exists():
        num_test = len(list(test_good.glob("*.png")))
        print(f"   ✓ Test samples: {num_test}")
    else:
        print(f"   ✗ Missing: {test_good}")
        sys.exit(1)
else:
    print(f"   ✗ Dataset not found at: {category_path}")
    print(f"   Expected structure: {category_path}/train/good/*.png")
    sys.exit(1)

# ── Check model can load ──
print("\n5. Testing model initialization...")
try:
    import os
    from models.spade import SPADE
    
    with open(config_dir / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"   ℹ  Using HF_TOKEN from environment")
    else:
        print(f"   ⚠  No HF_TOKEN found - model download may fail if authentication needed")
    
    model = SPADE(
        blip2_model_name=model_cfg["blip2"]["model_name"],
        patch_head_hidden=model_cfg["patch_head"]["hidden_dim"],
        patch_head_dropout=model_cfg["patch_head"]["dropout"],
        llm_embed_dim=model_cfg["projection"]["output_dim"],
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model initialized successfully")
    print(f"   ✓ Trainable parameters: {trainable:,}")
except Exception as e:
    print(f"   ✗ Model initialization failed: {e}")
    print(f"   💡 Make sure HF_TOKEN is set: export HF_TOKEN='your_token'")
    sys.exit(1)

# ── Check dataset loader ──
print("\n6. Testing dataset loader...")
try:
    from data.mvtec_dataset import MVTecDataset
    
    dataset = MVTecDataset(
        root=str(dataset_root),
        category=category,
        split="train",
        image_size=cfg["dataset"]["image_size"],
        patch_size=model_cfg["vit"]["patch_size"],
        synthetic_method="cutpaste",
    )
    
    sample = dataset[0]
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    print(f"   ✓ Sample keys: {list(sample.keys())}")
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Check wandb (optional) ──
print("\n7. Checking wandb...")
with open(config_dir / "train.yaml") as f:
    train_cfg = yaml.safe_load(f)
    
if train_cfg.get("wandb", {}).get("enabled", False):
    try:
        import wandb
        # Just check if wandb is importable and configured
        print(f"   ✓ Wandb enabled (project: {train_cfg['wandb']['project']})")
        print(f"   ⚠  Make sure to run 'wandb login' before training")
    except Exception as e:
        print(f"   ⚠  Wandb enabled but import failed: {e}")
else:
    print("   ℹ  Wandb disabled (set wandb.enabled: true to enable)")

print("\n" + "=" * 60)
print("✓ All checks passed! Ready to run training.")
print("=" * 60)
print("\nNext steps:")
print("  1. If wandb enabled: run 'wandb login'")
print("  2. Start training: python train.py")
print("  3. Monitor progress in wandb dashboard")
print("=" * 60)

