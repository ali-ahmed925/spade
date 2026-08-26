"""P2 — learned context modelling via a query bottleneck.

WHAT THIS TESTS
---------------
D4 established that transistor/cable defects barely move a patch off the normal
manifold (elevation 1.5-3x vs 8-75x on classes that work) and that transistor
defect patches are not separable even from their own neighbours (within-image
AUROC 0.63). D5 showed a HAND-CODED context feature ([x, mean(3x3)]) buys +5
points of within-image separability on cable and nothing on transistor.

D5's aggregator was fixed: it could not learn WHICH neighbours matter or HOW.
This trains that relationship instead of assuming it.

THE MECHANISM
-------------
    256 patch features  ->  cross-attention  ->  32 query tokens   (BOTTLENECK)
    32 query tokens + position  ->  cross-attention  ->  256 reconstructed patches
    loss = reconstruction error on NORMAL images only

32 queries cannot memorise 256 patches, so they are forced to encode the
*typical structure* of a normal object: what belongs where, and in what spatial
relationship. At test time a patch that violates that structure cannot be
reconstructed from the bottleneck, so its residual is large. A bent lead is
still lead — invisible to per-patch appearance scoring — but it is NOT where the
learned layout says a lead should be, so it fails to reconstruct.

This is why it targets exactly the defects that beat every previous attempt.

MAHALANOBIS IS PRESERVED
------------------------
The score is the Mahalanobis distance of the RESIDUAL (x - x_reconstructed),
with mu/Sigma fitted on normal residuals. Mahalanobis remains the scorer; what
changes is the space it operates in — from raw appearance to context-prediction
error.

GOAL 3
------
The 32 bottleneck queries must encode what-and-where to reconstruct the layout,
which is precisely the property the language head needs. `--dump-queries` saves
them for that downstream work.

SELECTION HYGIENE
-----------------
Training uses train/good only, with a held-in split of those normal images for
early stopping. The test val-half is scored once at the end. The held-out half
is never touched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mvtec_dataset import MVTecDataset  # noqa: E402
from data.transforms import get_eval_transforms  # noqa: E402
from eval import load_config  # noqa: E402
from scripts.diagnose_failure import mask_to_patch_grid, split_indices  # noqa: E402
from models.builder import checkpoint_path  # noqa: E402
from scripts.fit_fine_statistics import build_model  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from utils.logging import get_logger  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class QueryBottleneckReconstructor(nn.Module):
    """Q-Former-shaped context model: patches -> few queries -> patches.

    Same structure as the BLIP-2 Q-Former (learned query tokens cross-attending
    to visual features) but small and trained from scratch, so the mechanism can
    be tested in minutes on cached features. If it works, the same objective
    ports onto the real Q-Former, whose outputs already feed the LLM projection.
    """

    def __init__(
        self,
        feature_dim: int = 1408,
        hidden: int = 256,
        n_queries: int = 32,
        n_patches: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.proj_in = nn.Linear(feature_dim, hidden)
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden) * 0.02)
        self.patch_pos = nn.Parameter(torch.randn(1, n_patches, hidden) * 0.02)

        self.encoder = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.encoder_ffn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2),
                          nn.GELU(), nn.Linear(hidden * 2, hidden))
            for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.decoder_ffn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2),
                          nn.GELU(), nn.Linear(hidden * 2, hidden))
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, feature_dim)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, Q, H) bottleneck queries."""
        h = self.proj_in(patches)
        q = self.queries.expand(h.shape[0], -1, -1)
        for attn, ffn in zip(self.encoder, self.encoder_ffn):
            a, _ = attn(q, h, h, need_weights=False)
            q = q + a
            q = q + ffn(q)
        return q

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        """(B, Q, H) -> (B, N, D) reconstructed patch features."""
        p = self.patch_pos.expand(q.shape[0], -1, -1)
        for attn, ffn in zip(self.decoder, self.decoder_ffn):
            a, _ = attn(p, q, q, need_weights=False)
            p = p + a
            p = p + ffn(p)
        return self.proj_out(self.norm(p))

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.encode(patches)
        return self.decode(q), q


# ──────────────────────────────────────────────────────────────────────────────
# Feature caching
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def cache_train_features(model, paths, image_size, device, batch_size=4) -> torch.Tensor:
    from PIL import Image as PILImage

    transform = get_eval_transforms(image_size)
    out, batch = [], []

    def flush(b):
        if b:
            feats = model.vision_encoder(torch.stack(b).to(device))[:, 1:, :].float()
            out.append(feats.cpu())

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        batch.append(transform(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    flush(batch)
    return torch.cat(out)


@torch.no_grad()
def cache_test_features(model, loader, device, grid):
    feats, labels, cov = [], [], []
    for batch in loader:
        f = model.vision_encoder(batch["image"].to(device))[:, 1:, :].float().cpu()
        feats.append(f)
        for i in range(f.shape[0]):
            labels.append(int(batch["label"][i]))
            m = batch["mask"][i]
            m = m.numpy() if isinstance(m, torch.Tensor) else m
            cov.append(mask_to_patch_grid(m, grid))
    return torch.cat(feats), np.array(labels), cov


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train(recon, feats, device, logger, epochs=60, batch_size=8, lr=1e-3,
          val_frac=0.1, noise=0.0, seed=42, patience=None):
    """Reconstruct normal patch features from the query bottleneck."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(feats.shape[0], generator=g)
    n_val = max(2, int(feats.shape[0] * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    logger.info(f"train images={len(train_idx)}  held-in val={len(val_idx)}")

    # Standardize per channel using TRAIN statistics only.
    flat = feats[train_idx].reshape(-1, feats.shape[-1])
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)

    # Patience must scale with the schedule. With cosine T_max=epochs the LR is
    # still high early on, so a fixed small patience fires during the noisy
    # phase and stops at a worse point the longer you plan to train — which is
    # exactly backwards.
    if patience is None:
        patience = max(30, epochs // 4)
    opt = torch.optim.AdamW(recon.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        recon.train()
        order = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for i in range(0, len(order), batch_size):
            x = feats[order[i : i + batch_size]].to(device)
            x = (x - mean.to(device)) / std.to(device)
            inp = x + noise * torch.randn_like(x) if noise > 0 else x
            pred, _ = recon(inp)
            loss = F.mse_loss(pred, x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * x.shape[0]
        sched.step()

        recon.eval()
        with torch.no_grad():
            xv = feats[val_idx].to(device)
            xv = (xv - mean.to(device)) / std.to(device)
            pv, _ = recon(xv)
            vloss = float(F.mse_loss(pv, xv))

        if epoch == 1 or epoch % 25 == 0:
            logger.info(f"  epoch {epoch:3d}  train {total / len(order):.4f}  val {vloss:.4f}")

        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in recon.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"  early stop at epoch {epoch} (best val {best:.4f})")
                break

    if best_state:
        recon.load_state_dict(best_state)
    return mean, std, best


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def residuals(recon, feats, mean, std, device, batch_size=8) -> torch.Tensor:
    recon.eval()
    out = []
    for i in range(0, feats.shape[0], batch_size):
        x = feats[i : i + batch_size].to(device)
        xn = (x - mean.to(device)) / std.to(device)
        pred, _ = recon(xn)
        out.append((xn - pred).cpu())
    return torch.cat(out)


def fit_gaussian(res: torch.Tensor, shrinkage=0.1, reg=1e-4):
    flat = res.reshape(-1, res.shape[-1]).double()
    mu = flat.mean(0)
    c = flat - mu
    sigma = (c.T @ c) / max(len(flat) - 1, 1)
    d = sigma.shape[0]
    eye = torch.eye(d, dtype=sigma.dtype)
    sigma = (1 - shrinkage) * sigma + shrinkage * (torch.trace(sigma) / d) * eye
    return mu.float(), torch.linalg.inv(sigma + reg * eye).float()


def score_metrics(scores, labels, cov, defect_thresh=0.5) -> dict:
    normal, defect, within, image_scores = [], [], [], []
    for s, lab, c in zip(scores, labels, cov):
        k = min(3, s.size)
        image_scores.append(float(np.sort(s)[-k:].mean()))
        if lab == 0:
            normal.append(s)
            continue
        d, cl = s[c >= defect_thresh], s[c == 0.0]
        if d.size:
            defect.append(d)
        if d.size and cl.size:
            within.append(roc_auc_score(np.r_[np.zeros(len(cl)), np.ones(len(d))], np.r_[cl, d]))
    n_all, d_all = np.concatenate(normal), np.concatenate(defect)
    nm, dm = float(np.median(n_all)), float(np.median(d_all))
    return {
        "image_auroc": roc_auc_score(labels, image_scores),
        "across": roc_auc_score(np.r_[np.zeros(len(n_all)), np.ones(len(d_all))], np.r_[n_all, d_all]),
        "within": float(np.mean(within)) if within else float("nan"),
        "elevation": dm / nm if nm > 0 else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="P2: learned context reconstruction")
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["val", "heldout", "all"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--n-queries", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dump-queries", type=str, default=None)
    args = p.parse_args()

    cfg = load_config()
    cfg["dataset"]["category"] = args.category
    logger = get_logger("P2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or checkpoint_path(cfg, args.category)
    image_size, patch_size = cfg["vit"]["image_size"], cfg["vit"]["patch_size"]
    grid = image_size // patch_size

    train_paths = sorted(
        str(x) for x in (Path(cfg["dataset"]["root"]) / args.category / "train" / "good").glob("*.png")
    )
    probe = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                         image_size=image_size, patch_size=patch_size, synthetic_method=None)
    dataset = MVTecDataset(root=cfg["dataset"]["root"], category=args.category, split="test",
                           image_size=image_size, patch_size=patch_size, synthetic_method=None,
                           subset_indices=split_indices(len(probe), args.split))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = build_model(cfg, checkpoint, device)
    logger.info(f"caching features: {len(train_paths)} train, {len(dataset)} test")
    train_feats = cache_train_features(model, train_paths, image_size, device)
    test_feats, labels, cov = cache_test_features(model, loader, device, grid)
    del model
    torch.cuda.empty_cache()
    logger.info(f"train {tuple(train_feats.shape)}  test {tuple(test_feats.shape)}")

    # ── baseline: Mahalanobis on RAW features, same fitting procedure ──
    mu0, sinv0 = fit_gaussian(train_feats)
    flat = test_feats.reshape(-1, test_feats.shape[-1]) - mu0
    base_scores = torch.clamp((flat @ sinv0 * flat).sum(1), min=0).view(len(test_feats), -1).numpy()
    baseline = score_metrics(list(base_scores), labels, cov)

    # ── learned context reconstruction ──
    recon = QueryBottleneckReconstructor(
        feature_dim=train_feats.shape[-1], hidden=args.hidden,
        n_queries=args.n_queries, n_patches=train_feats.shape[1], n_layers=args.n_layers,
    ).to(device)
    n_params = sum(x.numel() for x in recon.parameters())
    logger.info(f"reconstructor: {n_params:,} params, {args.n_queries} queries "
                f"(bottleneck {args.n_queries}/{train_feats.shape[1]} patches)")

    t0 = time.time()
    mean, std, best_val = train(recon, train_feats, device, logger,
                                epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, noise=args.noise, patience=args.patience)
    logger.info(f"trained in {time.time() - t0:.0f}s, best val MSE {best_val:.4f}")

    train_res = residuals(recon, train_feats, mean, std, device)
    test_res = residuals(recon, test_feats, mean, std, device)

    mu_r, sinv_r = fit_gaussian(train_res)
    fr = test_res.reshape(-1, test_res.shape[-1]) - mu_r
    res_maha = torch.clamp((fr @ sinv_r * fr).sum(1), min=0).view(len(test_res), -1).numpy()
    res_norm = (test_res ** 2).sum(-1).numpy()

    learned_maha = score_metrics(list(res_maha), labels, cov)
    learned_norm = score_metrics(list(res_norm), labels, cov)

    print(f"\n{'=' * 88}")
    print(f"P2  LEARNED CONTEXT RECONSTRUCTION  —  {args.category}  (split={args.split})")
    print(f"{'=' * 88}")
    print(f"{'scorer':<38}{'image AUROC':>13}{'within':>10}{'across':>10}{'elevation':>12}")
    print("-" * 88)
    for label, r in (
        ("Mahalanobis on raw features (baseline)", baseline),
        ("Mahalanobis on context residual", learned_maha),
        ("residual norm", learned_norm),
    ):
        print(f"{label:<38}{r['image_auroc']:>13.4f}{r['within']:>10.4f}"
              f"{r['across']:>10.4f}{r['elevation']:>11.2f}x")
    print("-" * 88)
    d = learned_maha["within"] - baseline["within"]
    print(f"within-image separability change: {d:+.4f}  "
          f"({'MECHANISM CONFIRMED' if d > 0.03 else 'no meaningful gain'})")
    print(f"{'=' * 88}\n")

    if args.dump_queries:
        with torch.no_grad():
            q = recon.encode(((test_feats.to(device) - mean.to(device)) / std.to(device)))
        torch.save({"queries": q.cpu(), "labels": labels, "category": args.category},
                   args.dump_queries)
        logger.info(f"bottleneck queries -> {args.dump_queries}")


if __name__ == "__main__":
    main()
