# Gradient & Dead-Path Audit

Branch: `fix/gradient-correctness` (off `44626a4`, `phone-fixes`)
Scope: **correctness only.** No performance tuning, no HPA redesign.

The question this branch answers: *does every component we call "trainable"
actually learn, and does every loss term actually provide signal?*

Method: trace the real computational graph — run forward + backward and inspect
`p.grad` for every parameter, perturb each module's weights and measure whether
the output moves, and measure each score term's share of the total. Comments and
docstrings were treated as claims to verify, not as evidence.

Everything below is reproducible with:

```bash
pytest tests/test_gradient_flow.py -q          # 16 regression tests, no BLIP-2 needed
```

---

## Before

Audit of the pipeline exactly as it was on `phone-fixes`:

```
module                        params     max|grad|  statuses
------------------------------------------------------------------------
projection                         4     0.000e+00  DEAD:4
qformer                           11     6.788e-05  OK:11
query_patch_attn                   2     3.108e-05  OK:2
vision_encoder                     3     0.000e+00  FROZEN:3

!! 4 parameter tensor(s) trainable but receiving NO signal:
   - projection.proj.0.weight (DEAD)   ... and 3 more

OUTPUT SENSITIVITY (relative output change when weights perturbed)
qformer                         1.152e-07
query_patch_attn                8.993e-08
projection                      0.000e+00  <-- NO EFFECT ON OUTPUT
```

Read that second block carefully: perturbing every weight in the Q-Former moved
the patch scores by **0.00001%**. The module received gradients and was
"training" — it simply could not affect the answer.

## After

```
module                        params     max|grad|  statuses
------------------------------------------------------------------------
projection                         4     0.000e+00  FROZEN:4
qformer                           11     3.357e-01  OK:11
query_patch_attn                   2     1.482e-01  OK:2
vision_encoder                     3     0.000e+00  FROZEN:3

OK: every trainable parameter received a non-zero gradient.

OUTPUT SENSITIVITY
qformer                         6.588e-03      (was 1.152e-07 — 57,000x)
query_patch_attn                2.630e-03      (was 8.993e-08 — 29,000x)

SCORE COMPOSITION (share of total magnitude, config alpha/beta/gamma = 0.25/0.65/0.10)
spatial_mahalanobis           64.94%
attention                     24.97%
frequency                      9.99%
cross                          0.10%
```

The configured weights now produce the shares they claim. Previously the same
config produced roughly `1e-5 / 1.0 / 1e-4`.

---

## Findings and fixes

### 1. `LLMProjection` was never trained — CONFIRMED, FIXED

`visual_tokens` is produced in `SPADE.forward` and appears in **no loss** —
`grep` finds it in neither `train.py`, `eval.py`, nor `losses/`. Its four
parameter tensors got `grad=None` on every step of every run, yet were counted
in the "Trainable params" log line and written into every checkpoint.

**Fix** (`models/spade.py`): `projection_trainable` flag, default `False`, which
sets `requires_grad=False`. The parameter count now tells the truth and the
optimizer is not handed tensors it can never update. Set it to `True` only once
a loss actually supervises the text path.

*Consequence for the paper:* dropping the text module costs nothing, because it
was never trained. The weights in the released checkpoints are random init.

### 2. `PseudoAnomalyLoss` had an algebraically zero gradient — CONFIRMED, FIXED

```python
perturbed_scores = scores + self.epsilon * torch.randn_like(scores)
loss = F.relu(perturbed_scores - scores).mean()      # == relu(eps * noise)
```

`scores` cancels. The result is a random constant w.r.t. every parameter:

```
PseudoAnomalyLoss value: 0.0039
grad wrt scores -> all zero? True | max|g| = 0.0
```

`use_pseudo: true` was set in `config/train.yaml`, so this "contrastive
component" ran for every batch, contributing noise to the logged loss and
nothing to learning.

**Fix**: the perturbation is applied where the score actually comes from — the
patch embeddings — and the perturbed patch is re-scored via the new
`SPADE._score_perturbed`. The loss is now a margin: a perturbed patch must score
at least `pseudo_margin` above the clean patch it came from. Pinned by
`test_pseudo_loss_gradient_is_nonzero` and by
`test_old_pseudo_loss_formulation_would_have_zero_gradient`, which asserts the
old formulation's gradient is exactly zero so the bug cannot silently return.

### 3. Score streams differed by ~5 orders of magnitude — CONFIRMED, FIXED

From a real run's logs:

```
attn_contrib  =   0.003236   (alpha=0.7)
mahal_contrib = 318.102222   (beta=0.1)
```

The frozen ViT runs under `@torch.no_grad()`, so the Mahalanobis (0.65) and
frequency (0.10) streams have **no gradient path at all**. The only
gradient-bearing term was attention — and it was ~1e-5 of the score. Even a
perfect gradient could not have moved the output.

**Fix**: `normalize_streams` (default on) divides each stream by **one global
constant** per stream, EMA-estimated during training and held fixed at
inference. This is deliberately *not* per-image normalization — a per-image
divisor would destroy global calibration exactly as `EVALUATION_FIX.md` warns.
`normalize_streams: false` reproduces the legacy arithmetic exactly.

### 4. Attention importance had a structurally constant patch-mean — CONFIRMED, FIXED

`attention_importance = softmax(logits).sum(dim=queries)`. Each query's softmax
sums to 1 over patches, so the total is *always* exactly `num_queries` and the
patch-mean is the constant `num_queries / num_patches`:

```
softmax_sum  patch-mean per image: [0.015625, 0.015625, 0.015625]   # = 4/256, always
logit_mean   patch-mean per image: [0.009858, 0.020694, -0.005967]
```

So the loss's `mean()` term was mathematically blind to these weights, and the
`var()` term could only act on the *shape* — which a variance-minimizing
objective flattens toward uniform, i.e. erases.

**Fix**: `attention_aggregation` on `QueryPatchAttention` — `logit_mean`
(default; unnormalized, responds to the weights) or `softmax_sum` (legacy,
retained for reproduction).

### 5. Checkpoint save used an allowlist that silently dropped state — FIXED

`train.py` saved only keys matching six hard-coded prefixes. Any state added
later — such as the new stream-scale buffers — would have been dropped without
warning, producing checkpoints unable to reproduce their own scores. Replaced
with a denylist: save everything except `vision_encoder.` (restored from the
BLIP-2 download). Pinned by `test_checkpoint_keeps_stream_scale_buffers`.

### 6. A discarded covariance was computed every update — FIXED

`SPADE.forward` called `normal_stats_tracker.get_statistics()`, which builds a
full `D x D` covariance (1408x1408 from 10,000 patches), then **ignored both
return values** and re-stacked the buffer so `update_statistics()` could compute
them again. Replaced with an emptiness check. Pure waste, twice per update, on
both the spatial and frequency trackers.

### 7. Fail-loud self-check added

`train.py` now runs the audit on the first optimizer step and **raises** if any
trainable parameter received no gradient. A silent dead path is how finding #1
survived an entire project.

---

## Verified but NOT fixed (deliberate)

| Issue | Why not |
|---|---|
| **HPA's output is discarded.** `refined_patches` is unused and `selected_indices` appears only in a debug log; only the refined queries survive, feeding the attention term. | Out of scope by instruction — HPA redesign is phase 2. The audit tooling now makes its contribution measurable. |
| **`var_weight` term is still degenerate.** It can only act on the attention stream, and minimizing variance flattens it. | Fixing it means choosing a new objective — a design decision for phase 2, not a correctness repair. Documented here so it is not mistaken for working. |
| **`PatchAnomalyHead` (`models/heads.py`) is never instantiated.** | Dead module, not a dead *learning path* — no parameters exist in the model. Left in place rather than deleted. |
| **`Blip2QFormerWrapper.forward` is never called.** `SPADE` calls the inner HF module directly. | Dead code, not a correctness bug. Noted as a maintenance hazard. |
| **`freq_features.reshape(B, 256, ...)` hard-codes 256 patches.** | Correct for 224/14 but silently wrong for any other image size. Flagged; not exercised by current configs. |

---

## Effect on existing checkpoints

The released checkpoints (`checkpoints/<category>/spade_best.pt`) were trained
under the old arithmetic. They still load — the scale buffers default to 1.0
when absent, so `normalize_streams` is a no-op for them — but
`attention_aggregation` now defaults to `logit_mean`, which changes their
attention term. Since that term was ~1e-5 of their score, the effect is
negligible, but to reproduce baseline numbers **exactly**, set:

```yaml
scoring:
  normalize_streams: false
  attention_aggregation: "softmax_sum"
```

Baseline results on `phone-fixes` are untouched: wood 0.9947/0.9077,
leather 0.9963/0.9722, carpet 0.9980/0.9688, grid 1.0000/0.9655.

---

## What this branch does and does not claim

It claims: every parameter that reports as trainable now receives a non-zero
gradient, every loss term has a real derivative, the configured score weights
are the weights actually applied, and each of these is pinned by a test.

It does **not** claim better anomaly detection. The dominant signal is still
Mahalanobis distance over frozen ViT-G features, which no gradient touches. What
changed is that the learnable part is now *capable* of mattering — a
precondition for phase 2, not a result.
