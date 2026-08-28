#!/bin/bash
# lambda sweep for the auxiliary query-grounding objective.
#
# The question: do the query tokens become groundable (best-query pixel AUROC
# 0.63 -> 0.85+) WITHOUT losing image/pixel AUROC?
#
# Each arm is measured on TWO checkpoints, because they answer different things:
#
#   spade_last.pt  — fixed update budget, identical for every arm. This is the
#                    controlled comparison. Selecting per-arm would confound
#                    "what lambda did" with "which epoch each run peaked on",
#                    and on a category whose image AUROC saturates at epoch 1
#                    selection is close to arbitrary.
#   spade_best.pt  — val-selected, i.e. the number the method would report.
#
# Detection is always read from the HELD-OUT half, which nothing selects on.
# Query localisation is read from the val half.
#
# Usage:  bash scripts/grounding_sweep.sh wood 0 0.05 0.1 0.5

set -u
# Python block-buffers stdout when piped and grep buffers its own output, so a
# filtered training run can sit silent for minutes. Training streams unfiltered
# (the tqdm bar IS the live progress); summaries are line-buffered.
export PYTHONUNBUFFERED=1

CATEGORY="${1:?usage: grounding_sweep.sh CATEGORY LAMBDA [LAMBDA...]}"
shift
LAMBDAS=("$@")
[ ${#LAMBDAS[@]} -eq 0 ] && LAMBDAS=(0 0.05 0.1 0.5)

OUT="logs/grounding_sweep/${CATEGORY}"
mkdir -p "$OUT"

# The sweep rewrites two config files per arm. Restore them on ANY exit —
# including Ctrl+C — so an interrupted sweep never leaves the working tree
# mutated for the next git operation.
cp config/train.yaml "${OUT}/.train.yaml.orig"
cp config/data.yaml  "${OUT}/.data.yaml.orig"
restore() {
  cp "${OUT}/.train.yaml.orig" config/train.yaml
  cp "${OUT}/.data.yaml.orig"  config/data.yaml
  echo "[sweep] restored config/train.yaml and config/data.yaml"
}
trap restore EXIT INT TERM

sed -i "s/^  category: .*/  category: \"${CATEGORY}\"/" config/data.yaml

for LAM in "${LAMBDAS[@]}"; do
  TAG=$(echo "$LAM" | tr '.' 'p')
  echo "################ ${CATEGORY}  lambda=${LAM} ################"

  sed -i "s/^  grounding_weight: .*/  grounding_weight: ${LAM}        # set by grounding_sweep.sh/" config/train.yaml
  sed -i "s|^  checkpoint_dir: .*|  checkpoint_dir: \"./checkpoints_v2_lam${TAG}\"|" config/train.yaml

  python train.py 2>&1 | tee "${OUT}/train_lam${TAG}.log"

  for WHICH in last best; do
    CKPT="checkpoints_v2_lam${TAG}/${CATEGORY}/spade_${WHICH}.pt"
    [ -f "$CKPT" ] || { echo "[sweep] missing $CKPT — skipping"; continue; }

    echo "---- lambda=${LAM}  ${WHICH}  detection (held-out) ----"
    python eval.py --checkpoint "$CKPT" --split heldout 2>&1 \
      | tee "${OUT}/eval_${WHICH}_lam${TAG}.log" \
      | grep --line-buffered -E "Image AUROC|Pixel AUROC|PRO |topk_mean|max "
  done

  # Query localisation on the fixed-budget checkpoint, diffed against the
  # lambda=0 control at the same budget.
  CKPT="checkpoints_v2_lam${TAG}/${CATEGORY}/spade_last.pt"
  COMPARE=""
  [ -f "${OUT}/d1_last_lam0.json" ] && [ "$TAG" != "0" ] && COMPARE="--compare ${OUT}/d1_last_lam0.json"
  echo "---- lambda=${LAM}  last  query localisation (val) ----"
  python scripts/diagnose_attention.py --category "$CATEGORY" --split val \
    --checkpoint "$CKPT" --json "${OUT}/d1_last_lam${TAG}.json" $COMPARE 2>&1 \
    | tee "${OUT}/d1_last_lam${TAG}.log" \
    | grep --line-buffered -E "TOTAL|saliency|best query|mean |verdict|DEGRADED|before|after"
done

summary () {
  local which="$1" label="$2"
  echo
  echo "=========== ${CATEGORY}  —  ${label}  (detection: held-out) ==========="
  printf "%-8s %10s %10s %10s %12s %8s\n" lambda image pixel PRO best-query q-std
  for LAM in "${LAMBDAS[@]}"; do
    TAG=$(echo "$LAM" | tr '.' 'p')
    python - "$OUT" "$TAG" "$LAM" "$which" <<'PY'
import json, re, sys
out, tag, lam, which = sys.argv[1:5]
def num(pat, text):
    m = re.search(pat, text)
    return f"{float(m.group(1)):.4f}" if m else "  --  "
try:
    ev = open(f"{out}/eval_{which}_lam{tag}.log").read()
except OSError:
    print(f"{lam:<8} (no eval log)"); raise SystemExit
row = [num(r"Image AUROC: ([0-9.]+)", ev), num(r"Pixel AUROC: ([0-9.]+)", ev),
       num(r"PRO *: ([0-9.]+)", ev)]
try:
    d1 = json.load(open(f"{out}/d1_{which}_lam{tag}.json"))
    row += [f"{d1['best_query_auroc']:.4f}", f"{d1['query_std']:.4f}"]
except OSError:
    row += ["  --  ", "  --  "]
print(f"{lam:<8} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>12} {row[4]:>8}")
PY
  done
  echo "======================================================================"
}

summary last "FIXED BUDGET (spade_last) — the controlled comparison"
summary best "VAL-SELECTED (spade_best) — the reported number"
