#!/bin/bash
# Anti-collapse weight sweep.
#
# The question: is there a weight that holds effective rank up across epochs
# WITHOUT the regulariser dominating detection? Both failure directions are
# real, so a single value cannot be assumed.
#
#   weight 0     exact control -- no AntiCollapseLoss is constructed at all
#   too weak     rank still falls, val AUROC still turns over after epoch 2
#   too strong   rank holds but val AUROC is flat or worse; the model is
#                optimising for spread instead of for anomalies
#
# Detection is read from the HELD-OUT half only at the end, once. Weight
# selection happens on VAL.
#
# Usage:  bash scripts/collapse_sweep.sh screw 0 0.5 1 5

set -u
export PYTHONUNBUFFERED=1

CATEGORY="${1:?usage: collapse_sweep.sh CATEGORY WEIGHT [WEIGHT...]}"
shift
WEIGHTS=("$@")
[ ${#WEIGHTS[@]} -eq 0 ] && WEIGHTS=(0 0.5 1 5)

OUT="logs/collapse_sweep/${CATEGORY}"
mkdir -p "$OUT"

cp config/train.yaml "${OUT}/.train.yaml.orig"
cp config/data.yaml  "${OUT}/.data.yaml.orig"
restore() {
  cp "${OUT}/.train.yaml.orig" config/train.yaml
  cp "${OUT}/.data.yaml.orig"  config/data.yaml
  echo "[sweep] restored config"
}
trap restore EXIT INT TERM

sed -i "s/^  category: .*/  category: \"${CATEGORY}\"/" config/data.yaml

for W in "${WEIGHTS[@]}"; do
  TAG=$(echo "$W" | tr '.' 'p')
  # VICReg's 25:1 ratio between the two terms is held fixed; only the scale moves
  COV=$(python -c "print(f'{float($W) * 0.04:.6g}')")
  echo "################ ${CATEGORY}  collapse weight ${W} (cov ${COV}) ################"

  sed -i "s/^  collapse_variance_weight: .*/  collapse_variance_weight: ${W}/" config/train.yaml
  sed -i "s/^  collapse_covariance_weight: .*/  collapse_covariance_weight: ${COV}/" config/train.yaml
  sed -i "s|^  checkpoint_dir: .*|  checkpoint_dir: \"./checkpoints_collapse_${TAG}\"|" config/train.yaml

  python train.py 2>&1 | tee "${OUT}/train_${TAG}.log"

  for SPLIT in val heldout; do
    CKPT="checkpoints_collapse_${TAG}/${CATEGORY}/spade_best.pt"
    [ -f "$CKPT" ] || { echo "[sweep] missing $CKPT"; continue; }
    python eval.py --checkpoint "$CKPT" --split "$SPLIT" 2>&1 \
      | tee "${OUT}/eval_${SPLIT}_${TAG}.log" \
      | grep --line-buffered -E "Image AUROC|Pixel AUROC|PRO |local_knn|contextual|frequency|TOTAL"
  done
done

echo
echo "======== ${CATEGORY}: does rank hold, and does detection survive? ========"
printf "%-8s %10s %10s %12s %12s\n" weight rank_ep1 rank_last val_image heldout_image
for W in "${WEIGHTS[@]}"; do
  TAG=$(echo "$W" | tr '.' 'p')
  python - "$OUT" "$TAG" "$W" <<'PY'
import re, sys
out, tag, w = sys.argv[1:4]
try:
    log = open(f"{out}/train_{tag}.log").read()
except OSError:
    print(f"{w:<8} (no log)"); raise SystemExit
ranks = re.findall(r"eff\.rank ([0-9.]+)/", log)
def num(path, pat):
    try:
        m = re.search(pat, open(path).read())
        return f"{float(m.group(1)):.4f}" if m else "  --  "
    except OSError:
        return "  --  "
print(f"{w:<8} {(ranks[0] if ranks else '--'):>10} {(ranks[-1] if ranks else '--'):>10} "
      f"{num(f'{out}/eval_val_{tag}.log', r'Image AUROC: ([0-9.]+)'):>12} "
      f"{num(f'{out}/eval_heldout_{tag}.log', r'Image AUROC: ([0-9.]+)'):>12}")
PY
done
echo "=========================================================================="
