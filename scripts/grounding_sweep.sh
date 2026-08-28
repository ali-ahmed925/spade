#!/bin/bash
# lambda sweep for the auxiliary query-grounding objective.
#
# Trains one checkpoint per lambda, evaluates detection on the HELD-OUT half and
# query localisation on the VAL half, and diffs every arm against the lambda=0
# control. lambda=0 generates no synthetic anomalies at all, so that arm is the
# exact pre-grounding baseline.
#
# The question this answers: do the query tokens become groundable
# (best-query pixel AUROC 0.63 -> 0.85+) WITHOUT losing image/pixel AUROC?
#
# Usage:  bash scripts/grounding_sweep.sh wood 0 0.05 0.1 0.5

set -u
# Python block-buffers stdout when piped, and grep buffers its own output, so a
# filtered training run can sit silent for minutes. Training output is streamed
# unfiltered (the tqdm bar IS the live progress); only the eval/D1 summaries are
# filtered, with --line-buffered so matches appear immediately.
export PYTHONUNBUFFERED=1
CATEGORY="${1:?usage: grounding_sweep.sh CATEGORY LAMBDA [LAMBDA...]}"
shift
LAMBDAS=("$@")
[ ${#LAMBDAS[@]} -eq 0 ] && LAMBDAS=(0 0.05 0.1 0.5)

OUT="logs/grounding_sweep/${CATEGORY}"
mkdir -p "$OUT"

sed -i "s/^  category: .*/  category: \"${CATEGORY}\"/" config/data.yaml

for LAM in "${LAMBDAS[@]}"; do
  TAG=$(echo "$LAM" | tr '.' 'p')
  echo "################ ${CATEGORY}  lambda=${LAM} ################"

  sed -i "s/^  grounding_weight: .*/  grounding_weight: ${LAM}        # set by grounding_sweep.sh/" config/train.yaml
  sed -i "s|^  checkpoint_dir: .*|  checkpoint_dir: \"./checkpoints_v2_lam${TAG}\"|" config/train.yaml

  python train.py 2>&1 | tee "${OUT}/train_lam${TAG}.log"

  CKPT="checkpoints_v2_lam${TAG}/${CATEGORY}/spade_best.pt"

  # detection: held-out, never used for any selection
  python eval.py --checkpoint "$CKPT" --split heldout 2>&1 \
    | tee "${OUT}/eval_lam${TAG}.log" \
    | grep --line-buffered -E "Image AUROC|Pixel AUROC|PRO |topk_mean|max "

  # query localisation: val half, compared against the lambda=0 control
  COMPARE=""
  [ -f "${OUT}/d1_lam0.json" ] && [ "$TAG" != "0" ] && COMPARE="--compare ${OUT}/d1_lam0.json"
  python scripts/diagnose_attention.py --category "$CATEGORY" --split val \
    --checkpoint "$CKPT" --json "${OUT}/d1_lam${TAG}.json" $COMPARE 2>&1 \
    | tee "${OUT}/d1_lam${TAG}.log" \
    | grep --line-buffered -E "TOTAL|saliency|best query|mean |verdict|DEGRADED|before|after"
done

# restore the committed defaults so the working tree is not left mutated
git checkout config/train.yaml 2>/dev/null || true

echo
echo "================ SWEEP SUMMARY  ${CATEGORY} ================"
printf "%-10s %12s %12s %10s %14s %10s\n" lambda image pixel PRO best-query q-std
for LAM in "${LAMBDAS[@]}"; do
  TAG=$(echo "$LAM" | tr '.' 'p')
  python - "$OUT" "$TAG" "$LAM" <<'PY'
import json, re, sys
out, tag, lam = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    ev = open(f"{out}/eval_lam{tag}.log").read()
    g = lambda p: (re.search(p, ev) or [None, "nan"])[1]
    d1 = json.load(open(f"{out}/d1_lam{tag}.json"))
    print(f"{lam:<10} {g(r'Image AUROC: ([0-9.]+)'):>12} {g(r'Pixel AUROC: ([0-9.]+)'):>12} "
          f"{g(r'PRO *: ([0-9.]+)'):>10} {d1['best_query_auroc']:>14.4f} {d1['query_std']:>10.4f}")
except Exception as exc:
    print(f"{lam:<10} (incomplete: {type(exc).__name__})")
PY
done
echo "==========================================================="
