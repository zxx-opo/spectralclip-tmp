#!/usr/bin/env bash
set -uo pipefail
cd /root/spectralclip-tmp
export PYTHONPATH=/root/spectralclip-tmp
RES=/root/spectralclip-tmp/results
CKPT=/root/spectralclip-tmp/checkpoints
SEEDS="${SEEDS:-42 43 44}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-60}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-40}"

run_if_missing() {
  local tag="$1"; shift
  [[ -s "$RES/${tag}.json" ]] && { echo "[SKIP] $tag"; return; }
  echo "[RUN ] $tag"
  "$@" || echo "[FAIL] $tag (exit=$?)"
}

# Phase 1: pretrain per seed
for SEED in $SEEDS; do
  run_if_missing "pretrain_s${SEED}" \
    python3 src/run_experiment.py pretrain --seed $SEED --epochs $PRETRAIN_EPOCHS --per_class 200
done

# Phase 2: closed-set finetune (with pretrained init)
for SEED in $SEEDS; do
  for DS in PaviaU PaviaC IndianPines Salinas; do
    run_if_missing "finetune_${DS}_s${SEED}" \
      python3 src/run_experiment.py finetune --dataset $DS --seed $SEED --epochs $FINETUNE_EPOCHS --pretrain_ckpt $CKPT/pretrain_s${SEED}.pt
  done
done

# Phase 3: zero-shot evaluation
for SEED in $SEEDS; do
  for DS in PaviaU PaviaC IndianPines Salinas; do
    run_if_missing "zs_${DS}_s${SEED}" \
      python3 src/run_experiment.py zero_shot --dataset $DS --seed $SEED --ckpt $CKPT/pretrain_s${SEED}.pt
  done
done

# Phase 4: few-shot
for SEED in $SEEDS; do
  for DS in PaviaU PaviaC IndianPines Salinas; do
    for K in 1 5