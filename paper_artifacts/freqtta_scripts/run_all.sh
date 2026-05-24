#!/usr/bin/env bash
# Master driver for FAM-SSM experiments.
# Restart-safe: each run writes JSON to results/; existing results are skipped.

set -uo pipefail
cd /root/freqtta-hsi
export PYTHONPATH=/root/freqtta-hsi

EPOCHS=${EPOCHS:-60}
SEEDS=${SEEDS:-"42 43 44"}
RES=/root/freqtta-hsi/results

run_if_missing() {
    local tag="$1" ; shift
    local json="$RES/${tag}.json"
    if [[ -s "$json" ]]; then
        echo "[SKIP] $tag already done"
        return
    fi
    echo "[RUN ] $tag"
    "$@" || echo "[FAIL] $tag (exit=$?)"
}

# in-domain
for SEED in $SEEDS; do
  for DS in PaviaU PaviaC IndianPines Salinas; do
    for M in famssm s2mamba spectralformer hybridsn; do
      run_if_missing "indomain_${DS}_${M}_s${SEED}" \
        python3 src/run_experiment.py in_domain \
          --model $M --dataset $DS --seed $SEED --epochs $EPOCHS
    done
  done
done

# cross-scene + TTA
for SEED in $SEEDS; do
  for PAIR in S1_PaviaC2U S2_PaviaC_noise; do
    for M in famssm s2mamba spectralformer hybridsn; do
      run_if_missing "cross_${PAIR}_${M}_s${SEED}" \
        python3 src/run_experiment.py cross_scene_tta \
          --model $M --pair $PAIR --seed $SEED --epochs $EPOCHS
    done
  done
done

# ablation
for SEED in $SEEDS; do
  run_if_missing "ablation_S1_PaviaC2U_s${SEED}" \
    python3 src/run_experiment.py ablation \
      --seed $SEED --epochs $EPOCHS
done

echo "=============================="
echo "ALL EXPERIMENTS DONE: $(date)"
echo "=============================="