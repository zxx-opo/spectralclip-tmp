#!/usr/bin/env bash
set -uo pipefail
cd /root/freqtta-hsi
export PYTHONPATH=/root/freqtta-hsi
EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"42 43 44"}
RES=/root/freqtta-hsi/results

run_if_missing() {
    local tag="$1" ; shift
    [[ -s "$RES/${tag}.json" ]] && { echo "[SKIP] $tag"; return; }
    echo "[RUN ] $tag"
    "$@" || echo "[FAIL] $tag (exit=$?)"
}

echo "=== Phase: spatial-split + degradation cross-scene ==="
for SEED in $SEEDS; do
  for PAIR in P1_PaviaU_split P2_PaviaC_split P3_IndianPines_split P4_Salinas_split D1_PaviaU_noise D2_Salinas_noise; do
    for M in famssm s2mamba spectralformer hybridsn; do
      run_if_missing "cross_${PAIR}_${M}_s${SEED}" \
        python3 src/run_experiment.py cross_scene_tta \
          --model $M --pair $PAIR --seed $SEED --epochs $EPOCHS
    done
  done
done

echo "=== Phase: ablation ==="
for SEED in $SEEDS; do
  run_if_missing "ablation_P2_PaviaC_split_s${SEED}" \
    python3 src/run_experiment.py ablation \
      --seed $SEED --epochs $EPOCHS
done

echo "ALL DONE: $(date)"
