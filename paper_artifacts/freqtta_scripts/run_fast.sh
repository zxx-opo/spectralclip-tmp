#!/usr/bin/env bash
set -uo pipefail
cd /root/freqtta-hsi
export PYTHONPATH=/root/freqtta-hsi

EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"42 43 44"}
RES=/root/freqtta-hsi/results

run_if_missing() {
    local tag="$1" ; shift
    local json="$RES/${tag}.json"
    if [[ -s "$json" ]]; then
        echo "[SKIP] $tag"
        return
    fi
    echo "[RUN ] $tag"
    "$@" || echo "[FAIL] $tag (exit=$?)"
}

echo "=== Phase: cross-scene + TTA ==="
for SEED in $SEEDS; do
  for PAIR in S1_PaviaC2U S2_PaviaC_noise; do
    for M in famssm s2mamba spectralformer hybridsn; do
      run_if_missing "cross_${PAIR}_${M}_s${SEED}" \
        python3 src/run_experiment.py cross_scene_tta \
          --model $M --pair $PAIR --seed $SEED --epochs $EPOCHS
    done
  done
done

echo "=== Phase: ablation ==="
for SEED in $SEEDS; do
  run_if_missing "ablation_S1_PaviaC2U_s${SEED}" \
    python3 src/run_experiment.py ablation \
      --seed $SEED --epochs $EPOCHS
done

echo "ALL DONE: $(date)"
