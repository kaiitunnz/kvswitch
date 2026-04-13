#!/usr/bin/env bash
# Run paper experiments as defined in tmp/docs/EXPERIMENTAL_PLAN.md.
#
# Usage:
#   bash exp/run_exp.sh              # run all experiments
#   bash exp/run_exp.sh 1            # run experiment 1 only
#   bash exp/run_exp.sh 2 3a 4b      # run specific experiments
#
# Experiment IDs:
#   1   Microbenchmark: routing overhead
#   2   End-to-end: rate sweep
#   3a  Ablation: ECMP vs pinning
#   3b  Ablation: warm-up impact
#   4a  Sensitivity: prefix sharing ratio
#   4b  Sensitivity: KV cache capacity
#   4c  Sensitivity: number of workers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="bash $SCRIPT_DIR/run_eval.sh"
RESULTS_BASE="${RESULTS_BASE:-results/exp}"

# --- Throttle presets ---
MEDIUM=(--coalesce-interval-s 0.5 --load-update-interval-ms 100 --load-update-delta 100)
SLOW=(--coalesce-interval-s 2.0 --load-update-interval-ms 500 --load-update-delta 500)

# --- Common fixed parameters ---
COMMON=(--delay 0.5ms --admission-threshold 2 --seed 42)
ALL4="l4_ecmp,l7_rr,l7_pa,kvswitch"

# --- Constrained resource parameters ---
CONSTRAINED=(--max-num-seqs 4 --max-num-batched-tokens 2048)

# --- Helpers ---
LOGS_DIR="${LOGS_DIR:-results/logs}"

run() {
    local name=$1; shift
    local outdir="$RESULTS_BASE/$name"
    local logfile="$LOGS_DIR/$(echo "$name" | tr '/' '_').log"
    mkdir -p "$(dirname "$logfile")"
    echo ""
    echo "============================================================"
    echo "  Experiment: $name"
    echo "  Output:     $outdir"
    echo "  Log:        $logfile"
    echo "============================================================"
    $EVAL --output-dir "$outdir" "$@" 2>&1 | tee "$logfile"
}

# --- Experiment definitions ---
exp_1() {
    echo ">>> 1. Microbenchmark: routing overhead"
    run "1_microbenchmark" \
        --baselines "$ALL4" \
        --num-requests 200 --request-rate 10 \
        --warmup-per-group 0 \
        "${COMMON[@]}" "${MEDIUM[@]}"
}

exp_2() {
    echo ">>> 2. End-to-end: rate sweep"
    for rate in 10 50 100 200; do
        run "2_e2e/rate_${rate}" \
            --baselines "$ALL4" \
            --num-requests 500 --request-rate "$rate" \
            --warmup-per-group 20 \
            "${COMMON[@]}" "${MEDIUM[@]}"
    done
}

exp_3a() {
    echo ">>> 3a. Ablation: ECMP vs pinning"
    run "3a_ecmp_ablation/pin" \
        --baselines kvswitch \
        --num-requests 500 --request-rate 200 \
        --warmup-per-group 20 \
        --no-per-prefix-ecmp \
        "${COMMON[@]}" "${MEDIUM[@]}"
}

exp_3b() {
    echo ">>> 3b. Ablation: warm-up impact"
    run "3b_warmup_ablation/no_warmup" \
        --baselines "$ALL4" \
        --num-requests 500 --request-rate 200 \
        --warmup-per-group 0 \
        "${COMMON[@]}" "${MEDIUM[@]}"
}

exp_4a() {
    echo ">>> 4a. Sensitivity: prefix sharing ratio"
    for ratio in 0.2 0.4 0.6 0.8; do
        run "4a_prefix_sharing/ratio_${ratio}" \
            --baselines "$ALL4" \
            --num-requests 500 --request-rate 10 \
            --prefix-sharing-ratio "$ratio" \
            --warmup-per-group 0 \
            "${COMMON[@]}" "${MEDIUM[@]}"
    done
}

exp_4b() {
    echo ">>> 4b. Sensitivity: KV cache capacity"
    for cap in 4096 8192 16384 0; do
        local label=$cap
        local batched=8192
        if [ "$cap" -eq 0 ]; then
            label="unlimited"
        elif [ "$cap" -lt "$batched" ]; then
            batched=$cap
        fi
        run "4b_kv_capacity/cap_${label}" \
            --baselines "$ALL4" \
            --num-requests 500 --request-rate 200 \
            --warmup-per-group 20 \
            --kv-cache-capacity "$cap" \
            --max-num-batched-tokens "$batched" \
            "${COMMON[@]}" "${MEDIUM[@]}"
    done
}

exp_4c() {
    echo ">>> 4c. Sensitivity: number of workers"
    for nw in 4 8 12 16; do
        run "4c_workers/workers_$((nw * 2))" \
            --baselines "$ALL4" \
            --num-requests 500 --request-rate 200 \
            --warmup-per-group 20 \
            --n-worker-leaves 2 --workers-per-leaf "$nw" \
            "${COMMON[@]}" "${MEDIUM[@]}"
    done
}

# --- Main ---
EXPERIMENTS=("${@:-1 2 3a 3b 4a 4b 4c}")
if [ $# -eq 0 ]; then
    EXPERIMENTS=(1 2 3a 3b 4a 4b 4c)
fi

echo "Running experiments: ${EXPERIMENTS[*]}"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    case "$exp" in
        1)  exp_1  ;;
        2)  exp_2  ;;
        3a) exp_3a ;;
        3b) exp_3b ;;
        4a) exp_4a ;;
        4b) exp_4b ;;
        4c) exp_4c ;;
        *)  echo "Unknown experiment: $exp"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  All requested experiments complete."
echo "  Results in: $RESULTS_BASE/"
echo "============================================================"
