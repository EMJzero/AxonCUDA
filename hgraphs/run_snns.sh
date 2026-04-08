#!/usr/bin/env bash
set -euo pipefail

# Assume 'procure_hgraphs.sh' was used to create the "snns" folder

# -------------------------
# Configuration
# -------------------------
# retrieve the script's path
SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SOURCE" ]]; do
  SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

DATA_DIR="$(cd -P "$SCRIPT_DIR/snns" && pwd)"
TARGET_BIN="$(cd -P "$SCRIPT_DIR/.." && pwd)/hgraph_gpu.exe"
TARGET_ARGS=(-rfr 16 -cnc 4 -dtc -ipm -v 0) # can be later overriden per-run
RESULTS_DIR="$DATA_DIR/results_cnc4_rfr16"

PROFILING=0
NSIGHT=0
FAILURES=0

NSYS_BASE_CMD=(nsys profile --stats=true --force-overwrite=true)

# Nsight Compute Legend:
# sm__maximum_warps_per_active_cycle_pct -> Warp usage efficiency
# smsp__thread_inst_executed_per_inst_executed.ratio -> Threads per instruction
# smsp__sass_branch_targets_threads_divergent.avg -> Avg divergent threads
# smsp__sass_branch_targets_threads_divergent.sum -> Total divergent threads
# sm__cycles_elapsed.avg -> Avg SM cycles
# gpu__time_duration -> Kernel runtime
# smsp__sass_thread_inst_executed_op_integer_pred_on.sum -> Predicated int ops
# dram__bytes.sum -> Total DRAM bytes moved
# sm__throughput.avg.pct_of_peak_sustained_elapsed -> SM throughput vs peak (elapsed)
# sm__throughput.avg.pct_of_peak_sustained_active -> SM throughput vs peak (active)
# dram__throughput.avg.pct_of_peak_sustained_elapsed -> DRAM throughput vs peak

# Performance profiling:
NCU_BASE_CMD_=(
  ncu
  --target-processes all
  --set none
  --metrics
  sm__maximum_warps_per_active_cycle_pct,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
smsp__sass_branch_targets_threads_divergent.avg,\
smsp__sass_branch_targets_threads_divergent.sum,\
sm__cycles_elapsed.avg,\
gpu__time_duration,\
smsp__sass_thread_inst_executed_op_integer_pred_on.sum,\
dram__bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed
  --csv
)

# Instruction mix:
NCU_BASE_CMD=(
  ncu
  --target-processes all
  --set none
  --metrics
  sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum,\
sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
sm__sass_thread_inst_executed_op_control_pred_on.sum,\
sm__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum,\
sm__sass_thread_inst_executed_op_memory_pred_on.sum,\
sm__sass_thread_inst_executed_op_bit_pred_on.sum,\
sm__sass_thread_inst_executed_op_conversion_pred_on.sum,\
sm__sass_thread_inst_executed_op_misc_pred_on.sum
  --csv
)

# -------------------------
# Parse flags
# -------------------------
usage() {
  echo "Usage: $0 [-p | -n]" >&2
  echo "  -p   enable profiling" >&2
  echo "  -n   enable nsight compute" >&2
  exit 1
}

while getopts ":pn" opt; do
  case $opt in
    p) PROFILING=1 ;;
    n) NSIGHT=1 ;;
    \?) usage ;;
  esac
done

if (( PROFILING && NSIGHT )); then
  echo "Error: -p and -n cannot be used together." >&2
  usage
fi

# -------------------------
# Helper
# -------------------------
run_case() {
  local label="$1"
  local outfile="$2"
  shift 2

  local rc=0

  echo "========================================"
  echo "Running ${label}"
  echo "========================================"

  if (( PROFILING )); then
    if ! (
      cd "$DATA_DIR"
      "${NSYS_BASE_CMD[@]}" \
        --output="${RESULTS_DIR}/${outfile}_profile" \
        "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" \
        |& tee "${RESULTS_DIR}/${outfile}.txt"
    ); then
      rc=$?
    fi

    rm -f "${RESULTS_DIR}/${outfile}_profile".{nsys-rep,sqlite}

  elif (( NSIGHT )); then
    if ! (
      cd "$DATA_DIR"
      "${NCU_BASE_CMD[@]}" \
        --log-file "${RESULTS_DIR}/${outfile}.csv" \
        "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@"
    ); then
      rc=$?
    fi
  else
    if ! (
      cd "$DATA_DIR"
      "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" \
        |& tee "${RESULTS_DIR}/${outfile}"
    ); then
      rc=$?
    fi
  fi

  if (( rc == 0 )); then
    echo "Completed ${label}"
    return 0
  else
    echo "FAILED ${label} (exit code ${rc})" >&2
    return "$rc"
  fi
}

run_case_checked() {
  if ! run_case "$@"; then
    ((FAILURES+=1))
  fi
}

# -------------------------
# Build & setup
# -------------------------
#make -C ..
mkdir -p "$RESULTS_DIR"

# -------------------------
# Custom ANNs
# -------------------------
run_case_checked "8k"          "8k_model"          -r 8k_model_ordered_processed.snn -smh 0
run_case_checked "64k"         "64k_model"         -r 64k_model_ordered_processed.snn -smh 0
run_case_checked "256k"        "256k_model"        -r 256k_model_ordered_processed.snn -c loihi84 -smh 0
run_case_checked "1M"          "1M_model"          -r 1M_model_ordered_processed.snn -c loihi84
run_case_checked "16M"         "16M_model"         -r 16M_model_ordered_processed.snn -c loihi1024 -smh 12

# -------------------------
# Classic ANNs
# -------------------------
run_case_checked "LeNet"       "lenet"             -r lenet_cifar_ordered_processed.snn -smh 0
run_case_checked "VGG11"       "vgg11"             -r vgg11_cifar_model_ordered_processed.snn -c loihi84
run_case_checked "AlexNet"     "alexnet"           -r alexnet_cifar_ordered_processed.snn -c loihi84
run_case_checked "MobileNet"   "mobilenet"         -r mobilenet_imagenet_ordered_processed.snn -c loihi1024 -smh 8

# -------------------------
# SNNs
# -------------------------
run_case_checked "16k rand"    "16k_rand"          -r 16384L_rand_reservoir.snn -smh 0
run_case_checked "64k rand"    "64k_rand"          -r 65536L_rand_reservoir.snn -smh 0
run_case_checked "256k rand"   "256k_rand"         -r 262144L_rand_reservoir.snn -smh 0
run_case_checked "Allen V1"    "allen_v1"          -r allen_v1_ordered_processed.snn -c loihi84 -cnc 16

if (( FAILURES > 0 )); then
  echo "All runs completed, with ${FAILURES} failure(s)." >&2
  exit 1
fi

echo "All runs completed."