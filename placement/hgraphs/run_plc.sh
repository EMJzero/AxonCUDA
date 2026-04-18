#!/usr/bin/env bash
set -euo pipefail

# Assume 'build_hgraphs.sh' was used to create the "part_snns" folder

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

DATA_DIR="$(cd -P "$SCRIPT_DIR/part_snns" && pwd)"
TARGET_BIN="$(cd -P "$SCRIPT_DIR/.." && pwd)/hplace_gpu.exe"
TARGET_ARGS=(-lpr 16 -fdi 32 -dtc -v 0 -mso 16 -thr 1) # can be later overriden per-run
RESULTS_DIR="$DATA_DIR/results_lpr16_fdi32"

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
  local filename="$2"
  shift 2

  local rc=0

  echo "========================================"
  echo "Running ${label}"
  echo "========================================"

  if (( PROFILING )); then
    if ! (
      cd "$DATA_DIR"
      "${NSYS_BASE_CMD[@]}" \
        -r "${filename}.snn" \
        --output="${RESULTS_DIR}/${filename}_profile" \
        "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" \
        |& tee "${RESULTS_DIR}/${filename}.txt"
    ); then
      rc=$?
    fi

    rm -f "${RESULTS_DIR}/${filename}_profile".{nsys-rep,sqlite}

  elif (( NSIGHT )); then
    if ! (
      cd "$DATA_DIR"
      "${NCU_BASE_CMD[@]}" \
        -r "${filename}.snn" \
        --log-file "${RESULTS_DIR}/${filename}.csv" \
        "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@"
    ); then
      rc=$?
    fi
  else
    if ! (
      cd "$DATA_DIR"
      "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" \
        -r "${filename}.snn" \
        |& tee "${RESULTS_DIR}/${filename}"
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
run_case_checked "8k"          "8k_model_part"
run_case_checked "64k"         "64k_model_part"
run_case_checked "256k"        "256k_model_part" -c loihi84
run_case_checked "1M"          "1M_model_part" -c loihi84
run_case_checked "16M"         "16M_model_part" -c loihi1024

# -------------------------
# Classic ANNs
# -------------------------
run_case_checked "LeNet"       "lenet_part"
run_case_checked "VGG11"       "vgg11_part" -c loihi84
run_case_checked "AlexNet"     "alexnet_part" -c loihi84
run_case_checked "MobileNet"   "mobilenet_part" -c loihi1024

# -------------------------
# SNNs
# -------------------------
run_case_checked "16k rand"    "16k_rand_part"
run_case_checked "64k rand"    "64k_rand_part"
run_case_checked "256k rand"   "256k_rand_part"
run_case_checked "Allen V1"    "allen_v1_part" -c loihi84

if (( FAILURES > 0 )); then
  echo "All runs completed, with ${FAILURES} failure(s)." >&2
  exit 1
fi

echo "All runs completed."