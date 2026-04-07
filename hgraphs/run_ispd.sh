#!/usr/bin/env bash
set -euo pipefail

# Assume 'procure_hgraphs.sh' was used to create the "ispd98_16x" folder

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

DATA_DIR="$(cd -P "$SCRIPT_DIR/ispd98_16x" && pwd)"
TARGET_BIN="$(cd -P "$SCRIPT_DIR/.." && pwd)/hgraph_gpu.exe"
TARGET_ARGS=(-rfr 16 -cnc 4 -dtc -v 0 -smh 0)
RESULTS_DIR="$DATA_DIR/results_rfr4_cnc16"

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
# ISPD 98 - 16x - k = 2
# -------------------------
run_case_checked "01-k2"        "ispd98_01_k2"        -r ISPD98_ibm01.hgr -k 2 0.03 -om 5
run_case_checked "02-k2"        "ispd98_02_k2"        -r ISPD98_ibm02.hgr -k 2 0.03 -om 5
run_case_checked "03-k2"        "ispd98_03_k2"        -r ISPD98_ibm03.hgr -k 2 0.03 -om 5
run_case_checked "04-k2"        "ispd98_04_k2"        -r ISPD98_ibm04.hgr -k 2 0.03 -om 8
run_case_checked "05-k2"        "ispd98_05_k2"        -r ISPD98_ibm05.hgr -k 2 0.03 -om 5
run_case_checked "06-k2"        "ispd98_06_k2"        -r ISPD98_ibm06.hgr -k 2 0.03 -om 5
run_case_checked "07-k2"        "ispd98_07_k2"        -r ISPD98_ibm07.hgr -k 2 0.03 -om 5
run_case_checked "08-k2"        "ispd98_08_k2"        -r ISPD98_ibm08.hgr -k 2 0.03 -om 5
run_case_checked "09-k2"        "ispd98_09_k2"        -r ISPD98_ibm09.hgr -k 2 0.03 -om 5
run_case_checked "10-k2"        "ispd98_10_k2"        -r ISPD98_ibm10.hgr -k 2 0.03 -om 8
run_case_checked "11-k2"        "ispd98_11_k2"        -r ISPD98_ibm11.hgr -k 2 0.03 -om 5
run_case_checked "12-k2"        "ispd98_12_k2"        -r ISPD98_ibm12.hgr -k 2 0.03 -om 5
run_case_checked "13-k2"        "ispd98_13_k2"        -r ISPD98_ibm13.hgr -k 2 0.03 -om 5
run_case_checked "14-k2"        "ispd98_14_k2"        -r ISPD98_ibm14.hgr -k 2 0.03 -om 6
run_case_checked "15-k2"        "ispd98_15_k2"        -r ISPD98_ibm15.hgr -k 2 0.03 -om 5
run_case_checked "16-k2"        "ispd98_16_k2"        -r ISPD98_ibm16.hgr -k 2 0.03 -om 5
run_case_checked "17-k2"        "ispd98_17_k2"        -r ISPD98_ibm17.hgr -k 2 0.03 -om 8
run_case_checked "18-k2"        "ispd98_18_k2"        -r ISPD98_ibm18.hgr -k 2 0.03 -om 8

# -------------------------
# ISPD 98 - 16x - k = 4
# -------------------------
run_case_checked "01-k4"        "ispd98_01_k4"        -r ISPD98_ibm01.hgr -k 4 0.03 -om 5
run_case_checked "02-k4"        "ispd98_02_k4"        -r ISPD98_ibm02.hgr -k 4 0.03 -om 5
run_case_checked "03-k4"        "ispd98_03_k4"        -r ISPD98_ibm03.hgr -k 4 0.03 -om 5
run_case_checked "04-k4"        "ispd98_04_k4"        -r ISPD98_ibm04.hgr -k 4 0.03 -om 8
run_case_checked "05-k4"        "ispd98_05_k4"        -r ISPD98_ibm05.hgr -k 4 0.03 -om 5
run_case_checked "06-k4"        "ispd98_06_k4"        -r ISPD98_ibm06.hgr -k 4 0.03 -om 5
run_case_checked "07-k4"        "ispd98_07_k4"        -r ISPD98_ibm07.hgr -k 4 0.03 -om 5
run_case_checked "08-k4"        "ispd98_08_k4"        -r ISPD98_ibm08.hgr -k 4 0.03 -om 5
run_case_checked "09-k4"        "ispd98_09_k4"        -r ISPD98_ibm09.hgr -k 4 0.03 -om 5
run_case_checked "10-k4"        "ispd98_10_k4"        -r ISPD98_ibm10.hgr -k 4 0.03 -om 8
run_case_checked "11-k4"        "ispd98_11_k4"        -r ISPD98_ibm11.hgr -k 4 0.03 -om 5
run_case_checked "12-k4"        "ispd98_12_k4"        -r ISPD98_ibm12.hgr -k 4 0.03 -om 5
run_case_checked "13-k4"        "ispd98_13_k4"        -r ISPD98_ibm13.hgr -k 4 0.03 -om 5
run_case_checked "14-k4"        "ispd98_14_k4"        -r ISPD98_ibm14.hgr -k 4 0.03 -om 6
run_case_checked "15-k4"        "ispd98_15_k4"        -r ISPD98_ibm15.hgr -k 4 0.03 -om 5
run_case_checked "16-k4"        "ispd98_16_k4"        -r ISPD98_ibm16.hgr -k 4 0.03 -om 5
run_case_checked "17-k4"        "ispd98_17_k4"        -r ISPD98_ibm17.hgr -k 4 0.03 -om 8
run_case_checked "18-k4"        "ispd98_18_k4"        -r ISPD98_ibm18.hgr -k 4 0.03 -om 8

if (( FAILURES > 0 )); then
  echo "All runs completed, with ${FAILURES} failure(s)." >&2
  exit 1
fi

echo "All runs completed."