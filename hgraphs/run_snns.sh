#!/usr/bin/env bash
set -euo pipefail

# Assume 'procure_hgraphs.sh' was used to create the SNNs folder
cd snns

# -------------------------
# Configuration
# -------------------------
#TARGET="../hgraph_gpu.exe"
TARGET="../../hgraph_gpu.exe -rfr 16 -cnc 4"
RESULTS_DIR="results_rfr4_cnc16"
#RESULTS_DIR="results_nsight"
#RESULTS_DIR="results_profile"
PROFILING=0
NSIGHT=0

NSYS_BASE_CMD="nsys profile --stats=true --force-overwrite=true"

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
NCU_BASE_CMD_="ncu \
--target-processes all \
--set none \
--metrics \
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
dram__throughput.avg.pct_of_peak_sustained_elapsed \
--csv"

# Instruction mix:
NCU_BASE_CMD="ncu \
--target-processes all \
--set none \
--metrics \
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
sm__sass_thread_inst_executed_op_misc_pred_on.sum \
--csv"

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
  local run_args="$2"
  local outfile="$3"

  echo "========================================"
  echo "Running ${label}"
  echo "========================================"

  if [[ $PROFILING -eq 1 ]]; then
    $NSYS_BASE_CMD --output="${RESULTS_DIR}/${outfile%.txt}_profile" $TARGET $run_args |& tee "${RESULTS_DIR}/${outfile}.txt"
    rm -f "${RESULTS_DIR}/${outfile%.txt}_profile".{nsys-rep,sqlite}
  elif [[ $NSIGHT -eq 1 ]]; then
    $NCU_BASE_CMD --log-file "${RESULTS_DIR}/${outfile}.csv" $TARGET $run_args
  else
    $TARGET $run_args |& tee "${RESULTS_DIR}/${outfile}"
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
run_case "8k"        "-r 8k_model_ordered_processed.snn"                      "8k_model"
run_case "64k"       "-r 64k_model_ordered_processed.snn"                     "64k_model"
run_case "256k"      "-r 256k_model_ordered_processed.snn -c loihi84"         "256k_model"
run_case "1M"        "-r 1M_model_ordered_processed.snn -c loihi84"           "1M_model"
run_case "16M"       "-r 16M_model_ordered_processed.snn -c loihi1024"        "16M_model"

# -------------------------
# Classic ANNs
# -------------------------
run_case "LeNet"     "-r lenet_cifar_ordered_processed.snn"                   "lenet"
run_case "VGG11"     "-r vgg11_cifar_model_ordered_processed.snn -c loihi84"  "vgg11"
run_case "AlexNet"   "-r alexnet_cifar_ordered_processed.snn -c loihi84"      "alexnet"
run_case "MobileNet" "-r mobilenet_imagenet_ordered_processed.snn -c loihi84" "mobilenet"

# -------------------------
# SNNs
# -------------------------
run_case "16k rand"  "-r 16384L_rand_reservoir.snn"                           "16k_rand"
run_case "64k rand"  "-r 65536L_rand_reservoir.snn"                           "64k_rand"
run_case "256k rand" "-r 262144L_rand_reservoir.snn"                          "256k_rand"
run_case "Allen V1"  "-r allen_v1_ordered_processed.snn -c loihi84"           "allen_v1"

echo "All runs completed."