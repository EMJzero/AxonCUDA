#!/usr/bin/env bash
set -euo pipefail

# Assume '../hgraphs/procure_hgraphs.sh' was used to create the "../hgraphs/snns" folder

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

DATA_DIR="$(cd -P "$SCRIPT_DIR/../../hgraphs/snns" && pwd)"
TARGET_BIN="$(cd -P "$SCRIPT_DIR/../.." && pwd)/hgraph_gpu.exe"
TARGET_ARGS=(-rfr 16 -cnc 4 -dtc -ipm -v 0) # can be later overriden per-run
PART_SNNS_DIR="$SCRIPT_DIR/part_snns"

FAILURES=0

# -------------------------
# Helper
# -------------------------
run_case() {
  local label="$1"
  local outfile="$2"
  shift 2

  local rc=0

  echo "========================================"
  echo "Partitioning ${label}"
  echo "========================================"

  if ! (
      cd "$DATA_DIR"
      "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" \
      -s "${PART_SNNS_DIR}/${outfile}_part.snn"
  ); then
      rc=$?
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
mkdir -p "$PART_SNNS_DIR"

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
  echo "All partitionings completed, with ${FAILURES} failure(s)." >&2
  exit 1
fi

echo "All partitionings completed."