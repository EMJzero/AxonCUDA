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
TARGET_BIN="$(cd -P "$SCRIPT_DIR/.." && pwd)/hplace_cpu.exe"
THREADS="${THREADS:-}" # OpenMP threads, overridable from the environment (default: all hardware threads)
TARGET_ARGS=(-lpr 16 -fdi 1024 -dtc -v 0 -mso 64 -bs 64 -sfc hilb ${THREADS:+-thr "$THREADS"}) # can be later overriden per-run
RESULTS_DIR="$DATA_DIR/results_lpr16_fdi1024_mso2"

PROFILING=0
FAILURES=0

# -------------------------
# Parse flags
# -------------------------
usage() {
  echo "Usage: $0 [-p]" >&2
  echo "  -p   enable profiling (per-phase times)" >&2
  exit 1
}

while getopts ":p" opt; do
  case $opt in
    p) PROFILING=1 ;;
    \?) usage ;;
  esac
done

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
      "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" -v 1 \
        -r "${filename}.snn" \
        |& tee "${RESULTS_DIR}/${filename}.txt"
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
run_case_checked "8k"          "8k_model_part" -c loihi
run_case_checked "64k"         "64k_model_part"
run_case_checked "256k"        "256k_model_part" -c loihi84
run_case_checked "1M"          "1M_model_part" -c loihi84
#run_case_checked "16M"         "16M_model_part" -c loihi1024

# -------------------------
# Classic ANNs
# -------------------------
run_case_checked "LeNet"       "lenet_part" -c loihi
run_case_checked "VGG11"       "vgg11_part" -c loihi84
run_case_checked "AlexNet"     "alexnet_part" -c loihi84
run_case_checked "MobileNet"   "mobilenet_part" -c loihi84

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