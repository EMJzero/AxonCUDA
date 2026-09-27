#!/usr/bin/env bash
set -euo pipefail

# Assume '../../hgraphs/procure_hgraphs.sh' was used to create the "ispd98_16x" folder,
# inputs are read from there, results are stored here

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

DATA_DIR="$(cd -P "$SCRIPT_DIR/../../hgraphs/ispd98_16x" && pwd)"
TARGET_BIN="$(cd -P "$SCRIPT_DIR/.." && pwd)/hgraph_cpu.exe"
THREADS="${THREADS:-}" # OpenMP threads, overridable from the environment (default: all hardware threads)
TARGET_ARGS=(-rfr 16 -cnc 4 -dtc -v 0 -smh 0 ${THREADS:+-t "$THREADS"})
RESULTS_DIR="$SCRIPT_DIR/ispd98_16x/results_cnc4_rfr16"

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
  local outfile="$2"
  shift 2

  local rc=0

  echo "========================================"
  echo "Running ${label}"
  echo "========================================"

  if (( PROFILING )); then
    if ! (
      cd "$DATA_DIR"
      "$TARGET_BIN" "${TARGET_ARGS[@]}" "$@" -v 1 \
        |& tee "${RESULTS_DIR}/${outfile}.txt"
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
#make -C ../..
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