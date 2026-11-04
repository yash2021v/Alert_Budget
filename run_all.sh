#!/usr/bin/env bash
# Run the whole AlertBudget pipeline: data -> panel -> split -> features -> models -> certification -> experiments -> figures.
#   ./run_all.sh --quick   15% account sample, 100 Monte Carlo draws, writes to results_quick/ and data/processed_quick/ (minutes)
#   ./run_all.sh           full data, writes to results/ and data/processed/
# Raw data: set DATA_DIR to the folder holding the seven dataset files (default: <repo>/data/raw). Activate the virtualenv first
# (or set PYTHON=/path/to/python). Frozen model files are regenerated (fixed seeds) only when they are missing.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
if [[ "${1:-}" == "--quick" ]]; then export ALERTBUDGET_MODE=quick; else export ALERTBUDGET_MODE=full; fi
RES=$($PY -c "from src import config as C; print(C.RESULTS)")
PROC=$($PY -c "from src import config as C; print(C.PROCESSED)")
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
run() { step "$*"; $PY -m "src.$@"; }
START=$(date +%s)

echo "mode=$ALERTBUDGET_MODE results=$RES processed=$PROC"
if [[ $ALERTBUDGET_MODE == full ]]; then step "day0 data checks"; $PY src/day0_check.py > /dev/null; fi
run data
run panel
run split
run features

# frozen entity-level scorer (LightGBM): regenerated only if missing
if [[ ! -f "$RES/lgbm_frozen.joblib" || ! -f "$PROC/scores.parquet" ]]; then run train; fi
run add_margin
run recompute_day1_metrics

# frozen transaction-level scorer
if [[ ! -f "$RES/lgbm_txn_frozen.joblib" || ! -f "$PROC/txn_scores.parquet" ]]; then run transaction_model; fi
run headline_scorer

# validity gate first, then the certified curves
step "validity gate"
if ! $PY -m src.validity_gate; then
  if [[ $ALERTBUDGET_MODE == quick ]]; then echo "NOTE: gate FAIL tolerated in quick mode (100 draws, noisy references)"; else exit 1; fi
fi
run run_budget_curve
run run_budget_curve_fraction
run shift_diagnostic

# experiments
run run_h1
step "H1 completion"; $PY -m src.h1_complete
step "H1 primary scorer (post-hoc)"; $PY -m src.run_h1 --primary; $PY -m src.h1_complete --primary
run run_h3
run elliptic
run elliptic_spike
run h4_elliptic
step "H2 synthetic"; $PY -m src.h2_synthetic; $PY -m src.h2_synthetic --extended
run ibm_delay_check

echo; echo "DONE in $(( $(date +%s) - START )) s. Results: $RES  Figures: $RES/figures"
