#!/bin/sh
set -eu
cd "$(dirname "$0")"
REPRO_PYTHON="${REPRO_PYTHON:-python3}"
"$REPRO_PYTHON" scripts/reproduce_core.py
"$REPRO_PYTHON" scripts/reproduce_validation.py
"$REPRO_PYTHON" scripts/verify_results.py
"$REPRO_PYTHON" scripts/make_figures.py
