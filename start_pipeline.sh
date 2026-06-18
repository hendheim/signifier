#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
# ===========================================================
# FaDeLive-Pipeline starten (macOS / Linux)
# Verarbeitet den Korpus (Schritte s01-s07) ueber run_pipeline.py.
# NICHT ueber den CLI-/Konsolen-Einstiegspunkt des Pakets.
#
# Aufruf (Beispiele):
#   ./start_pipeline.sh
#   ./start_pipeline.sh --steps 1 2 3
#   ./start_pipeline.sh --config config/fadelive_v2.toml
# ===========================================================
cd "$(dirname "$0")"

PROJECT_ROOT="$(pwd)"
CONFIG="config/fadelive_v3.toml"

python3 -u run_pipeline.py --project-root "$PROJECT_ROOT" --config "$CONFIG" "$@" || {
    echo
    echo "Pipeline fehlgeschlagen. Sind die Pakete installiert und die Pfade in der TOML korrekt?"
    echo "  pip install -r requirements.txt"
}
