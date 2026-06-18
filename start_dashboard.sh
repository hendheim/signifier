#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
# ===========================================================
# FaDeLive-Dashboard starten (macOS / Linux)
# Beenden mit Strg+C
# ===========================================================
cd "$(dirname "$0")"
python3 -m streamlit run Willkommen.py || {
    echo
    echo "Start fehlgeschlagen. Sind die Pakete installiert?"
    echo "  pip install -r requirements.txt"
}
