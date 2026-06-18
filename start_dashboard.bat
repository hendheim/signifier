@echo off
chcp 65001 >nul
set PYTHONUTF8=1
REM ===========================================================
REM FaDeLive-Dashboard starten (Windows)
REM Doppelklick genuegt. Beenden: dieses Fenster schliessen.
REM ===========================================================
cd /d "%~dp0"
python -m streamlit run Willkommen.py
if errorlevel 1 (
    echo.
    echo Start fehlgeschlagen. Sind die Pakete installiert?
    echo   pip install -r requirements.txt
    pause
)
