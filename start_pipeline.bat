@echo off
chcp 65001 >nul
set PYTHONUTF8=1
REM ===========================================================
REM FaDeLive-Pipeline starten (Windows)
REM Verarbeitet den Korpus (Schritte s01-s07) ueber run_pipeline.py.
REM NICHT ueber den CLI-/Konsolen-Einstiegspunkt des Pakets.
REM
REM Aufruf (Beispiele):
REM   start_pipeline.bat
REM   start_pipeline.bat --steps 1 2 3
REM   start_pipeline.bat --config config\fadelive_v2.toml
REM ===========================================================
cd /d "%~dp0"

REM Projektordner = dieser Ordner. Bei abweichender Lage anpassen:
set PROJECT_ROOT=%~dp0
set CONFIG=config\fadelive_v3.toml

python -u run_pipeline.py --project-root "%PROJECT_ROOT%" --config "%CONFIG%" %*
if errorlevel 1 (
    echo.
    echo Pipeline fehlgeschlagen. Sind die Pakete installiert und die Pfade in der TOML korrekt?
    echo   pip install -r requirements.txt
    pause
)
