#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.pipeline_runner
=============================

UI-freie Logik für die Dashboard-Seite "Korpus verarbeiten".

Startet die bestehende Pipeline (s01–s07) als **Subprozess** und liefert die
Log-Ausgabe zeilenweise zurück, damit die Streamlit-Oberfläche nicht
blockiert. Die Pipeline selbst wird NICHT umgeschrieben – wir rufen den
Runner ``run_pipeline.py`` auf, der wiederum die vorhandenen Funktionen
``load_config`` / ``run_pipeline_with_cfg`` programmatisch nutzt.

Warum Subprozess statt Thread?
- Die Pipeline lädt schwere Bibliotheken (spaCy, gensim) und nutzt teils
  globalen Zustand; ein eigener Prozess isoliert das sauber.
- stdout lässt sich zeilenweise streamen (Live-Log).
- Ein hängender Lauf kann hart beendet werden, ohne den Streamlit-Server
  zu gefährden.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Schritt-Beschriftungen – wörtlich aus dem Docstring von pipeline_config.py.
STEP_LABELS: Dict[str, str] = {
    "1": "s01_1 · Vorverarbeitung (korpus_min/lem/stop)",
    "2": "s01_2 · Vokabular",
    "4": "s01_4 · POS-Tagging der Top-5000",
    "5": "s02 · Gensim-Preprocessing (korpus_gen)",
    "6": "s03 · DTM- & TF-IDF-Matrizen",
    "7": "s04 · Kosinus-Matrizen",
    "8": "s05 · Intervall-Matrizen",
    "9": "s06 · TF-IDF-Ranglisten",
    "10": "s07 · Word2Vec-Modelle",
}


def available_configs(project_root: Path) -> List[Path]:
    """Findet die TOML-Konfigurationen im Projektordner (config/*.toml)."""
    root = Path(project_root)
    configs: List[Path] = []
    for pattern in ("config/*.toml", "*.toml"):
        configs.extend(sorted(root.glob(pattern)))
    # Duplikate entfernen, pyproject.toml ausblenden (keine Pipeline-Config)
    seen, result = set(), []
    for p in configs:
        if p.name == "pyproject.toml":
            continue
        if p.resolve() not in seen:
            seen.add(p.resolve())
            result.append(p)
    return result


def build_command(
    runner_path: Path,
    project_root: Path,
    config_path: Path,
    steps: Optional[List[str]] = None,
    python_exe: Optional[str] = None,
    intervals: Optional[List[str]] = None,
    w2v_params: Optional[Dict] = None,
    remove_names: bool = False,
    detect_names: bool = False,
    terms_file: Optional[Path] = None,
    make_pos: bool = False,
    pos_file: Optional[Path] = None,
    top_terms: Optional[int] = None,
) -> List[str]:
    """Baut den Subprozess-Befehl: python -u run_pipeline.py …

    ``-u`` schaltet die Ausgabepufferung ab, damit das Log live ankommt.
    ``w2v_params`` (optional) werden als JSON an ``--w2v-params`` übergeben.
    ``detect_names`` erkennt nur Eigennamen (``--detect-names``); mit ``make_pos``
    wird zuerst eine neue POS-Liste für ``top_terms`` Top-Ausdrücke erzeugt, mit
    ``pos_file`` die PROPN aus einer vorhandenen POS-Liste geladen.
    ``remove_names`` startet die namensbereinigte Variante (``--remove-names``),
    optional mit ``terms_file`` (kuratierte Auswahl); die Schritte steuert der
    Runner selbst.
    """
    exe = python_exe or sys.executable
    try:
        cfg_arg = str(config_path.relative_to(project_root))
    except ValueError:
        cfg_arg = str(config_path)
    cmd = [exe, "-u", str(runner_path),
           "--project-root", str(project_root),
           "--config", cfg_arg]
    if detect_names:
        cmd += ["--detect-names"]
        if make_pos:
            cmd += ["--make-pos"]
            if top_terms:
                cmd += ["--top-terms", str(int(top_terms))]
        elif pos_file:
            cmd += ["--pos-file", str(pos_file)]
    elif remove_names:
        cmd += ["--remove-names"]
        if terms_file:
            cmd += ["--terms-file", str(terms_file)]
    elif steps:
        cmd += ["--steps", *steps]
    if intervals:
        cmd += ["--intervals", *intervals]
    if w2v_params:
        import json
        cmd += ["--w2v-params", json.dumps(w2v_params)]
    return cmd


def stream_pipeline(
    runner_path: Path,
    project_root: Path,
    config_path: Path,
    steps: Optional[List[str]] = None,
    python_exe: Optional[str] = None,
    intervals: Optional[List[str]] = None,
    w2v_params: Optional[Dict] = None,
    remove_names: bool = False,
    detect_names: bool = False,
    terms_file: Optional[Path] = None,
    make_pos: bool = False,
    pos_file: Optional[Path] = None,
    top_terms: Optional[int] = None,
) -> Iterator[Tuple[str, object]]:
    """Startet die Pipeline und liefert Ereignisse als (typ, inhalt):

    - ("log", "<Zeile>")  für jede stdout-Zeile
    - ("done", returncode) am Ende

    Beispiel:
        for kind, payload in stream_pipeline(...):
            if kind == "log":  ...
            elif kind == "done": ...
    """
    cmd = build_command(runner_path, project_root, config_path, steps, python_exe,
                        intervals=intervals, w2v_params=w2v_params,
                        remove_names=remove_names, detect_names=detect_names,
                        terms_file=terms_file, make_pos=make_pos, pos_file=pos_file,
                        top_terms=top_terms)
    # Kindprozess auf UTF-8 zwingen, damit Emoji-Ausgaben der Pipeline auch
    # unter Windows (cp1252) nicht zu UnicodeEncodeError führen.
    import os
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield "log", line.rstrip("\n")
    proc.wait()
    yield "done", proc.returncode


def list_output_tree(project_root: Path, max_entries: int = 60) -> List[str]:
    """Listet die Inhalte von ``output/`` (für die Ergebnisanzeige nach dem Lauf)."""
    out = Path(project_root) / "output"
    if not out.exists():
        return []
    entries: List[str] = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            rel = path.relative_to(project_root)
            entries.append(str(rel))
            if len(entries) >= max_entries:
                entries.append("… (weitere Dateien vorhanden)")
                break
    return entries
