#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline
============

Startet die FaDeLive-Pipeline (Schritte s01–s07) – OHNE den paketierten
CLI-Einstiegspunkt (`fadelive`-Kommando bzw. `python -m fadelive.pipeline_config`).

Stattdessen werden die vorhandenen Funktionen ``load_config`` und
``run_pipeline_with_cfg`` aus ``pipeline_config.py`` **programmatisch**
importiert und aufgerufen. Damit ist der Start unabhängig davon, ob das
Paket installiert ist.

Wichtig: Die Pfade in der TOML sind relativ (z. B. ``korpus/korpus.csv``),
deshalb MUSS dieser Runner mit dem Arbeitsverzeichnis = Projekt-Root laufen.
Das übernehmen die Wrapper ``start_pipeline.bat`` / ``start_pipeline.sh`` und
die Dashboard-Seite "Korpus verarbeiten".

Direkter Aufruf (z. B. zum Testen):

    python run_pipeline.py --project-root /pfad/zum/projekt \\
        --config config/fadelive_v3.toml --steps 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _import_pipeline(project_root: Path):
    """Importiert load_config + run_pipeline_with_cfg möglichst robust.

    Reihenfolge der Versuche (keiner nutzt den CLI-/Konsolen-Einstiegspunkt):
      1. Paket ``<projekt>/nlp_pipeline`` (aktuelles Layout)
      2. installiertes Paket ``fadelive``
      3. Paketquelle ``<projekt>/src`` (Alt-Layout laut pyproject.toml)
      4. flach im Projekt-Root liegende ``pipeline_config.py``
    """
    # 1) bevorzugt: Paket nlp_pipeline im Projektordner
    if (project_root / "nlp_pipeline").exists():
        sys.path.insert(0, str(project_root))
        from nlp_pipeline.pipeline_config import load_config, run_pipeline_with_cfg
        return load_config, run_pipeline_with_cfg

    # 2) installiertes Paket
    try:
        from fadelive.pipeline_config import load_config, run_pipeline_with_cfg
        return load_config, run_pipeline_with_cfg
    except ImportError:
        pass

    # 2) src/fadelive im Projektordner
    src = project_root / "src"
    if (src / "fadelive").exists():
        sys.path.insert(0, str(src))
        from fadelive.pipeline_config import load_config, run_pipeline_with_cfg
        return load_config, run_pipeline_with_cfg

    # 3) pipeline_config.py direkt im Projekt-Root
    if (project_root / "pipeline_config.py").exists():
        sys.path.insert(0, str(project_root))
        from pipeline_config import load_config, run_pipeline_with_cfg  # type: ignore
        return load_config, run_pipeline_with_cfg

    raise ImportError(
        "Konnte pipeline_config nicht finden. Erwartet wurde '<projekt>/nlp_pipeline/"
        "pipeline_config.py', ein installiertes 'fadelive'-Paket, "
        "'<projekt>/src/fadelive/pipeline_config.py' oder '<projekt>/pipeline_config.py'."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FaDeLive-Pipeline programmatisch starten.")
    parser.add_argument("--project-root", default=".",
                        help="Projektordner mit korpus/, output/, resources/, config/ (Default: .)")
    parser.add_argument("--config", required=True,
                        help="Pfad zur TOML-Konfiguration, relativ zum Projektordner.")
    parser.add_argument("--steps", nargs="*",
                        help="Optional: nur bestimmte Schritte, z. B. 1 2 3.")
    parser.add_argument("--intervals", nargs="*",
                        help='Optional: Intervalle für Vokabular (Schritt 2) und '
                             'Gensim/Word2Vec (Schritte 5/10), z. B. 1782-1852 1853-1864. '
                             'Überschreibt die Werte aus der TOML.')
    parser.add_argument("--w2v-params",
                        help='Optional: JSON-Objekt mit Word2Vec-Hyperparametern '
                             '(z. B. \'{"vector_size":150,"window":5,"epochs":20}\'). '
                             'Überschreibt für Schritt 10 die größenabhängigen Defaults.')
    args = parser.parse_args(argv)

    # Windows-Konsole (cp1252) kann Emoji-Ausgaben der Skripte nicht codieren
    # -> Ausgabe auf UTF-8 umstellen, damit print("📁 …") nicht abstürzt.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    project_root = Path(args.project_root).resolve()

    # Relative TOML-/Datenpfade auflösen: ab hier ist das CWD der Projekt-Root.
    import os
    os.chdir(project_root)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    print(f"[run_pipeline] Projektordner : {project_root}", flush=True)
    print(f"[run_pipeline] Konfiguration : {config_path}", flush=True)
    if args.steps:
        print(f"[run_pipeline] Schritte      : {', '.join(args.steps)}", flush=True)

    load_config, run_pipeline_with_cfg = _import_pipeline(project_root)

    cfg = load_config(config_path)
    if args.steps:
        cfg.setdefault("run", {})["steps"] = list(args.steps)

    # Intervalle überschreiben: betreffen das Vokabular (Schritt 2), die
    # Gensim-Vorverarbeitung (Schritt 5, erzeugt korpus_gen_<I>.csv -> Schritt 10
    # trainiert daraus Intervall-Word2Vec-Modelle) sowie die Intervall-Matrizen
    # (Schritt 8). Intervalle entstehen NUR über diese explizite Eingabe.
    if args.intervals:
        intervals = list(args.intervals)
        print(f"[run_pipeline] Intervalle    : {', '.join(intervals)}", flush=True)
        if "step2_s01_2_vocabular" in cfg:
            cfg["step2_s01_2_vocabular"]["intervals"] = intervals
        if "step5_s02_preprocessing_gensim" in cfg:
            cfg["step5_s02_preprocessing_gensim"]["intervals"] = intervals
        if "step8_s05_dtm_tfidf_cos_intervals" in cfg:
            cfg["step8_s05_dtm_tfidf_cos_intervals"]["intervals"] = intervals

    # Word2Vec-Hyperparameter aus dem Dashboard (JSON) in die Step-10-Config
    # einspeisen; s07 kombiniert sie mit den größenabhängigen Defaults.
    if args.w2v_params:
        import json
        try:
            w2v = json.loads(args.w2v_params)
        except json.JSONDecodeError as exc:
            print(f"[run_pipeline] WARN: --w2v-params ist kein gültiges JSON ({exc}); ignoriert.",
                  flush=True)
            w2v = None
        if isinstance(w2v, dict) and w2v:
            print(f"[run_pipeline] Word2Vec-Hyperparameter: {w2v}", flush=True)
            cfg.setdefault("step10_s07_word_vector_model", {})["hyperparams"] = w2v

    run_pipeline_with_cfg(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
