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
    parser.add_argument("--remove-names", action="store_true",
                        help="Namensbereinigte Variante '-n' erzeugen: tilgt "
                             "Eigennamen (Figuren/Autoren) aus den vorhandenen "
                             "Stufen korpus_min/lem/stop/gen und rechnet die "
                             "Folgeschritte in '-n'-Ordner. Startet ab corpus_stop "
                             "(keine Neu-Lemmatisierung). Mit --terms-file werden "
                             "genau die dort gelisteten Ausdrücke getilgt, sonst "
                             "automatisch erkannt.")
    parser.add_argument("--terms-file",
                        help="Optional: CSV mit Spalte 'term' – die zu tilgenden "
                             "Ausdrücke (kuratierte Auswahl). Nur mit --remove-names.")
    parser.add_argument("--detect-names", action="store_true",
                        help="Nur Eigennamen erkennen und als CSV "
                             "(output/processed_corpus/name_candidates.csv, Spalten "
                             "term,frequency) speichern – ohne zu tilgen.")
    parser.add_argument("--make-pos", action="store_true",
                        help="Optional (mit --detect-names): zuerst eine neue "
                             "POS-Liste erzeugen (gleiche Grundlage wie Schritt 4: "
                             "vocab_full_stop.json) und daraus die PROPN-Kandidaten "
                             "ableiten. Ohne Angabe wird die vorhandene POS-Liste "
                             "genutzt bzw. – ganz ohne --pos-file – das Korpus getaggt.")
    parser.add_argument("--pos-file",
                        help="Optional (mit --detect-names): vorhandene POS-Liste "
                             "(CSV mit Spalten word,pos,count), aus der die PROPN "
                             "als Kandidaten geladen werden.")
    parser.add_argument("--top-terms", type=int, default=None,
                        help="Optional (mit --make-pos): für wie viele Top-Ausdrücke "
                             "die POS-Liste erzeugt wird.")
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

    # Für die Eigennamen-Erkennung bewusst das große Modell (bessere POS/NER);
    # load_tagger fällt auf ein vorhandenes Modell zurück.
    NER_MODEL = "de_core_news_lg"
    _step1 = cfg.get("step1_s01_1_preprocessing", {})
    _processed_dir = Path(_step1.get("output_dir", "output/processed_corpus"))
    _metadata_path = Path("korpus") / "metadaten.csv"

    if args.detect_names:
        # Nur erkennen: Kandidatenliste (term, frequency) schreiben – ohne Tilgung.
        try:
            from nlp_pipeline.s08_name_removal import (
                detect_name_candidates, propn_candidates_from_pos)
        except ImportError:
            from s08_name_removal import (
                detect_name_candidates, propn_candidates_from_pos)
        print("=" * 80, flush=True)
        print("EIGENNAMEN ERKENNEN (Kandidatenliste)", flush=True)
        print("=" * 80, flush=True)
        if args.make_pos:
            # Neue POS-Liste erzeugen (gleiche Grundlage wie Schritt 4), dann PROPN.
            step4 = cfg.get("step4_s01_4_pos_tag", {})
            input_json = step4.get("input_json", "output/vocabular/vocab_full_stop.json")
            model = step4.get("model", "de_core_news_md")
            limit = int(args.top_terms or step4.get("limit", 5000))
            pos_out = Path("output/vocabular") / f"vocab_top{limit}_stop_pos.csv"
            print(f"[detect-names] Erzeuge POS-Liste (Top-{limit}) aus "
                  f"{input_json} mit {model} ...", flush=True)
            try:
                from nlp_pipeline.s01_pos_tag import run as pos_run
            except ImportError:
                from s01_pos_tag import run as pos_run
            pos_run(input_json, str(pos_out), model=model, limit=limit)
            cand = propn_candidates_from_pos(pos_out)
        elif args.pos_file:
            cand = propn_candidates_from_pos(Path(args.pos_file))
        else:
            cand = detect_name_candidates(_processed_dir, model=NER_MODEL,
                                          metadata_path=_metadata_path)
        out_csv = _processed_dir / "name_candidates.csv"
        cand.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[detect-names] {len(cand)} Kandidaten gespeichert: {out_csv}", flush=True)
        return 0

    if args.remove_names:
        # Namensbereinigte Variante "-n" (ab corpus_stop, ohne Neu-Lemmatisierung).
        # Mit --terms-file: genau die gelisteten Ausdrücke; sonst automatisch erkennen.
        try:
            from nlp_pipeline.s08_name_removal import (
                build_stop_n_corpus, strip_terms_corpus, load_terms, derive_stop_n_cfg)
        except ImportError:
            from s08_name_removal import (
                build_stop_n_corpus, strip_terms_corpus, load_terms, derive_stop_n_cfg)

        out_dir = _processed_dir.parent / (_processed_dir.name + "-n")
        print("=" * 80, flush=True)
        print("NAMENSTILGUNG → Variante '-n'", flush=True)
        print("=" * 80, flush=True)
        if args.terms_file:
            terms = load_terms(Path(args.terms_file))
            print(f"[remove-names] kuratierte Liste: {len(terms)} Ausdrücke "
                  f"aus {args.terms_file}", flush=True)
            diag = strip_terms_corpus(_processed_dir, out_dir, terms)
        else:
            diag = build_stop_n_corpus(_processed_dir, out_dir, model=NER_MODEL,
                                       metadata_path=_metadata_path)
        print(f"[remove-names] {diag['n_names']} Ausdrücke getilgt; "
              f"Stufen: {', '.join(diag['written']) or '—'}", flush=True)

        run_pipeline_with_cfg(derive_stop_n_cfg(cfg))
        return 0

    run_pipeline_with_cfg(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
