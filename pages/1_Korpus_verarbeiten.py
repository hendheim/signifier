#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 8: Korpus verarbeiten
===========================

Startet die bestehende NLP-Pipeline (Schritte s01–s07) aus dem Dashboard.

Die Pipeline wird NICHT umgeschrieben: Diese Seite ruft den Runner
``run_pipeline.py`` als Subprozess auf (siehe ``explorer_core.pipeline_runner``),
der die vorhandenen Funktionen ``load_config`` / ``run_pipeline_with_cfg``
nutzt. Derselbe Lauf ist auch per Doppelklick über ``start_pipeline.bat`` /
``start_pipeline.sh`` möglich.

Voraussetzung: Auf der Startseite ist der Projektordner gesetzt (er enthält
``config/`` mit der TOML sowie ``konfig/``, ``output/``, ``resources/``).
"""

from pathlib import Path

import streamlit as st

from ui_helpers import get_store, show_error, APP_DIR
from explorer_core.pipeline_runner import (
    STEP_LABELS, available_configs, stream_pipeline, list_output_tree,
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die Word2Vec-Hyperparameter-Voreinstellung
# ---------------------------------------------------------------------------

def _suggest_w2v_params(num_tokens: int) -> dict:
    """Größenabhängige Word2Vec-Defaults.

    Nutzt – wenn verfügbar – dieselbe Funktion wie die Pipeline
    (``nlp_pipeline.s07_word_vectors.calculate_word2vec_params``), damit Vorschau und
    tatsächlicher Lauf identisch sind. Fällt auf eine lokale, wertgleiche
    Kopie zurück, falls der Import (z. B. gensim) hier nicht möglich ist.
    """
    try:  # bevorzugt die echte Pipeline-Funktion
        from nlp_pipeline.s07_word_vectors import calculate_word2vec_params
        return calculate_word2vec_params(num_tokens)
    except Exception:
        base = {"workers": 4, "sg": 1, "hs": 0, "sample": 1e-3, "seed": 42}
        if num_tokens < 100_000:
            base.update(vector_size=100, window=5, min_count=2, negative=10, epochs=40, _category="KLEIN")
        elif num_tokens < 1_000_000:
            base.update(vector_size=150, window=5, min_count=5, negative=5, epochs=20, _category="MITTEL")
        else:
            base.update(vector_size=300, window=5, min_count=10, negative=5, epochs=15, _category="GROSS")
        return base


def _detect_delimiter_safe(path: Path) -> str:
    """Trennzeichen robust erkennen – ohne Abhängigkeit vom Importpfad.

    Versucht zuerst die projektweite (gehärtete) Erkennung; schlägt der Import
    fehl (im Dashboard liegt ``pipeline_utils`` oft nicht unter dem bloßen
    Namen auf dem sys.path), nutzt sie eine selbständige, konsistenzbasierte
    Erkennung statt blind auf ";" zu setzen.
    """
    for modpath in ("explorer_core.data_store", "pipeline_utils"):
        try:
            mod = __import__(modpath, fromlist=["detect_delimiter"])
            return mod.detect_delimiter(path)
        except Exception:
            continue
    # Selbständiger Fallback (gleiche Logik wie pipeline_utils.detect_delimiter)
    import csv, io
    from collections import Counter
    candidates = [";", ",", "\t", "|"]
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(131072)
    except Exception:
        return ";"
    if not sample.strip():
        return ";"
    best, best_key = None, None
    for d in candidates:
        try:
            rows = [r for r in csv.reader(io.StringIO(sample), delimiter=d) if r][:25]
        except csv.Error:
            continue
        if len(rows) < 2:
            continue
        counts = [len(r) for r in rows]
        modal = Counter(counts).most_common(1)[0][0]
        if modal < 2:
            continue
        key = (round(sum(c == modal for c in counts) / len(counts), 4), modal)
        if best_key is None or key > best_key:
            best_key, best = key, d
    return best or ";"


def _count_corpus_tokens(input_dir: Path, pattern: str,
                         extra_dirs: tuple = ()) -> dict:
    """Zählt Tokens/Dokumente der korpus_gen-Grundlage für die Voreinstellung.

    Bevorzugt den Gesamtkorpus (Datei ohne ``_JJJJ-JJJJ``), sonst die größte
    Intervall-Datei. Sucht rekursiv unter ``input_dir`` und optionalen
    ``extra_dirs`` (z. B. dem output_dir). Das Trennzeichen wird
    importunabhängig erkannt (siehe ``_detect_delimiter_safe``).

    Gibt ein Diagnose-Dict zurück: tokens, docs, file, sep, content_col,
    searched (durchsuchte Verzeichnisse), n_files.
    """
    import re
    import pandas as pd

    diag = {"tokens": 0, "docs": 0, "file": "", "sep": "", "content_col": None,
            "searched": [], "n_files": 0}

    search_dirs = [d for d in (input_dir, *extra_dirs) if d is not None]
    files: list = []
    for d in search_dirs:
        diag["searched"].append(str(d))
        if d.exists():
            files.extend(sorted(d.rglob(pattern)))
    # Duplikate (gleicher Pfad) entfernen, Reihenfolge wahren
    seen = set()
    files = [f for f in files if not (f.resolve() in seen or seen.add(f.resolve()))]
    diag["n_files"] = len(files)
    if not files:
        return diag

    interval_re = re.compile(r"_\d{4}-\d{4}")
    gesamt = [f for f in files if not interval_re.search(f.stem)]
    chosen = gesamt[0] if gesamt else max(files, key=lambda p: p.stat().st_size)
    diag["file"] = chosen.name

    sep = _detect_delimiter_safe(chosen)
    diag["sep"] = sep
    try:
        df = pd.read_csv(chosen, encoding="utf-8", sep=sep)
    except Exception:
        return diag

    content_col = None
    try:  # bevorzugt die projektweite Content-Spalten-Erkennung
        from nlp_pipeline.pipeline_utils import identify_content_column
        content_col = identify_content_column(df)
    except Exception:
        pass
    if content_col is None:
        for cand in ("content_gen", "content_stop", "content_lem",
                     "content_min", "content", "text", "clean_text"):
            if cand in df.columns:
                content_col = cand
                break
    diag["content_col"] = content_col
    diag["docs"] = len(df)
    if content_col is None:
        return diag

    texts = df[content_col].astype(str)
    diag["tokens"] = int(texts.apply(lambda x: len(x.split())).sum())
    return diag

st.set_page_config(page_title="Korpus verarbeiten", layout="wide")
st.title("⚙️ Korpus verarbeiten")
st.caption("Vorverarbeitungs-Pipeline ausführen: Vorverarbeitung, Vokabular, "
           "Statistik, POS-Tagging, Matrizen, TF-IDF-Ranglisten. (Die "
           "Wortvektoren entstehen auf der Seite 'Wortvektormodell erstellen'.)")

store = get_store()
project_root = Path(store.project_root)
runner_path = APP_DIR / "run_pipeline.py"

st.info(
    "Diese Seite startet die vorhandene Pipeline – sie schreibt Ergebnisse "
    "nach `output/`. Lange Schritte (Word2Vec, POS-Tagging) können einige "
    "Minuten dauern. Das Log erscheint live."
)

# ---------------------------------------------------------------------------
# 1) Konfiguration wählen
# ---------------------------------------------------------------------------
st.subheader("1 · Konfiguration")

configs = available_configs(project_root)
if not configs:
    st.warning(
        f"Keine TOML-Konfiguration im Projektordner gefunden "
        f"(`{project_root}`). Erwartet z. B. `config/fadelive_v3.toml`. "
        "Pfad ggf. auf der Startseite korrigieren."
    )
    st.stop()

# fadelive_v3.toml als Default vorauswählen, falls vorhanden
names = [str(p.relative_to(project_root)) for p in configs]
default_idx = next((i for i, n in enumerate(names) if n.endswith("fadelive_v3.toml")), 0)
chosen_name = st.selectbox("TOML-Konfiguration", names, index=default_idx)
config_path = project_root / chosen_name

# ---------------------------------------------------------------------------
# 2) Schritte wählen
# ---------------------------------------------------------------------------
st.subheader("2 · Schritte")

# Schritt 10 (Word2Vec/s07) ist auf eine eigene Seite ausgelagert.
all_steps = [s for s in STEP_LABELS.keys() if s != "10"]
run_all = st.checkbox("Alle Vorverarbeitungsschritte ausführen", value=True)

if run_all:
    steps = all_steps
    st.caption("Es laufen alle Schritte in Reihenfolge.")
else:
    chosen = st.multiselect(
        "Einzelne Schritte auswählen",
        options=all_steps,
        default=all_steps,
        format_func=lambda s: f"{s} · {STEP_LABELS[s]}",
    )
    # Reihenfolge 1→10 erzwingen, egal wie ausgewählt
    steps = [s for s in all_steps if s in chosen]

with st.expander("Was bedeuten die Schritte?"):
    for s in all_steps:
        st.markdown(f"- **{s}** — {STEP_LABELS[s]}")

# ---------------------------------------------------------------------------
# 3) Starten
# ---------------------------------------------------------------------------
st.subheader("3 · Starten")

# Optional: Intervalle für Vokabular (Schritt 2) und Word2Vec (Schritte 5/10).
# Leer = keine Intervalle erzwingen (dynamische Erzeugung bzw. Werte aus der TOML).
intervals_raw = st.text_input(
    "Intervalle (optional, kommagetrennt)",
    value="",
    help="z. B. 1782-1852, 1853-1864, 1865-1876, 1877-1891. "
         "Wirkt hier auf das Vokabular (Schritt 2). Die Word2Vec-Modelle je "
         "Intervall werden auf der Seite 'Wortvektormodell erstellen' trainiert. "
         "Leer lassen = ohne Intervalle.",
)
intervals = [s.strip() for s in intervals_raw.replace(";", ",").split(",") if s.strip()] or None
if intervals:
    st.caption("Aktive Intervalle: " + ", ".join(intervals))

# Word2Vec (Schritt 10/s07) ist auf eine eigene Seite ausgelagert -
# hier werden keine Word2Vec-Parameter mehr gesetzt.
w2v_params = None

if not runner_path.exists():
    st.error(f"Runner nicht gefunden: {runner_path}. "
             "Liegt `run_pipeline.py` im Dashboard-Ordner?")
    st.stop()

if not steps:
    st.warning("Bitte mindestens einen Schritt auswählen.")

start = st.button("▶️ Pipeline starten", type="primary", disabled=not steps)

if start:
    log_lines: list[str] = []
    log_box = st.empty()
    returncode = None
    try:
        with st.status("Pipeline läuft …", expanded=True) as status:
            for kind, payload in stream_pipeline(
                runner_path=runner_path,
                project_root=project_root,
                config_path=config_path,
                steps=steps,
                intervals=intervals,
                w2v_params=w2v_params,
            ):
                if kind == "log":
                    log_lines.append(str(payload))
                    # Nur die letzten ~400 Zeilen anzeigen (Performance)
                    log_box.code("\n".join(log_lines[-400:]), language="text")
                elif kind == "done":
                    returncode = payload
            if returncode == 0:
                status.update(label="Pipeline abgeschlossen ✅", state="complete")
            else:
                status.update(label=f"Pipeline mit Fehler beendet (Code {returncode})",
                              state="error")
    except Exception as exc:  # einheitliche, laienfreundliche Fehleranzeige
        show_error(exc)
    else:
        if returncode == 0:
            st.success("Fertig. Ergebnisse liegen in `output/`.")
        else:
            st.error("Die Pipeline ist nicht sauber durchgelaufen. Bitte das "
                     "Log oben prüfen (häufig: fehlende Eingabedateien, "
                     "falsche Pfade in der TOML, fehlende Modelle).")

        # Vollständiges Log zum Download
        st.download_button(
            "⬇️ Vollständiges Log herunterladen",
            "\n".join(log_lines).encode("utf-8"),
            file_name="pipeline_log.txt",
            mime="text/plain",
        )

        # Ergebnisübersicht
        outputs = list_output_tree(project_root)
        if outputs:
            with st.expander(f"Erzeugte Dateien in output/ ({len(outputs)})",
                            expanded=True):
                st.code("\n".join(outputs), language="text")

st.divider()
st.caption(
    "💡 Ohne Dashboard startbar: Doppelklick auf `start_pipeline.bat` "
    "(Windows) bzw. `./start_pipeline.sh` (macOS/Linux)."
)
