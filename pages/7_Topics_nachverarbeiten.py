#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Topics nachbearbeiten
============================

Topic-Postprocessing über ``tt_s02_topics.py``: Topic-Ranking, Top-N-
Dokumente pro Topic, Metadaten-Mapping und Jahr-/Text-Aggregationen.
Trenner und Jahresspalte werden automatisch über die App-Funktionen
erkannt (date wird wie year behandelt).
"""

import sys
import io
import contextlib
import importlib.util
from pathlib import Path

import streamlit as st

from ui_helpers import get_store, get_schema, show_error, APP_DIR
from explorer_core.data_store import detect_delimiter, read_csv_auto

st.set_page_config(page_title="Topics nachbearbeiten", layout="wide")
st.title("🧮 Topics nachbearbeiten")
st.caption("Topic-Postprocessing (tt_s02): Ranking, Top-N-Dokumente, Metadaten-Mapping, Jahr-Aggregationen.")

store = get_store()
schema = get_schema()
project_root = Path(store.project_root)
korpus_csv = project_root / "korpus" / "korpus.csv"


def _load(modname: str, filename: str):
    """Lädt ein Skript-Modul über seinen Dateipfad (robust ggü. Bindestrich)."""
    path = APP_DIR / "nlp_pipeline" / filename
    if not path.exists():
        raise FileNotFoundError(f"Skript nicht gefunden: {path}")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_capture(fn, *args, **kwargs):
    """Ruft fn auf und schneidet stdout/stderr als Log mit."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _glob_select(label: str, patterns, key: str):
    """Dateiauswahl per Glob im Projektordner, mit manueller Alternative."""
    hits = []
    for pat in patterns:
        hits.extend(sorted(project_root.glob(pat)))
    hits = list(dict.fromkeys(hits))
    if hits:
        names = [str(p.relative_to(project_root)) for p in hits]
        choice = st.selectbox(label, names, index=len(names) - 1, key=key)
        return project_root / choice
    manual = st.text_input(f"{label} (Pfad)", value="", key=f"{key}_manual")
    return Path(manual) if manual else None


st.subheader("Topic-Postprocessing")
input_file = _glob_select(
    "Document-Topic-Distribution (CSV)",
    ["resources/topic-models/**/document-topics-distribution*.csv",
     "resources/topic-models/**/*document*topic*.csv"], key="s02_in")
out_dir = Path(st.text_input("Ausgabeordner", value="output/processed_topics",
                             key="s02_out"))
c1, c2, c3 = st.columns(3)
top_n = c1.number_input("Top-N Dokumente pro Topic", 1, 500, 50,
                        key="s02_topn")
header_row = c2.number_input("Header-Zeile (0-basiert)", 0, 10, 0,
                             key="s02_hdr")
index_col = c3.number_input("Index-Spalte (0-basiert, -1 = keine)", -1, 20,
                            0, key="s02_idx")
meta_file = Path(st.text_input("Metadaten-Datei", value=str(korpus_csv),
                               key="s02_meta"))

# Trenner und Jahresspalte automatisch über die bestehenden App-Funktionen
# erkennen, damit kein manuelles Einstellen nötig ist. 'date' wird dabei
# genauso als Jahr erkannt wie 'year'/'jahr' (Schema-Kandidaten).
detected_sep = "auto"
detected_year_col = None
if meta_file and str(meta_file).strip() and meta_file.exists():
    try:
        detected_sep = detect_delimiter(meta_file)
        yf, ym = schema.find_year_columns(read_csv_auto(meta_file))
        detected_year_col = yf or ym
        disp = {"\t": "Tab", " ": "Leerzeichen"}.get(detected_sep, detected_sep)
        st.caption(
            f"Automatisch erkannt - Trenner: '{disp}'"
            + (f", Jahresspalte: '{detected_year_col}'" if detected_year_col
               else ", keine Jahresspalte gefunden"))
    except Exception:
        st.caption("Trenner/Jahresspalte nicht automatisch erkennbar - "
                   "es werden Standardwerte verwendet.")
with st.expander("Erkennung übersteuern (optional)"):
    sep_override = st.text_input("Trenner (leer = automatisch)", value="",
                                 key="s02_sep")
    year_override = st.text_input("Jahresspalte (leer = automatisch)",
                                  value="", key="s02_year")
strip_txt = st.checkbox("'.txt'-Endung der Dokument-IDs entfernen",
                        key="s02_strip")

if st.button("Topics verarbeiten (s02)", key="s02_btn", type="primary"):
    if not input_file or not input_file.exists():
        st.warning("Bitte eine gültige Document-Topic-Distribution wählen.")
    else:
        try:
            s02 = _load("tt_s02_topics", "tt_s02_topics.py")
            meta_sep = (sep_override.strip() or detected_sep or "auto")
            year_col = (year_override.strip() or (detected_year_col or ""))
            argv = ["--input-file", str(input_file),
                    "--output-dir", str(project_root / out_dir),
                    "--header-row", str(int(header_row)),
                    "--index-col", str(int(index_col)),
                    "--top-n-docs", str(int(top_n)),
                    "--meta-sep", meta_sep]
            if meta_file and str(meta_file).strip():
                argv += ["--metadata-file", str(meta_file)]
            if year_col:
                argv += ["--year-column", year_col]
            if strip_txt:
                argv += ["--strip-txt-suffix"]
            old = sys.argv
            sys.argv = ["tt_s02_topics.py"] + argv
            try:
                log = _run_capture(s02.main)
            finally:
                sys.argv = old
            st.success("s02 abgeschlossen.")
            st.code(log or "(keine Ausgabe)", language="text")
        except Exception as e:
            show_error(e)
