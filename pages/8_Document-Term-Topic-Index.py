#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Document-Term-Topic-Index (DTTI)
=======================================

Termset-Topic-DTTI in zwei Schritten/Tabs: ``tt_s03_dtti.py`` erzeugt die
DTTI-Matrizen (Tab "DTTI berechnen"), ``tt_s04_dtti.py`` verarbeitet sie nach
(Tab "DTTI nachverarbeiten"). Trenner und Jahresspalte werden automatisch über
die App-Funktionen erkannt (date wie year).
"""

import sys
import io
import contextlib
import importlib.util
from pathlib import Path

import streamlit as st

from ui_helpers import get_store, get_schema, show_error, APP_DIR
from explorer_core.data_store import detect_delimiter, read_csv_auto

st.set_page_config(page_title="Document-Term-Topic-Index", layout="wide")
st.title("🧮 Document-Term-Topic-Index (DTTI)")
st.caption("Es werden DTTI-Matrizen erzeugt und nachverarbeitet.")

store = get_store()
schema = get_schema()
project_root = Path(store.project_root)
metadaten_csv = project_root / "korpus" / "metadaten.csv"


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


st.caption("Berechung des DTTI und Nachverarbeitung - jeder Schritt in "
           "seinem eigenen Tab.")

with st.expander("Was ist der DTTI? (Erläuterung & Formel)"):
    st.markdown(
        "Der **Dokument-Term-Topic-Index (DTTI)** misst, wie stark ein "
        "Dokument über seine Topics an ein **Termset** *B* gebunden ist – "
        "ein Termset ist eine kuratierte Wortliste (z. B. 'Gegenstände'). "
        "Hoch ist der Wert, wenn (1) das Topic im Dokument prominent ist "
        "und (2) das Dokument genau jene Termset-Wörter verwendet, die im "
        "Topic charakteristisch (hoher TF-IDF) und weit oben platziert sind."
    )
    st.latex(
        r"\mathrm{DTTI}(D,T)=\cos(D,T)\;\sum_{w\,\in\,D\cap T\cap B}"
        r"\operatorname{freq}(w,D)\;\cdot\;"
        r"\frac{\operatorname{tfidf\_sum}(w)}{\log\!\big(\operatorname{rang}_T(w)+1\big)}"
    )
    st.markdown(
        "- $\\cos(D,T)$: Gewicht des Topics $T$ im Dokument $D$ "
        "(aus der Dokument-Topic-Verteilung)\n"
        "- $D\\cap T\\cap B$: Wörter, die zugleich im Dokument, in den "
        "Top-Wörtern des Topics **und** im Termset stehen\n"
        "- $\\operatorname{freq}(w,D)$: Häufigkeit von $w$ in $D$ (aus der DTM)\n"
        "- $\\operatorname{tfidf\\_sum}(w)$: über das Korpus aufsummierter "
        "TF-IDF-Wert von $w$\n"
        "- $\\operatorname{rang}_T(w)$: Position von $w$ in der Topic-Wortliste "
        "– frühe Positionen zählen durch $1/\\log(\\operatorname{rang}+1)$ stärker"
    )
    st.markdown(
        "Auf **Topic-Ebene** (ohne einzelnes Dokument) wird derselbe Kern "
        "als *TFIDF-Positions-Score* verwendet:"
    )
    st.latex(
        r"\mathrm{Score}(T)=\sum_{w\,\in\,T\cap B}"
        r"\frac{\operatorname{tfidf\_sum}(w)}{\log\!\big(\operatorname{pos}_T(w)+1\big)}"
    )

tab_calc, tab_post = st.tabs(["DTTI berechnen", "DTTI nachverarbeiten"])

# ===========================================================================
# Tab: DTTI berechnen (s03)
# ===========================================================================
with tab_calc:
    st.caption("Erzeugt aus Termset, Topic-Wörtern, TF-IDF und DTM "
               "die DTTI-Matrizen (u. a. dtti_matrix_norm und tag_topic_rank).")

    st.markdown("**Benötigte Dateien**")
    termset_file = _glob_select("Termset (Pivot-CSV)",
                                ["resources/termsets/*.csv"], key="s03_termset")
    topic_word_file = _glob_select("Top-Topic-Words-Matrix",
                                   ["resources/topic-models/**/*words*tag*.csv",
                                    "resources/topic-models/**/*word*.csv"],
                                   key="s03_tw")
    topic_rank_file = _glob_select("Topic-Rangliste (tag_topic_rank)",
                                   ["output/processed_termset/**/*tag_topic_rank*.csv",
                                    "output/processed_topics/*rank*.csv"],
                                   key="s03_rank")
    tfidf_file = _glob_select("TF-IDF-Matrix",
                              ["output/dtm_tfidf*/tfidf*.csv"], key="s03_tfidf")
    doc_topic_file = _glob_select(
        "Document-Topic-Distribution (CSV)",
        ["resources/topic-models/**/document-topics-distribution*.csv"],
        key="s03_dt")
    dtm_file = _glob_select("Document-Term-Matrix (DTM)",
                            ["output/dtm_tfidf*/dtm*.csv"], key="s03_dtm")
    out_dir3 = st.text_input(
        "Ausgabeordner (Termset-Unterordner)",
        value="output/processed_termset", key="s03_out")

    # --- Metadaten-Grenzen automatisch über die App-Funktionen erkennen ---
    # ID-Spalte über die Schema-Kandidaten, Feature-Start = Index der ersten
    # Term-Spalte (alles davor sind Metadaten). Wird nur bei Dateiwechsel neu
    # bestimmt und seedet die anpassbaren Felder unten.
    def _detect_meta_bounds(path):
        try:
            dfh = read_csv_auto(Path(path))
        except Exception:
            return None, None
        start = None
        try:
            tcols = store.term_columns(dfh)
            if tcols:
                start = int(min(dfh.columns.get_loc(c) for c in tcols))
        except Exception:
            start = None
        return start, schema.find_id_column(dfh)

    sig = f"{tfidf_file}|{dtm_file}"
    if st.session_state.get("_s03_detsig") != sig:
        st.session_state["_s03_detsig"] = sig
        det_dtm_start = det_id = det_tfidf_start = None
        if dtm_file and Path(dtm_file).exists():
            det_dtm_start, det_id = _detect_meta_bounds(dtm_file)
        if tfidf_file and Path(tfidf_file).exists():
            det_tfidf_start, _ = _detect_meta_bounds(tfidf_file)
        def_tfs = det_tfidf_start if det_tfidf_start is not None else 10
        def_dts = det_dtm_start if det_dtm_start is not None else 10
        def_id = det_id or "_id"
        st.session_state["s03_tfs"] = int(def_tfs)
        st.session_state["s03_dts"] = int(def_dts)
        st.session_state["s03_idcol"] = def_id
        st.session_state["_s03_detinfo"] = (
            (det_tfidf_start is not None or det_dtm_start is not None
             or bool(det_id)), def_id, int(def_tfs), int(def_dts))

    info = st.session_state.get("_s03_detinfo")
    if info and info[0]:
        st.caption(f"Automatisch erkannt - ID-Spalte: '{info[1]}', "
                   f"Feature-Start TF-IDF: {info[2]}, DTM: {info[3]}. "
                   "Bei Bedarf unten anpassen.")

    st.markdown("**Einstellungen** (automatisch erkannt, anpassbar)")
    c1, c2, c3 = st.columns(3)
    tfidf_start = c1.number_input(
        "TF-IDF Start-Spalte (Index)", 0, 100, key="s03_tfs",
        help="Spaltenindex, ab dem die Term-/Feature-Spalten beginnen (davor "
             "stehen die Metadaten). Wird aus der Datei erkannt.")
    dtm_start = c2.number_input(
        "DTM Start-Spalte (Index)", 0, 100, key="s03_dts",
        help="Spaltenindex, ab dem die Term-Spalten der DTM beginnen. Wird aus "
             "der Datei erkannt.")
    dtm_id_col = c3.text_input(
        "DTM ID-Spalte", key="s03_idcol",
        help="Spalte mit der Dokument-ID; über die Schema-Kandidaten erkannt "
             "(z. B. _id, id, doc_id).")

    if st.button("DTTI berechnen (s03)", key="s03_btn", type="primary"):
        needed = {"Termset": termset_file, "Topic-Words": topic_word_file,
                  "Topic-Rank": topic_rank_file, "TF-IDF": tfidf_file,
                  "Doc-Topic": doc_topic_file, "DTM": dtm_file}
        missing = [k for k, v in needed.items() if not v or not Path(v).exists()]
        if missing:
            st.warning("Fehlende/ungültige Eingaben: " + ", ".join(missing))
        else:
            try:
                s03 = _load("tt_s03_dtti", "tt_s03_dtti.py")
                termset_basename = Path(termset_file).stem
                output_dir = project_root / out_dir3 / termset_basename
                log3 = _run_capture(
                    s03.run,
                    termset_file=termset_file, topic_word_file=topic_word_file,
                    topic_rank_file=topic_rank_file, tfidf_file=tfidf_file,
                    doc_topic_file=doc_topic_file, dtm_file=dtm_file,
                    output_dir=output_dir,
                    tfidf_start_col_index=int(tfidf_start),
                    dtm_start_col_index=int(dtm_start),
                    dtm_id_col=dtm_id_col)
                st.session_state["dtti_last_dir"] = str(output_dir)
                st.success(f"DTTI berechnet (s03). Ergebnis in `{output_dir}`. "
                           "Weiter im Tab 'DTTI nachverarbeiten'.")
                st.code(log3 or "(keine Ausgabe)", language="text")
            except Exception as e:
                show_error(e)

# ===========================================================================
# Tab: DTTI nachverarbeiten (s04)
# ===========================================================================
with tab_post:
    st.caption("Verarbeitet die in 'DTTI berechnen' erzeugten "
               "Matrizen nach (Top-Dokumente je Topic, Metadaten-Mapping, "
               "Jahr-Aggregationen).")

    st.markdown("**Einstellungen**")
    c4, c5, c6 = st.columns(3)
    top_n_docs = c4.number_input("Top-N Dokumente", 1, 500, 50, key="s04_topn")
    top_k_topics = c5.number_input("Top-K Topics", 1, 100, 10, key="s04_topk")
    max_rank = c6.number_input("Max. Rang", 1, 200, 30, key="s04_maxrank")

    st.markdown("**Benötigte Dateien (Ergebnis aus 'DTTI berechnen')**")
    dtti_norm = _glob_select(
        "DTTI-Matrix (normalisiert)",
        ["output/processed_termset/**/*dtti_matrix_norm*.csv"], key="s04_norm")
    rank_for_s04 = _glob_select(
        "Tag-Topic-Rangliste",
        ["output/processed_termset/**/*tag_topic_rank*.csv"], key="s04_rank")


    st.markdown("**Metadaten**")
    meta_file4 = Path(st.text_input("Metadaten-Datei", value=str(metadaten_csv),
                                    key="s04_meta"))
    # Trenner, ID- und Jahresspalte automatisch erkennen; 'date' wird genauso
    # als Jahr erkannt wie 'year' (Schema-Kandidaten).
    detected_sep4 = "auto"
    detected_year_col4 = None
    detected_id_col4 = None
    if meta_file4 and str(meta_file4).strip() and meta_file4.exists():
        try:
            detected_sep4 = detect_delimiter(meta_file4)
            _mdf4 = read_csv_auto(meta_file4)
            yf4, ym4 = schema.find_year_columns(_mdf4)
            detected_year_col4 = yf4 or ym4
            detected_id_col4 = schema.find_id_column(_mdf4)
            disp4 = {"\t": "Tab", " ": "Leerzeichen"}.get(detected_sep4, detected_sep4)
            st.caption(
                f"Automatisch erkannt - Trenner: '{disp4}'"
                + (f", ID-Spalte: '{detected_id_col4}'" if detected_id_col4 else "")
                + (f", Jahresspalte: '{detected_year_col4}'" if detected_year_col4
                   else ", keine Jahresspalte gefunden"))
        except Exception:
            st.caption("Trenner/Metadaten nicht automatisch erkennbar - "
                       "Standardwerte werden verwendet.")
    with st.expander("Erkennung übersteuern (optional)"):
        sep_override4 = st.text_input("Trenner (leer = automatisch)", value="",
                                      key="s04_sep")
        id_override4 = st.text_input("ID-Spalte (leer = automatisch)", value="",
                                     key="s04_idcol")
        year_override4 = st.text_input("Jahresspalte (leer = automatisch)",
                                       value="", key="s04_year")

    if st.button("DTTI nachverarbeiten (s04)", key="s04_btn", type="primary"):
        if (not dtti_norm or not Path(dtti_norm).exists()
                or not rank_for_s04 or not Path(rank_for_s04).exists()):
            st.warning("Bitte eine gültige DTTI-Matrix und Tag-Topic-Rangliste "
                       "wählen. Diese entstehen im Tab 'DTTI berechnen'.")
        else:
            try:
                s04 = _load("tt_s04_dtti", "tt_s04_dtti.py")
                meta_sep4 = (sep_override4.strip() or detected_sep4 or "auto")
                year_col4 = (year_override4.strip()
                             or (detected_year_col4 or "")) or None
                id_col4 = (id_override4.strip()
                           or (detected_id_col4 or "")) or None
                output_dir = Path(dtti_norm).parent
                log4 = _run_capture(
                    s04.run,
                    dtti_matrix_norm=Path(dtti_norm),
                    topic_rank_file=Path(rank_for_s04),
                    metadata_file=meta_file4, meta_sep=meta_sep4,
                    year_column=year_col4, metadata_id_column=id_col4,
                    output_dir=output_dir, top_n_docs=int(top_n_docs),
                    top_k_topics=int(top_k_topics), max_rank=int(max_rank))
                st.success("DTTI nachverarbeitet (s04).")
                st.code(log4 or "(keine Ausgabe)", language="text")
            except Exception as e:
                show_error(e)
