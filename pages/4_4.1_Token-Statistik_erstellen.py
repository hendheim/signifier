#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Statistik
================

Ersetzt den früheren Pipeline-Schritt ``s01_3_statistics`` durch eine
metadatengetriebene, auswählbare Auswertung:

- **Tokens je Stufe** (content/min/lem/stop/gen) – per Checkbox.
- **Milestones** – Gesamttokens / N gleich große Abschnitte mit Jahresspanne.
- **Frequenz je Metadatenspalte** – Dokumente *und* Tokens je Wert in einer
  Grafik; je Spalte per Checkbox (alle vorausgewählt).

Optional lassen sich die Ergebnisse nach ``output/statistics/`` schreiben
(u. a. ``year_count_tokens.csv``, das die Tag-Topics-Seite nutzt).

Rechenlogik UI-frei in ``explorer_core.analysis_stats``.
"""

from pathlib import Path

import streamlit as st

from ui_helpers import get_store, get_schema, show_error, df_with_download
from explorer_core import analysis_stats as stats
from explorer_core.viz_export import save_figure

st.set_page_config(page_title="Token-Statistik erstellen", layout="wide")
st.title("📊 Token-Statistik")
st.caption("Tokens je Stufe, Milestones und Frequenzen je Metadatenspalte – "
           "alles auswählbar.")

store = get_store()
schema = get_schema()
project_root = Path(store.project_root)


# ---------------------------------------------------------------------------
# Gecachte Daten-Grundlagen: Ohne Cache liefen base_doc_table (liest die
# größte Korpus-Stufe) und stage_token_summary (liest ALLE gewählten Stufen)
# bei JEDER Widget-Interaktion dieser Seite neu – bei großen Korpora Minuten
# pro Klick. Der Datei-Stempel (mtime) invalidiert nach Pipeline-Läufen.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Lese Korpus-Stufe und zähle Tokens …")
def _base_doc_table_cached(root_str: str, stage: str, stamp: tuple,
                           _schema) -> "pd.DataFrame":
    return stats.base_doc_table(Path(root_str), _schema, stage=stage)


@st.cache_data(show_spinner="Zähle Tokens je Stufe …")
def _stage_summary_cached(root_str: str, stages: tuple, stamp: tuple,
                          _schema) -> "pd.DataFrame":
    return stats.stage_token_summary(Path(root_str), list(stages), _schema)


stop_csv = project_root / "output" / "processed_corpus" / "korpus_stop.csv"
if not stop_csv.exists():
    st.info("Voraussetzung: `output/processed_corpus/korpus_stop.csv`. Bitte "
            "zuerst das Korpus verarbeiten (Seite 'Korpus verarbeiten').")
    st.stop()

try:
    base = _base_doc_table_cached(
        str(project_root), "min",
        stats.stage_file_stamp(project_root, ["min"]), schema)
except Exception as exc:
    show_error(exc)
    st.stop()

# ---------------------------------------------------------------------------
# 1) Tokens je Stufe
# ---------------------------------------------------------------------------
st.subheader("1 · Tokens je Stufe")
avail = stats.available_stages(project_root)
cols = st.columns(max(1, len(avail)))
chosen_stages = [s for i, s in enumerate(avail)
                 if cols[i].checkbox(s, value=True, key=f"stage_{s}")]
if chosen_stages:
    try:
        tbl = _stage_summary_cached(
            str(project_root), tuple(chosen_stages),
            stats.stage_file_stamp(project_root, chosen_stages), schema)
        df_with_download(tbl, "tokens_je_stufe", key="stages")
    except Exception as exc:
        show_error(exc)
else:
    st.caption("Keine Stufe gewählt.")

# ---------------------------------------------------------------------------
# 2) Milestones
# ---------------------------------------------------------------------------
st.subheader("2 · Milestones")
st.caption("Die Dokumente werden chronologisch in N gleich große Token-Abschnitte "
           "geteilt; je Abschnitt werden Jahresspanne und Tokenzahl angezeigt "
           "(Basis: stop-Stufe).")
n_ms = st.number_input("Anzahl Meilensteine", min_value=1, max_value=200, value=5,
                       key="n_milestones")
try:
    total_tokens, ms_df = stats.token_milestones(base, int(n_ms))
    st.metric("Gesamttokens (min)", f"{total_tokens:,}".replace(",", "."))
    if ms_df.empty:
        st.info("Keine Jahresangaben gefunden – Milestones nicht berechenbar.")
    else:
        df_with_download(ms_df, "milestones", key="milestones")
except Exception as exc:
    show_error(exc)
    ms_df = None

# ---------------------------------------------------------------------------
# 3) Frequenz je Metadatenspalte
# ---------------------------------------------------------------------------
st.subheader("3 · Frequenz je Metadatenspalte")
st.caption("Pro gewählter Spalte: Dokumente und Tokens je Wert – in einer Grafik.")
meta_cols = stats.metadata_columns(base, schema)
if not meta_cols:
    st.caption("Keine Metadatenspalten gefunden.")
    chosen_cols = []
else:
    grid = st.columns(min(4, len(meta_cols)))
    chosen_cols = [c for i, c in enumerate(meta_cols)
                   if grid[i % len(grid)].checkbox(c, value=True, key=f"mcol_{c}")]

if st.button("Frequenz-Grafiken erstellen", type="primary", disabled=not chosen_cols):
    for c in chosen_cols:
        try:
            freq = stats.metadata_frequency(base, c)
            st.markdown(f"**{c}** — {len(freq)} Werte")
            fig = stats.plot_metadata_frequency(freq, c)
            save_figure(fig, f"frequenz_{c}", params={"Spalte": c}, key=f"freq_{c}")
            with st.expander(f"Tabelle: {c}"):
                df_with_download(freq, f"frequenz_{c}", key=f"freqtab_{c}")
        except Exception as exc:
            show_error(exc)

# ---------------------------------------------------------------------------
# 4) Nach output/statistics/ speichern (für Downstream, z. B. Tag-Topics)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("4 · Nach output/statistics/ speichern")
st.caption("Schreibt die gewählten Auswertungen als CSV (u. a. "
           "`year_count_tokens.csv`, das die Tag-Topics-Seite verwendet).")
if st.button("💾 Statistik-CSVs speichern"):
    try:
        outdir = project_root / "output" / "statistics"
        outdir.mkdir(parents=True, exist_ok=True)
        written = []
        if chosen_stages:
            _stage_summary_cached(
                str(project_root), tuple(chosen_stages),
                stats.stage_file_stamp(project_root, chosen_stages),
                schema).to_csv(
                outdir / "tokens.csv", index=False, encoding="utf-8")
            written.append("tokens.csv")
        stats.year_token_counts(base).to_csv(
            outdir / "year_count_tokens.csv", index=False, encoding="utf-8")
        written.append("year_count_tokens.csv")
        if ms_df is not None and not ms_df.empty:
            ms_df.to_csv(outdir / "milestones.csv", index=False, encoding="utf-8")
            written.append("milestones.csv")
        for c in chosen_cols:
            safe = "".join(ch if ch.isalnum() else "_" for ch in c)
            stats.metadata_frequency(base, c).to_csv(
                outdir / f"frequenz_{safe}.csv", index=False, encoding="utf-8")
            written.append(f"frequenz_{safe}.csv")
        st.success(f"Gespeichert in `output/statistics/`: {', '.join(written)}")
    except Exception as exc:
        show_error(exc)
