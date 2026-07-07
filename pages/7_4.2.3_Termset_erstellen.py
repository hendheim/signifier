#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Termset erstellen
========================

Erstellt ein Termset aus der Tagset-Pivot-Tabelle der Seite
**Tags verarbeiten** (``output/processed_tag/<basisname>_tagset*.csv``,
Spalten = Tags bzw. Tag-Kombinationen, Zeilen = zugehörige Wörter).

Ablauf:
1. Pivot-Tabelle wählen (die sortierte ``*_tagset_sorted.csv`` wird
   bevorzugt vorgeschlagen).
2. Tags per Markierung auswählen (Checkbox-Spalte, wie beim Taggen der
   Ausdrücke) – mit Vorschau des entstehenden Termsets.
3. Unter frei wählbarem Namen nach ``resources/termsets/`` speichern.
   Das Format ist identisch mit der Pivot-Tabelle (Spalten = Tags,
   Zellen = Wörter) und wird von den Seiten **Termset-Vektoren erkunden**
   und **DTTI erstellen** direkt verstanden.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui_helpers import get_store, show_error, df_with_download
from explorer_core.data_store import read_csv_auto

st.set_page_config(page_title="Termset erstellen", layout="wide")
st.title("🧺 Termset erstellen")
st.caption("Tags aus der Pivot-Tabelle (Seite „Tags verarbeiten“) markieren "
           "und als Termset nach `resources/termsets/` speichern.")

store = get_store()
project_root = Path(store.project_root)

# ---------------------------------------------------------------------------
# 1) Pivot-Tabelle wählen
# ---------------------------------------------------------------------------
st.subheader("1 · Pivot-Tabelle wählen")

tag_dir = project_root / "output" / "processed_tag"
candidates = sorted(tag_dir.glob("*_tagset*.csv")) if tag_dir.exists() else []
if not candidates:
    st.info("Keine Tagset-Pivot-Tabelle gefunden (erwartet: "
            "`output/processed_tag/*_tagset*.csv`). Bitte zuerst die Seite "
            "**Tags verarbeiten** ausführen.")
    st.stop()

names = [c.name for c in candidates]
# Die TF-IDF-sortierte Fassung ist die übliche Arbeitsgrundlage.
default_idx = next((i for i, n in enumerate(names)
                    if n.endswith("_tagset_sorted.csv")), 0)
chosen_name = st.selectbox("Pivot-Tabelle", names, index=default_idx,
                           key="ts_source_select")
pivot_path = tag_dir / chosen_name

try:
    pivot = read_csv_auto(pivot_path).fillna("")
    pivot = pivot.apply(lambda col: col.astype(str).str.strip())
except Exception as exc:
    show_error(exc)
    st.stop()

tag_cols = [str(c) for c in pivot.columns]
if not tag_cols:
    st.warning("Die Pivot-Tabelle enthält keine Tag-Spalten.")
    st.stop()

# ---------------------------------------------------------------------------
# 2) Tags markieren
# ---------------------------------------------------------------------------
st.subheader("2 · Tags markieren")

def _build_overview() -> pd.DataFrame:
    rows = []
    for c in tag_cols:
        words = [w for w in pivot[c].tolist() if w]
        preview = ", ".join(words[:5]) + (" …" if len(words) > 5 else "")
        rows.append({"auswählen": False, "Tag": c,
                     "Wörter": len(words), "Beispiele": preview})
    return pd.DataFrame(rows)

# Baseline pro Quelldatei stabil halten (Muster wie auf der Tagging-Seite):
# Der data_editor bekommt über Reruns dieselben Eingabedaten; seine Edits
# liegen unter key='ts_editor'. Programmatische Änderungen (alle an/aus,
# neue Quelldatei) setzen die Baseline neu und verwerfen den Editor-Status.
_source_id = f"{pivot_path}|{pivot_path.stat().st_mtime_ns}"
if st.session_state.get("ts_source_id") != _source_id:
    st.session_state["ts_source_id"] = _source_id
    st.session_state["ts_overview"] = _build_overview()
    st.session_state.pop("ts_editor", None)

bcol1, bcol2, _ = st.columns([1, 1, 3])
if bcol1.button("Alle markieren", key="ts_all"):
    df_all = st.session_state["ts_overview"].copy()
    df_all["auswählen"] = True
    st.session_state["ts_overview"] = df_all
    st.session_state.pop("ts_editor", None)
    st.rerun()
if bcol2.button("Auswahl aufheben", key="ts_none"):
    df_none = st.session_state["ts_overview"].copy()
    df_none["auswählen"] = False
    st.session_state["ts_overview"] = df_none
    st.session_state.pop("ts_editor", None)
    st.rerun()

edited = st.data_editor(
    st.session_state["ts_overview"],
    use_container_width=True,
    hide_index=True,
    height=min(520, 60 + 35 * len(tag_cols)),
    column_config={
        "auswählen": st.column_config.CheckboxColumn(
            "auswählen", help="Tag in das Termset übernehmen."),
        "Tag": st.column_config.TextColumn("Tag", disabled=True),
        "Wörter": st.column_config.NumberColumn("Wörter", disabled=True),
        "Beispiele": st.column_config.TextColumn("Beispiele", disabled=True),
    },
    key="ts_editor",
)
chosen_tags = edited.loc[edited["auswählen"], "Tag"].astype(str).tolist()
st.caption(f"{len(chosen_tags)} von {len(tag_cols)} Tags markiert.")

# ---------------------------------------------------------------------------
# 3) Vorschau + Speichern
# ---------------------------------------------------------------------------
st.subheader("3 · Speichern")

if chosen_tags:
    termset = pivot[chosen_tags].copy()
    # Zeilen entfernen, die nach der Spaltenauswahl komplett leer sind.
    termset = termset.loc[~termset.eq("").all(axis=1)].reset_index(drop=True)
    with st.expander(f"Vorschau des Termsets ({len(chosen_tags)} Tags)",
                     expanded=False):
        df_with_download(termset, "termset_vorschau", key="ts_preview")
else:
    termset = None
    st.caption("Noch keine Tags markiert.")

target_dir = project_root / "resources" / "termsets"
name_raw = st.text_input("Name des Termsets", key="ts_name",
                         placeholder="z. B. Gegenstaende_v1")
overwrite = st.checkbox("Vorhandene Datei überschreiben", key="ts_overwrite")

if st.button("💾 Termset speichern", type="primary",
             disabled=termset is None or not name_raw.strip()):
    # Dateiname säubern (nur Buchstaben/Ziffern/-/_), Endung ergänzen.
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                   for ch in name_raw.strip())
    if not safe:
        st.warning("Bitte einen gültigen Namen eingeben.")
    else:
        out_path = target_dir / f"{safe}.csv"
        if out_path.exists() and not overwrite:
            st.error(f"`{out_path.name}` existiert bereits – anderen Namen "
                     "wählen oder „Überschreiben“ aktivieren.")
        else:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                termset.to_csv(out_path, index=False, encoding="utf-8")
                store.set_path("termset", out_path)  # direkt als aktives Termset
                st.success(f"Termset gespeichert: "
                           f"`resources/termsets/{out_path.name}` "
                           f"({len(chosen_tags)} Tags) – als aktives Termset "
                           "übernommen (Seite „Verarbeitetes Korpus laden“).")
            except Exception as exc:
                show_error(exc)
