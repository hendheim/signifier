#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui_helpers
==========

Gemeinsame Streamlit-Hilfsfunktionen für alle Dashboard-Seiten:
- zentraler Zugriff auf DataStore/ModelStore/Schema (Session-State)
- Download-Buttons für Tabellen (CSV) und Plots (PNG)
- einheitliche Fehleranzeige

Die eigentliche Logik liegt in ``explorer_core`` – diese Datei kümmert sich
nur um die "Verkabelung" mit Streamlit.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from explorer_core import MetadataSchema, DataStore, ModelStore, detect_project_root

# Basisordner der App (= Ordner, in dem app.py liegt)
APP_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = APP_DIR / "config" / "metadata_schema.yaml"


def get_schema() -> MetadataSchema:
    """Metadatenschema einmal pro Sitzung laden (Seite 'Schema' kann es neu laden)."""
    if "schema" not in st.session_state:
        st.session_state["schema"] = MetadataSchema(SCHEMA_PATH)
    return st.session_state["schema"]


def reload_schema() -> MetadataSchema:
    """Schema nach dem Speichern neu einlesen und Daten-Caches invalidieren."""
    st.session_state["schema"] = MetadataSchema(SCHEMA_PATH)
    if "store" in st.session_state:
        st.session_state["store"].schema = st.session_state["schema"]
        st.session_state["store"].invalidate()
    return st.session_state["schema"]


def get_store() -> DataStore:
    """Zentraler DataStore (lebt über alle Seiten hinweg in der Session)."""
    if "store" not in st.session_state:
        root = detect_project_root(APP_DIR)
        st.session_state["store"] = DataStore(root, get_schema())
    return st.session_state["store"]


def get_models() -> ModelStore:
    """ModelStore für das Word2Vec-Modell."""
    if "models" not in st.session_state:
        st.session_state["models"] = ModelStore(get_store().paths["w2v_model"])
    return st.session_state["models"]


# ----------------------------------------------------------------------------
# Anzeige-Bausteine
# ----------------------------------------------------------------------------

def show_error(exc: Exception) -> None:
    """Fehler einheitlich und laienfreundlich anzeigen."""
    st.error(f"⚠️ {exc}")
    st.caption("Tipp: Prüfe auf der Startseite, ob alle Dateipfade stimmen, "
               "und auf der Seite **Schema**, ob die Spaltenzuordnung passt.")


def df_with_download(df: pd.DataFrame, filename: str, key: str,
                     height: int | None = None) -> None:
    """Tabelle anzeigen (sortierbar, durchsuchbar) + CSV-Download.

    Ersetzt die Treeviews der Tkinter-GUIs inkl. Sortieren/Kopieren/Export.
    """
    st.dataframe(df, use_container_width=True, hide_index=True, **({"height": height} if height is not None else {}))
    st.download_button(
        "⬇️ CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{filename}.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def fig_with_download(fig: plt.Figure, filename: str, key: str) -> None:
    """Matplotlib-Plot anzeigen + PNG-Download (300 dpi wie bisher)."""
    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "⬇️ PNG herunterladen",
        buf.getvalue(),
        file_name=f"{filename}.png",
        mime="image/png",
        key=f"dl_{key}",
    )
    plt.close(fig)  # Speicher freigeben (wichtig bei vielen Plots)


def metadata_multiselect(label: str, key: str, n_default: int = 3) -> list[str]:
    """Auswahl der Anzeigespalten – Defaults kommen aus dem Schema.

    Ersetzt die hartkodierten Dropdowns (author_surname/title/year_final)
    der Legacy-GUIs durch eine schema-gesteuerte Auswahl.
    """
    store, schema = get_store(), get_schema()
    try:
        meta = store.load_metadata()
        options = schema.selectable_metadata(meta)
        defaults = schema.display_columns(meta, n=n_default)
    except FileNotFoundError:
        options, defaults = [], []
    return st.multiselect(label, options=options, default=defaults, key=key)


def parse_terms(raw: str) -> list[str]:
    """Komma-getrennte Eingabe in eine Begriffsliste verwandeln."""
    return [t.strip() for t in (raw or "").split(",") if t.strip()]


def parse_year_range(raw: str) -> tuple[int, int] | None:
    """'1780-1900' → (1780, 1900); leere/ungültige Eingabe → None.

    Akzeptiert auch Float-/Komma-Schreibweisen wie '1780.0-1900.0' oder
    '1780,0-1900,0' sowie den Halbgeviertstrich '–', damit die Eingabe nicht
    still durchfällt, wenn Jahre als 1786.0 angezeigt werden.
    """
    raw = (raw or "").strip().replace("–", "-")
    if "-" in raw:
        try:
            lo, hi = (int(float(p.strip().replace(",", "."))) for p in raw.split("-", 1))
            if lo <= hi:
                return lo, hi
        except ValueError:
            pass
    return None
