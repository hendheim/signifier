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
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from explorer_core import (MetadataSchema, DataStore, ModelStore,
                           detect_project_root, read_csv_auto)
from explorer_core.token_index import TokenIndex

# Basisordner der App (= Ordner, in dem app.py liegt)
APP_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = APP_DIR / "config" / "metadata_schema.yaml"

_perf_log = logging.getLogger("signifier.perf")


@contextmanager
def timed(label: str):
    """Dauer eines Blocks messen: Log + dezente Anzeige unter dem Ergebnis."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _perf_log.info("%s: %.2fs", label, dt)
    st.caption(f"⏱ {label}: {dt:.2f} s")


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


@st.cache_data(show_spinner="Lade Daten …")
def _cached_csv(path_str: str, mtime_ns: int, delimiter: str,
                **kwargs) -> pd.DataFrame:
    """Prozessweiter CSV-Cache: überlebt Browser-Reloads und neue Sessions.

    ``mtime_ns`` ist Teil des Cache-Keys – nach einem Pipeline-Lauf wird
    die geänderte Datei automatisch neu gelesen.
    """
    return read_csv_auto(Path(path_str), delimiter=delimiter, **kwargs)


def _csv_reader(path: Path, delimiter: str, **kwargs) -> pd.DataFrame:
    return _cached_csv(str(path), Path(path).stat().st_mtime_ns,
                       delimiter, **kwargs)


def get_store() -> DataStore:
    """Zentraler DataStore (lebt über alle Seiten hinweg in der Session)."""
    if "store" not in st.session_state:
        root = detect_project_root(APP_DIR)
        store = DataStore(root, get_schema())
        store.reader = _csv_reader  # CSV-Reads über den Prozess-Cache
        st.session_state["store"] = store
    return st.session_state["store"]


@st.cache_resource(show_spinner="Baue Token-Index (einmalig pro Korpus) …")
def _token_index_cached(path_str: str, mtime_ns: int,
                        _corpus: pd.DataFrame) -> TokenIndex:
    # _corpus ist per Unterstrich vom Cache-Key ausgenommen; der Key ist
    # Pfad + mtime der Korpusdatei (Invalidierung nach Pipeline-Läufen).
    return TokenIndex.build(_corpus["doc_id"].astype(str).tolist(),
                            _corpus["text"].tolist())


def get_token_index() -> TokenIndex:
    """Token-Index über das geladene Korpus (einmal pro Prozess & Korpusdatei).

    Grundlage für Kollokationen und Dokument-Frequenzen ohne wiederholte
    Tokenisierung des gesamten Korpus.
    """
    store = get_store()
    corpus = store.load_corpus()
    path = Path(store.paths["corpus"])
    mtime = path.stat().st_mtime_ns if path.exists() else 0
    return _token_index_cached(str(path), mtime, corpus)


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


def render_fig_png(fig: plt.Figure) -> bytes:
    """PNG-Bytes (300 dpi) einer Figur – einmal gerendert, am Objekt gecacht.

    ``savefig`` mit 300 dpi kostet 0,4–1,5 s pro Figur; ohne Cache lief das
    bei jedem Streamlit-Rerun erneut (bei mehreren Dendrogrammen entsprechend
    mehrfach). Der Cache hängt an der Figur selbst, sodass persistierte
    Figuren (session_state) bei Reruns nichts mehr rendern müssen.
    """
    png = getattr(fig, "_signifier_png", None)
    if png is None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        png = buf.getvalue()
        fig._signifier_png = png
    return png


def fig_with_download(fig: plt.Figure, filename: str, key: str) -> None:
    """Matplotlib-Plot anzeigen + PNG-Download (300 dpi wie bisher).

    Angezeigt wird das einmal gerenderte 300-dpi-PNG (``st.image``) statt
    eines erneuten ``st.pyplot``-Renderings pro Rerun – identisches Bild,
    aber Reruns kosten praktisch nichts mehr.
    """
    png = render_fig_png(fig)
    st.image(png, width="stretch")
    st.download_button(
        "⬇️ PNG herunterladen",
        png,
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
