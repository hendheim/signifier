#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 7: Schema
===============

Editor für das Metadatenschema (``config/metadata_schema.yaml``).

Hier wird festgelegt, welche Spalte der Metadatendatei welche Rolle spielt
(ID, Jahr, Anzeige, Facetten). So lässt sich das Dashboard ohne eine einzige
Zeile Code an ein neues Korpus mit anderen Spaltennamen anpassen.

Die Auswahl wird in die YAML-Datei geschrieben; danach werden Schema und
Daten-Caches neu geladen, sodass alle Seiten sofort die neue Zuordnung nutzen.
"""

import streamlit as st

from ui_helpers import (get_store, get_schema, reload_schema, show_error,
                        SCHEMA_PATH)

st.set_page_config(page_title="Schema", layout="wide")
st.title("🗂️ Metadatenschema fixieren")
st.caption(f"Konfigurationsdatei: `{SCHEMA_PATH}`")

store = get_store()
schema = get_schema()

# ---------------------------------------------------------------------------
# Metadaten laden – ohne sie kann nur die Roh-YAML bearbeitet werden
# ---------------------------------------------------------------------------
meta = None
try:
    meta = store.load_metadata()
except Exception:
    st.warning("Die Metadatendatei konnte nicht geladen werden. "
               "Bitte zuerst auf der **Startseite** den Projektordner und "
               "die Pfade einstellen. Unten kann die YAML-Datei trotzdem "
               "direkt bearbeitet werden.")

AUTO = "(automatisch erkennen)"
NONE = "(keine)"


def _role_selectbox(label: str, candidates: list, columns: list,
                    key: str, help_text: str = "") -> str:
    """Dropdown für eine Rollen-Zuordnung.

    Vorauswahl = erste Kandidatin aus dem Schema, die in den Metadaten
    existiert; sonst "automatisch erkennen".
    """
    options = [AUTO] + columns
    current = AUTO
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            current = lower_map[cand.lower()]
            break
    return st.selectbox(label, options, index=options.index(current),
                        key=key, help=help_text or None)


if meta is not None:
    cols = list(meta.columns)
    cfg = schema.to_dict()

    st.markdown("### 1. Rollen-Zuordnung")
    st.caption("Welche Spalte der Metadatendatei übernimmt welche Funktion? "
               "»automatisch erkennen« nutzt die eingebauten Kandidatenlisten.")

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_id = _role_selectbox(
            "Dokument-ID", cfg.get("id_candidates", []), cols, "sc_id",
            "Eindeutige Kennung jedes Textes (verknüpft Metadaten, DTM, "
            "Topics usw.)")
    with c2:
        sel_year_first = _role_selectbox(
            "Jahr (Erstausgabe, hat Vorrang)",
            cfg.get("year", {}).get("first_candidates", []), cols,
            "sc_yfirst",
            "Falls vorhanden, wird dieses Jahr bevorzugt verwendet.")
    with c3:
        sel_year_main = _role_selectbox(
            "Jahr (Haupt/Fallback)",
            cfg.get("year", {}).get("main_candidates", []), cols, "sc_ymain",
            "Wird verwendet, wenn das Erstausgabe-Jahr fehlt.")

    st.markdown("### 2. Anzeige & Gruppierung")
    sel_display = st.multiselect(
        "Voreingestellte Anzeigespalten (Tabellen, Hover, Legenden)",
        options=cols,
        default=[c for c in schema.display_columns(meta) if c in cols],
        key="sc_display")

    auto_facets = schema.facet_columns(meta)
    sel_facets = st.multiselect(
        "Facetten (kategoriale Spalten zum Einfärben/Gruppieren)",
        options=cols,
        default=[c for c in (cfg.get("facets") or auto_facets)
                 if c in cols],
        key="sc_facets",
        help="Sinnvoll sind Spalten mit wenigen verschiedenen Werten, "
             f"z. B. {', '.join(auto_facets[:4]) if auto_facets else 'Gattung, Textklasse'}.")

    st.markdown("### 3. Diachrone Plots")
    c4, c5 = st.columns(2)
    use_min_year = c4.checkbox(
        "Frühestes Jahr begrenzen", value=cfg.get("min_year") is not None,
        key="sc_use_min",
        help="Blendet Texte vor einem Stichjahr aus den diachronen (zeitlichen) "
             "Auswertungen aus. Nützlich, wenn einzelne sehr frühe Texte die "
             "Verlaufskurven verzerren. Aus = alle Jahre einbeziehen.")
    min_year_val = c5.number_input(
        "Frühestes Jahr", 0, 3000, int(cfg.get("min_year") or 1800),
        key="sc_min_year", disabled=not use_min_year)

    # -----------------------------------------------------------------------
    # Speichern: Auswahl VORNE in die Kandidatenlisten schreiben, damit sie
    # gewinnt, die Auto-Detection für andere Korpora aber erhalten bleibt.
    # -----------------------------------------------------------------------
    if st.button("💾 Schema speichern", type="primary", key="sc_save"):
        try:
            def _prepend(selection: str, candidates: list) -> list:
                if selection == AUTO:
                    return candidates
                rest = [c for c in candidates
                        if c.lower() != selection.lower()]
                return [selection] + rest

            schema.id_candidates = _prepend(
                sel_id, cfg.get("id_candidates", []))
            year_cfg = cfg.get("year", {})
            schema.year_first_candidates = _prepend(
                sel_year_first, year_cfg.get("first_candidates", []))
            schema.year_main_candidates = _prepend(
                sel_year_main, year_cfg.get("main_candidates", []))
            schema.display_columns_cfg = sel_display
            schema.facets_cfg = sel_facets
            schema.min_year = int(min_year_val) if use_min_year else None

            schema.save(SCHEMA_PATH)     # in YAML schreiben
            reload_schema()              # Schema + Caches neu laden
            # Seitenspezifische Zwischenstände verwerfen
            for k in ("topics_year_df", "tx_umap"):
                st.session_state.pop(k, None)
            st.success("Schema gespeichert. Alle Seiten verwenden ab jetzt "
                       "die neue Zuordnung.")
        except Exception as e:
            show_error(e)

    # -----------------------------------------------------------------------
    # Vorschau: Wie interpretiert das Schema die Metadaten aktuell?
    # -----------------------------------------------------------------------
    st.markdown("### Aktuelle Interpretation (Vorschau)")
    try:
        sch = get_schema()
        y_first, y_main = sch.find_year_columns(meta)
        info = {
            "Dokument-ID": sch.find_id_column(meta) or "– (Zeilennummer)",
            "Jahr (Erstausgabe)": y_first or "–",
            "Jahr (Haupt)": y_main or "–",
            "Anzeigespalten": ", ".join(sch.display_columns(meta)) or "–",
            "Facetten": ", ".join(sch.facet_columns(meta)) or "–",
            "Frühestes Jahr": sch.min_year or "kein Filter",
        }
        for k, v in info.items():
            st.markdown(f"- **{k}:** {v}")
    except Exception as e:
        show_error(e)

# ---------------------------------------------------------------------------
# Experten-Modus: YAML direkt bearbeiten
# ---------------------------------------------------------------------------
with st.expander("🔧 Experten-Modus: YAML direkt bearbeiten"):
    st.caption("Nur bei Bedarf – Änderungen hier überschreiben die Datei "
               "komplett.")
    try:
        raw = SCHEMA_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        raw = ""
    edited = st.text_area("metadata_schema.yaml", raw, height=400,
                          key="sc_yaml")
    if st.button("YAML speichern", key="sc_yaml_save"):
        try:
            import yaml
            yaml.safe_load(edited)  # Validierung vor dem Schreiben
            SCHEMA_PATH.write_text(edited, encoding="utf-8")
            reload_schema()
            for k in ("topics_year_df", "tx_umap"):
                st.session_state.pop(k, None)
            st.success("YAML gespeichert und neu geladen.")
        except Exception as e:
            st.error(f"Ungültige YAML-Datei – nicht gespeichert: {e}")
