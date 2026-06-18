#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 10: POS-Tags verarbeiten
==============================

Verarbeitet eine getaggte POS-Liste mit dem vorhandenen Modul-3-Skript
``tt_s01_stop_pos_tag.py``: Tags an die TF-IDF-Rangliste anhängen,
kombinierte ``tag``-Spalte erzeugen, Tag-Statistiken/-Matrizen berechnen,
Tagset erstellen und nach TF-IDF sortieren.

Die Logik liegt in ``explorer_core.tag_processing`` (dünner Wrapper, der die
Skriptfunktionen unverändert aufruft). Die Schritte sind leichtgewichtig und
laufen direkt im Prozess; ihre Konsolenausgaben werden mitgeschnitten.
"""

from pathlib import Path

import streamlit as st

from ui_helpers import get_store, show_error
from explorer_core import tag_processing

st.set_page_config(page_title="POS-Tags verarbeiten", layout="wide")
st.title("🧩 Semantische Tags verarbeiten")
st.caption("Aus der getaggten POS-Liste Tag-Statistiken, Matrizen und ein "
           "nach TF-IDF sortiertes Tagset erzeugen (Modul 3).")

store = get_store()
project_root = Path(store.project_root)

# Vorhandensein des Skripts prüfen (verständliche Meldung, bevor es losgeht)
script = tag_processing.locate_script(project_root)
if script is None:
    st.error("`tt_s01_stop_pos_tag.py` wurde im Projektordner nicht "
             "gefunden. Bitte den Projektordner auf der Startseite prüfen.")
    st.stop()
st.caption(f"Verwendetes Skript: `{script.relative_to(project_root)}`")

# ---------------------------------------------------------------------------
# 1) Eingaben
# ---------------------------------------------------------------------------
st.subheader("1 · Eingaben")

# Getaggte POS-Dateien in resources/stop_pos_tag/ zur Auswahl anbieten
tag_dir = project_root / "resources" / "stop_pos_tag"
tagged_files = sorted(tag_dir.glob("*tag*.csv")) if tag_dir.exists() else []

if tagged_files:
    names = [p.name for p in tagged_files]
    chosen = st.selectbox("Getaggte POS-Datei (word, pos, count, tag1–tag3)", names,
                          index=len(names) - 1)  # neueste Version vorauswählen
    pos_file = tag_dir / chosen
else:
    st.warning("Keine getaggte POS-Datei in `resources/stop_pos_tag/` gefunden. "
               "Sie kann auf der Seite **POS-Liste taggen** erstellt werden.")
    pos_file = Path(st.text_input("Pfad zur getaggten POS-Datei", value=""))

tfidf_default = project_root / "output" / "tfidf_rank" / "tfidf-2000_vocab_rank.csv"
tfidf_file = Path(st.text_input("TF-IDF-Rangliste (erste Spalte = Wort)",
                                value=str(tfidf_default)))

output_dir = Path(st.text_input("Ausgabeordner",
                                value=str(project_root / "output" / "processed_tag")))

max_combo = st.number_input("max-combo-size (Größe der Tag-Kombinationen)",
                            min_value=1, max_value=5, value=3, step=1)

# ---------------------------------------------------------------------------
# 2) Modus
# ---------------------------------------------------------------------------
st.subheader("2 · Modus")

mode = st.radio(
    "Was soll ausgeführt werden?",
    ["Komplette Pipeline (empfohlen)",
     "Einzelschritt: Tags an TF-IDF anhängen",
     "Einzelschritt: tag1–tag3 zu 'tag' kombinieren",
     "Einzelschritt: Tag-Statistiken & Tagset",
     "Einzelschritt: Tagset nach TF-IDF sortieren"],
)

with st.expander("Was erzeugt die komplette Pipeline?"):
    st.markdown(
        "- `<basis>_tfidf_tagged.csv` — TF-IDF-Rangliste mit tag1–tag3\n"
        "- `<basis>_combined.csv` — POS-Liste mit kombinierter `tag`-Spalte\n"
        "- `<basis>_tagset.csv` und `<basis>_tagset_sorted.csv` — Tagset (roh / "
        "nach TF-IDF sortiert)\n"
        "- `tag_stats/…` — Häufigkeiten und Sparse-Matrizen\n\n"
        "`<basis>` ist der Dateiname der POS-Datei ohne `.csv` und ohne `_vN`."
    )

# ---------------------------------------------------------------------------
# 3) Ausführen
# ---------------------------------------------------------------------------
st.subheader("3 · Ausführen")

if st.button("▶️ Verarbeiten", type="primary"):
    try:
        with st.spinner("Verarbeite …"):
            if mode.startswith("Komplette"):
                log, created = tag_processing.run_full(
                    project_root=project_root,
                    pos_file=pos_file, tfidf_file=tfidf_file,
                    output_dir=output_dir, max_combo_size=int(max_combo),
                )
            elif "Tags an TF-IDF" in mode:
                out = output_dir / (Path(pos_file).stem + "_tfidf_tagged.csv")
                log, created = tag_processing.run_step(
                    project_root, "add-tags-to-tfidf",
                    pos_file=pos_file, tfidf_file=tfidf_file, output_file=out,
                )
            elif "kombinieren" in mode:
                out = output_dir / (Path(pos_file).stem + "_combined.csv")
                log, created = tag_processing.run_step(
                    project_root, "combine-tags",
                    input_file=pos_file, output_file=out,
                )
            elif "Tag-Statistiken" in mode:
                tagset_out = output_dir / (Path(pos_file).stem + "_tagset.csv")
                log, created = tag_processing.run_step(
                    project_root, "build-tag-stats",
                    input_file=pos_file, output_dir=output_dir / "tag_stats",
                    tagset_output=tagset_out, max_combo_size=int(max_combo),
                )
            else:  # Tagset sortieren
                tagset_in = output_dir / (Path(pos_file).stem + "_tagset.csv")
                out = output_dir / (Path(pos_file).stem + "_tagset_sorted.csv")
                log, created = tag_processing.run_step(
                    project_root, "sort-tagset",
                    tagset_file=tagset_in, tfidf_file=tfidf_file, output_file=out,
                )
    except Exception as exc:
        show_error(exc)
    else:
        st.success("Verarbeitung abgeschlossen.")
        if log.strip():
            st.code(log, language="text")
        if created:
            st.markdown("**Erzeugt:**")
            for p in created:
                st.markdown(f"- `{p}`")
        else:
            st.info("Es wurden keine neuen Dateien gefunden – bitte das Log "
                    "und die Eingabepfade prüfen.")
