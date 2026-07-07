#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Topics taggen
====================

Zeigt die **vollständige Topic-Word-Matrix** eines Topic-Modells (alle Wörter je
Topic, horizontal scrollbar) und lässt je Topic einen **komplexen Namen** direkt
in der Topic-Spalte eintragen. Datengrundlage ist die Auswahl einer
Topic-Word-Matrix (Ausgabe von ``topic_model.py``/MALLET). Gespeichert wird
versioniert nach ``resources/topic_names/``.

Die Rechenlogik liegt in ``explorer_core.topic_tagging`` – diese Datei baut nur
die Oberfläche.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui_helpers import get_store, show_error
from explorer_core import topic_tagging

st.set_page_config(page_title="Topics taggen", layout="wide")
st.title("🏷️ Topics taggen")
st.caption("Vollständige Topic-Word-Matrix ansehen (nach rechts scrollen für alle "
           "Wörter) und je Topic einen Namen in die Topic-Spalte eintragen.")

store = get_store()
project_root = Path(store.project_root)


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


# ---------------------------------------------------------------------------
# 1) Topic-Word-Matrix wählen
# ---------------------------------------------------------------------------
st.subheader("1 · Topic-Word-Matrix wählen")

matrix_path = _glob_select(
    "Topic-Word-Matrix (Zeilen = Topics, Spalten = Top-Wörter)",
    ["resources/topic-models/**/*topic_words*.csv",
     "resources/topic-models/**/*word*.csv",
     "resources/topic_names/*.csv"],
    key="tt_matrix",
)

if st.button("Laden", key="tt_load"):
    if not matrix_path or not Path(matrix_path).exists():
        st.warning("Bitte eine vorhandene Topic-Word-Matrix wählen.")
    else:
        try:
            df = topic_tagging.load_topic_table(
                Path(matrix_path), delimiter=store.schema.delimiter)
            ok, msg = topic_tagging.validate(df)
            if not ok:
                st.error(msg)
            else:
                st.session_state["topic_tag_df"] = df
                st.session_state["topic_tag_src"] = str(matrix_path)
                st.success(f"{len(df):,} Topics × {df.shape[1] - 1} Wörter geladen.")
        except Exception as exc:
            show_error(exc)

if "topic_tag_df" not in st.session_state:
    st.info("Bitte zuerst eine Topic-Word-Matrix laden.")
    st.stop()

df: pd.DataFrame = st.session_state["topic_tag_df"]
topic_col = topic_tagging.topic_id_column(df)

# ---------------------------------------------------------------------------
# 2) Benennen
# ---------------------------------------------------------------------------
st.subheader("2 · Topics benennen")

done, total = topic_tagging.naming_progress(df, topic_col)
st.progress(done / total if total else 0.0,
            text=f"{done} von {total} Topics benannt")
st.caption("Die Wortspalten sind gesperrt; **nach rechts scrollen**, um alle "
           f"{df.shape[1] - 1} Wörter je Topic zu sehen. In der Spalte "
           f"**`{topic_col}`** einen sprechenden Namen eintragen.")

# Topic-Spalte editierbar (Name), alle Wortspalten gesperrt + schmal → es
# entsteht eine breite Tabelle, die horizontal scrollt.
column_config = {
    topic_col: st.column_config.TextColumn(
        f"{topic_col} (Name eintragen)",
        help="Trage hier einen komplexen, sprechenden Namen für das Topic ein "
             "(z. B. 'Schulische Bildung & Erziehung'). Vorbelegt mit der "
             "Topic-Nummer – einfach überschreiben.",
        width="large"),
}
for col in topic_tagging.word_columns(df, topic_col):
    column_config[col] = st.column_config.TextColumn(col, disabled=True,
                                                     width="small")

edited = st.data_editor(
    df,
    use_container_width=True,
    height=540,
    hide_index=True,
    num_rows="fixed",   # Topics sind durch das Modell vorgegeben, kein Hinzufügen
    column_config=column_config,
    key="topic_tag_editor",
)
st.session_state["topic_tag_df"] = edited

# ---------------------------------------------------------------------------
# 3) Speichern (versioniert)
# ---------------------------------------------------------------------------
st.subheader("3 · Speichern")

target_dir = project_root / "resources" / "topic_names"
src_stem = Path(st.session_state["topic_tag_src"]).stem
suggested = topic_tagging.next_version_path(target_dir, base=f"{src_stem}_named")
st.caption(f"Zielordner: `{target_dir}` — wird nicht überschrieben, der "
           "Versionszähler erhöht sich automatisch.")

save_name = st.text_input("Dateiname", value=suggested.name, key="tt_name")
if st.button("💾 Speichern", type="primary", key="tt_save"):
    try:
        ok, msg = topic_tagging.validate(edited)
        if not ok:
            st.error(msg)
        else:
            saved = topic_tagging.save_named(edited, target_dir / save_name,
                                             topic_col=topic_col)
            done, total = topic_tagging.naming_progress(edited, topic_col)
            st.success(f"Gespeichert: {saved}  ·  {done}/{total} Topics benannt.")
    except Exception as exc:
        show_error(exc)
