#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Topics taggen
====================

Zeigt die **vollständige Topic-Word-Matrix** eines Topic-Modells (alle Wörter je
Topic, horizontal scrollbar) und lässt je Topic einen **komplexen Namen** direkt
in der Topic-Spalte eintragen. Datengrundlage ist die Auswahl einer
Topic-Word-Matrix (Ausgabe von ``topic_model.py``/MALLET). Gespeichert wird in
den **Topic-Modell-Ordner** – die Namen wandern in die Topic-Word-Matrix
**und** die Document-Topic-Matrix (deren Topic-Spalten werden umbenannt).

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

st.caption("Es kann eine frische Topic-Word-Matrix **oder** eine bereits "
           "begonnene (getaggte) Matrix (`*_topic_words_tag_v*.csv`) geladen "
           "und weiterbearbeitet werden.")
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
                # Editor-Status eines früheren Ladevorgangs verwerfen.
                st.session_state.pop("topic_tag_editor", None)
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

# WICHTIG: Die Baseline (``df``) NICHT mit dem Editor-Ergebnis überschreiben.
# Der data_editor hält seine Änderungen unter key="topic_tag_editor"; würde man
# die Eingabe bei jedem Rerun durch die bereits editierte Fassung ersetzen,
# geraten Eingabe und Editor-Status in Konflikt und Eintragungen „springen"
# bzw. erscheinen erst nach erneutem Laden. Das fertige Ergebnis steht in
# ``edited`` und wird direkt für Fortschritt und Speichern verwendet.
edited = st.data_editor(
    df,
    use_container_width=True,
    height=540,
    hide_index=True,
    num_rows="fixed",   # Topics sind durch das Modell vorgegeben, kein Hinzufügen
    column_config=column_config,
    key="topic_tag_editor",
)

# ---------------------------------------------------------------------------
# 3) Speichern (Topic-Modell-Ordner, beide Matrizen)
# ---------------------------------------------------------------------------
st.subheader("3 · Speichern")

model_dir, model_name = topic_tagging.model_dir_and_name(
    st.session_state["topic_tag_src"])
st.caption(f"Zielordner: `{model_dir}` (Topic-Modell-Ordner). Die Namen werden "
           "in die Topic-Word-Matrix **und** die Document-Topic-Matrix "
           "geschrieben; der Versionszähler erhöht sich automatisch.")

tw_default = topic_tagging.next_tag_version_path(
    model_dir, model_name, "topic_words").name
tw_name = st.text_input("Dateiname Topic-Word-Matrix", value=tw_default,
                        key="tt_name_tw")

dist_paths = topic_tagging.find_document_topic_matrices(model_dir)
dist_choice = None
dt_name = None
if dist_paths:
    dist_choice = st.selectbox(
        "Document-Topic-Matrix (deren Topic-Spalten umbenannt werden)",
        [p.name for p in dist_paths], key="tt_dist")
    dt_default = topic_tagging.next_tag_version_path(
        model_dir, model_name, "document-topics-distribution").name
    dt_name = st.text_input("Dateiname Document-Topic-Matrix", value=dt_default,
                            key="tt_name_dt")
else:
    st.info("Keine Document-Topic-Matrix (`document-topics-distribution*.csv`) "
            "im Modell-Ordner gefunden – es wird nur die Topic-Word-Matrix "
            "gespeichert.")

if st.button("💾 Speichern", type="primary", key="tt_save"):
    try:
        ok, msg = topic_tagging.validate(edited)
        if not ok:
            st.error(msg)
        else:
            saved = topic_tagging.save_named(edited, model_dir / tw_name,
                                             topic_col=topic_col)
            parts = [f"Topic-Word-Matrix: `{saved.name}`"]

            if dist_choice:
                names_in_order = edited[topic_col].astype(str).tolist()
                out, n_renamed = topic_tagging.save_named_document_topics(
                    model_dir / dist_choice, names_in_order,
                    model_dir / dt_name)
                if n_renamed:
                    parts.append(f"Document-Topic-Matrix: `{out.name}` "
                                 f"({n_renamed} Topics umbenannt)")
                else:
                    parts.append("Document-Topic-Matrix übersprungen (noch "
                                 "keine Namen eingetragen).")

            done, total = topic_tagging.naming_progress(edited, topic_col)
            st.success("Gespeichert – " + " · ".join(parts)
                       + f"  ·  {done}/{total} Topics benannt.")
    except Exception as exc:
        show_error(exc)
