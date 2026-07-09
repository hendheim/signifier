#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 9: POS-Liste taggen
=========================

Ergänzt die POS-Frequenzliste der Stufe "stop" (aus ``s01_4_pos_tag.py``,
Spalten ``word, pos, count``) interaktiv um drei semantische Tag-Spalten
``tag1, tag2, tag3`` und speichert sie versioniert nach
``resources/stop_pos_tag/`` – im Format, das die Seite "POS-Tags verarbeiten"
(Modul 3) erwartet.

POS-Tags stammen aus spaCy (wie in s01_4), nicht aus HanTa. spaCy wird nur
für NEU ergänzte Wörter gebraucht; das Bearbeiten vorhandener Zeilen geht
ohne spaCy.

Die Rechenlogik liegt in ``explorer_core.tagging`` – diese Datei baut nur
die Oberfläche.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui_helpers import get_store, show_error
from explorer_core import tagging  # Modul mit der Tagging-Logik

st.set_page_config(page_title="Ausdrücke taggen", layout="wide")
st.title("🏷️ POS-Liste taggen")
st.caption("Top-5000-Ausdrücke der Stufe „stop“ in bis zu drei Kategorien "
           "(tag1/tag2/tag3) einordnen.")

store = get_store()
project_root = Path(store.project_root)

# Standardpfad der POS-Liste (Ausgabe von s01_4); auf der Startseite/Schema
# nicht hinterlegt, daher hier als anpassbares Feld.
default_pos = project_root / "output" / "vocabular" / "vocab_top5000_stop_pos.csv"

# ---------------------------------------------------------------------------
# 1) Liste laden
# ---------------------------------------------------------------------------
st.subheader("1 · POS-Liste laden")
st.caption("Frische Rohliste aus s01_4 laden **oder** eine bereits begonnene "
           "Tag-Liste aus `resources/stop_pos_tag/` weiterbearbeiten.")

resources_dir = project_root / "resources" / "stop_pos_tag"
started = tagging.existing_tag_lists(resources_dir)

# Auswahl: Rohliste + begonnene Tag-Listen (mit manueller Pfad-Alternative).
options: dict = {}
if default_pos.exists():
    options[f"🆕 Rohliste: {default_pos.name}"] = default_pos
for p in started:
    options[f"✏️ Begonnene Tag-Liste: {p.name}"] = p

sel_path = None
if options:
    sel_label = st.selectbox("Liste wählen", list(options.keys()),
                             key="tag_src_sel")
    sel_path = options[sel_label]

with st.expander("…oder eigenen Pfad angeben"):
    manual = st.text_input(
        "Pfad zur POS-/Tag-Liste (Spalten `word`, `pos`, `count`)",
        value="" if sel_path is not None else str(default_pos),
        key="tag_path_manual")

chosen_path = (Path(manual) if manual.strip()
               else (sel_path if sel_path is not None else default_pos))

load_clicked = st.button("Laden", use_container_width=False)

# Geladene Tabelle lebt in der Session, damit Edits beim Tagging erhalten bleiben.
if load_clicked:
    try:
        df = tagging.load_pos_list(Path(chosen_path),
                                   delimiter=store.schema.delimiter)
        ok, msg = tagging.validate(df)
        if not ok:
            st.error(msg)
        else:
            st.session_state["tag_df"] = df
            # Alten Editor-Status verwerfen, sonst würde der gespeicherte
            # Bearbeitungsstand (Deltas) auf die NEU geladene Liste angewandt.
            st.session_state.pop("tag_editor", None)
            st.session_state["tag_path"] = str(chosen_path)
            _done, _total = tagging.tagging_progress(df)
            st.success(f"{len(df):,} Zeilen geladen "
                       f"({_done} bereits getaggt).")
    except Exception as exc:
        show_error(exc)

if "tag_df" not in st.session_state:
    st.info("Bitte zuerst eine POS-Liste laden.")
    st.stop()

df: pd.DataFrame = st.session_state["tag_df"]

# ---------------------------------------------------------------------------
# 2) Vorhandene Tags (Wiederverwendung)
# ---------------------------------------------------------------------------
st.subheader("2 · Taggen")

vocab = tagging.existing_tags(df)
if vocab:
    st.caption("Bereits vergebene Tags (zum Wiederverwenden – einfach exakt so "
               "eintippen): " + ", ".join(f"`{t}`" for t in vocab))
else:
    st.caption("Noch keine Tags vergeben. Eigene Kategorien frei vergeben "
               "(z. B. Begriff, Gegenstand, Praktik).")

# Editierbare Tabelle. word/pos/count sind gesperrt, nur tag1–tag3 editierbar.
column_config = {
    "word": st.column_config.TextColumn("word", disabled=True),
    "pos": st.column_config.TextColumn("pos", disabled=True),
    "count": st.column_config.NumberColumn("count", disabled=True),
    "tag1": st.column_config.TextColumn("tag1", help="Erste semantische Kategorie für das Wort - ein selbst gewähltes Schlagwort, das es thematisch einordnet (z. B. Natur, Gefühl, Körper). Wörter mit demselben Tag bilden später ein gemeinsames Termset."),
    "tag2": st.column_config.TextColumn("tag2", help="Optionale zweite Kategorie, wenn das Wort zu mehreren Themen passt."),
    "tag3": st.column_config.TextColumn("tag3", help="Optionale dritte Kategorie."),
}

# WICHTIG: 'df' (Session-Baseline) muss über Reruns STABIL bleiben und darf NICHT
# mit dem Editor-Ergebnis überschrieben werden. Sonst bekäme der data_editor bei
# jedem Tastendruck neue Eingabedaten, würde sich neu aufbauen, den Zellfokus
# verlieren (Cursor springt nicht weiter / nicht zurück) und gerade getippte
# Werte teils verwerfen. Die Edits hält Streamlit unter key="tag_editor"; das
# fertige Ergebnis steht in 'edited'.
edited = st.data_editor(
    df,
    use_container_width=True,
    height=520,
    hide_index=True,
    num_rows="dynamic",   # erlaubt das Hinzufügen neuer Wörter
    column_config=column_config,
    key="tag_editor",
)

# Fortschritt aus dem aktuellen Bearbeitungsstand (nicht aus der Baseline).
done, total = tagging.tagging_progress(edited)
st.progress(done / total if total else 0.0,
            text=f"{done} von {total} Ausdrücken getaggt")

# ---------------------------------------------------------------------------
# 3) POS für neu ergänzte Wörter (spaCy)
# ---------------------------------------------------------------------------
with st.expander("POS-Tags für neu ergänzte Wörter mit spaCy bestimmen"):
    st.caption("Falls oben neue Zeilen ohne `pos` ergänzt wurden: spaCy "
               "(`de_core_news_lg`) bestimmt das Wortart-Tag – wie in s01_4. "
               "HanTa wird hier bewusst nicht verwendet.")
    if st.button("Fehlende POS-Tags ergänzen (spaCy)"):
        mask = edited["pos"].fillna("").astype(str).str.strip().eq("")
        new_words = edited.loc[mask, "word"].astype(str).tolist()
        if not new_words:
            st.info("Keine Zeilen ohne POS-Tag gefunden.")
        else:
            try:
                pos_map = tagging.pos_for_words(new_words)
                edited.loc[mask, "pos"] = edited.loc[mask, "word"].map(pos_map)
                # Programmatische Änderung: neue Baseline setzen UND den
                # Editor-Status verwerfen, sonst übernimmt der data_editor die
                # ergänzten POS-Tags nicht. Danach neu ausführen.
                st.session_state["tag_df"] = edited
                st.session_state.pop("tag_editor", None)
                st.success(f"{len(new_words)} POS-Tags ergänzt.")
                st.rerun()
            except ModuleNotFoundError:
                st.warning("spaCy ist nicht installiert. Bei Bedarf: "
                           "`pip install spacy` und das Modell "
                           "`de_core_news_lg` installieren.")
            except Exception as exc:
                show_error(exc)

# ---------------------------------------------------------------------------
# 4) Speichern (versioniert)
# ---------------------------------------------------------------------------
st.subheader("3 · Speichern")

resources_dir = project_root / "resources" / "stop_pos_tag"
suggested = tagging.next_version_path(resources_dir)
st.caption(f"Zielordner: `{resources_dir}` — wird nicht überschrieben, der "
           "Versionszähler erhöht sich automatisch.")

save_name = st.text_input("Dateiname", value=suggested.name)
if st.button("💾 Speichern", type="primary"):
    try:
        target = resources_dir / save_name
        ok, msg = tagging.validate(edited)
        if not ok:
            st.error(msg)
        else:
            saved = tagging.save_tagged(edited, target)
            done, total = tagging.tagging_progress(edited)
            st.success(f"Gespeichert: {saved}  ·  {done}/{total} Zeilen getaggt.")
            st.caption("Weiter geht es auf der Seite **POS-Tags verarbeiten**.")
    except Exception as exc:
        show_error(exc)
