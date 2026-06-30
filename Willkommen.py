#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signifier-Dashboard – Startseite
===============================

Webbasiertes Dashboard zur explorativen Korpusanalyse. Ersetzt die beiden
Tkinter-GUIs (Korpus-Explorer und Tag-Topic-Explorer) durch eine
gemeinsame Oberfläche mit konfigurierbarem Metadatenschema.

Start:  streamlit run Willkommen.py   (oder Doppelklick auf start_dashboard.bat/.sh)

Diese Seite begrüßt die Nutzer:innen, erklärt kurz die beiden Eingabedateien
und skizziert sie jeweils als Tabelle mit einem vollständigen Beispiel­dokument
(Marie von Ebner-Eschenbach: „Krambambuli", 1883).
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="signifier-Dashboard", page_icon="📖", layout="wide")

st.title("📖 *signifier-Dashboard*")

# ---------------------------------------------------------------------------
# Willkommen / Kurzerklärung
# ---------------------------------------------------------------------------
st.markdown(
    "Willkommen! *signifier* ist ein Werkzeug zur Verarbeitung von Korpora "
    "für **Distant Reading** und **Scalable Reading**.\n\n"
    "Zum Starten kannst du \n\n"
    "+ ein Korpus aus `.txt`-Dateien in eine `.csv`-Datenbank übertragen und die Metadaten selbst ergänzen,\n"
    "+ ein Korpus aus `.xml`-Dateien in eine `.csv`-Datenbank übertragen und die benötigten Metadaten auswählen\n"
    "+ oder ein Korpus in `korpus/` als `korpus.csv` und `metadaten.csv` laden.\n"
)

st.divider()
st.markdown("**Die Datenbank**\n\n"

    "`metadaten.csv` – die relevanten bibliografischen Angaben und andere Metadaten pro Text (eine Zeile je Dokument, *ohne* den Volltext)\n\n"
    "`korpus.csv` – dieselben Datensätze **plus** den Volltext in der Spalte `content`\n\n"
    "Beide Dateien teilen sich die Schlüsselspalte `id`, über die Metadaten "
    "und Texte einander zugeordnet werden. Welche Metadatenspalten es gibt, "
    "ist frei konfigurierbar (Seite **Metadatenschema**); fest erwartet wird "
    "nur `id` (eindeutig) und – im Korpus – `content`. Als Spaltentrenner "
    "sind `,` oder `;` möglich; das Dashboard erkennt das Trennzeichen "
    "automatisch."
)

st.divider()
st.markdown("**Beispiel**")

# Gemeinsame Beispielwerte (Marie von Ebner-Eschenbach: Krambambuli, 1883)
_id = "Ebner-Eschenbach_Krambambuli"
_title = "Krambambuli"
_prename = "Marie von"
_surname = "Ebner-Eschenbach"
_year = 1883
# Öffentlich (gemeinfrei); hier als Auszug, im echten Korpus steht der
# vollständige Text in dieser Zelle.
_content = ("Vorliebe empfindet der Mensch für allerlei Dinge und Wesen. "
            "Liebe, die echte, unvergängliche, die lernt er – wenn überhaupt "
            "– nur einmal kennen. […]")

# ---------------------------------------------------------------------------
# 1) metadata.csv
# ---------------------------------------------------------------------------
st.markdown("**1 · `korpus/metadata.csv`** — Metadaten je Dokument "
            "(ohne Volltext):")
meta_df = pd.DataFrame([{
    "id": _id,
    "title": _title,
    "author_prename": _prename,
    "author_surname": _surname,
    "year": _year,
}])
st.dataframe(meta_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 2) korpus.csv
# ---------------------------------------------------------------------------
st.markdown("**2 · `korpus/korpus.csv`** — dieselben Metadaten **plus** den "
            "Volltext in `content`:")
korpus_df = pd.DataFrame([{
    "id": _id,
    "title": _title,
    "author_prename": _prename,
    "author_surname": _surname,
    "year": _year,
    "content": _content,
}])
st.dataframe(korpus_df, hide_index=True, use_container_width=True)

st.caption("Die Spalte `content` enthält in der Datenbank alle Texte – hier aus Platzgründen nur der Anfang.")

st.divider()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.markdown(
    "Wenn Du Dein Korpus in eine Datenbank geladen oder eine fertige Datenbank hochgeladen hast, kannst Du das Korpus erkunden.\n\n"
    "Wähle links im Menü eine Seite:\n\n"
    "- **Verarbeitung:** Korpus verarbeiten · Semantisches Taggen · "
    "Semantische Tags verarbeiten\n"
    "- **Einrichtung:** Verarbeitetes Korpus laden · Metadatenschema\n"
    "- **Exploration:** Ausdrücke · Texte · Wort-Vektoren · Termset-Vektoren · "
    "Topics · Tag-Topics"
)
