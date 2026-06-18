#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: MALLET einrichten
========================

Halb-automatische Einrichtung von MALLET fuer das Topic-Modelling:
- prueft, ob Java vorhanden ist (installiert Java NICHT automatisch),
- laedt das MALLET-Binary herunter ODER nimmt eine lokal hochgeladene ZIP
  (offline), entpackt es und macht den Starter ausfuehrbar,
- merkt den Starter-Pfad in resources/mallet/mallet_path.txt, damit die
  Topic-Seite ihn findet.

Logik in explorer_core.mallet_setup.
"""

import tempfile
from pathlib import Path

import streamlit as st

from ui_helpers import get_store, show_error
from explorer_core import mallet_setup as ms

st.set_page_config(page_title="MALLET einrichten", layout="wide")
st.title("⚙️ MALLET einrichten")
st.caption("Laedt das MALLET-Binary und prueft Java. Java muss separat "
           "installiert sein (siehe Hinweise unten).")

store = get_store()
project_root = Path(store.project_root)
default_target = project_root / "resources" / "mallet"
config_file = default_target / "mallet_path.txt"

# --- Java-Status ---
java = ms.find_java()
if java:
    st.success(f"Java gefunden: {ms.java_version() or java}")
else:
    st.error("Keine Java-Laufzeit gefunden. MALLET benoetigt Java.")
    with st.expander("Java installieren (Anleitung)"):
        st.markdown(
            "- **Linux (Debian/Ubuntu):** `sudo apt install default-jre`\n"
            "- **macOS (Homebrew):** `brew install openjdk`\n"
            "- **Windows:** Adoptium Temurin JRE/JDK installieren "
            "(https://adoptium.net) und Java zum PATH hinzufuegen.\n\n"
            "Danach diese Seite neu laden.")

st.divider()

# --- MALLET-Quelle ---
st.subheader("MALLET-Binary beschaffen")
mode = st.radio("Quelle", ["Herunterladen (Netz noetig)",
                           "Lokale ZIP hochladen (offline)"], horizontal=True)

target_dir = Path(st.text_input("Zielordner", value=str(default_target)))
source = None
uploaded_zip_path = None

if mode.startswith("Herunterladen"):
    source = st.text_input("Download-URL", value=ms.DEFAULT_MALLET_URL,
                           help="Standard ist MALLET 2.0.8. Falls die URL nicht "
                                "mehr gilt, hier eine gueltige ZIP-URL eintragen.")
else:
    up = st.file_uploader("MALLET-ZIP", type=["zip"])
    if up is not None:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getvalue())
        uploaded_zip_path = tmp
        source = str(tmp)

if st.button("MALLET einrichten", type="primary"):
    if not source:
        st.warning("Bitte eine URL angeben oder eine ZIP hochladen.")
    else:
        try:
            bar = st.progress(0.0, text="Lade MALLET ...")

            def _progress(frac):
                bar.progress(min(1.0, frac), text=f"Lade MALLET ... {int(frac*100)} %")

            res = ms.setup_mallet(source=source, target_dir=target_dir,
                                  progress=_progress if mode.startswith("Herunter") else None)
            bar.progress(1.0, text="Fertig")
            if res["mallet_path"]:
                config_file.parent.mkdir(parents=True, exist_ok=True)
                config_file.write_text(res["mallet_path"], encoding="utf-8")
                st.success(f"MALLET eingerichtet: `{res['mallet_path']}`")
                st.caption(f"Pfad gemerkt in `{config_file.relative_to(project_root)}` "
                           f"- die Topic-Seite kann MALLET nun nutzen.")
            else:
                st.error("Starter `bin/mallet` nicht gefunden - ist die ZIP korrekt?")
        except Exception as e:
            show_error(e)

st.divider()

# --- Status / Test ---
st.subheader("Status")
stt = ms.mallet_status(target_dir)
col1, col2 = st.columns(2)
col1.metric("Java", "OK" if stt["java"] else "fehlt")
col2.metric("MALLET", "OK" if stt["mallet_path"] else "fehlt")
if stt["mallet_path"]:
    st.caption(f"Starter: `{stt['mallet_path']}`")
if stt["ready"]:
    st.success("Bereit fuer das Topic-Modelling mit MALLET.")
else:
    st.info("Noch nicht vollstaendig: Java und/oder MALLET-Binary fehlen.")
