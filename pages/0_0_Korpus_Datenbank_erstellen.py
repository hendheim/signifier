#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Korpus Datenbank erstellen
=================================

Stellt aus einzelnen Dateien ein Korpus zusammen und speichert es nach
``/korpus`` als ``korpus.csv`` (Metadaten + Volltext) und ``metadaten.csv``
(nur Metadaten) – UTF-8, komma-separiert. Logik in
``explorer_core.corpus_build`` (an den Notebooks 00_01/00_02 ausgerichtet).

- **Aus txt-Dateien** (Notebook 00_01, Teil B): TXT-Dateien werden über eine
  ID-Spalte (Dateiname = ID) mit einer Metadaten-CSV zusammengeführt. Ohne
  Metadaten-CSV entsteht ein minimales Korpus (``id`` + ``content``).
- **Aus xml-Dateien** (TEI-Notebook 00_02): Metadaten per XPath aus dem
  TEI-Header, Fließtext aus ``.//tei:text//tei:body`` (Fallback ``.//tei:text``).
  Die Ziel-XML-Pfade sind als Metadaten-Kategorien frei definierbar.

Die Metadaten werden vor dem Speichern angezeigt und sind bearbeitbar.
Navigation: direkt nach **Willkommen** (Dateipräfix ``0_``).
"""

import io
from pathlib import Path

import streamlit as st
import pandas as pd

from ui_helpers import get_store, show_error
from explorer_core import corpus_build as cb

st.set_page_config(page_title="Korpus Datenbank erstellen", layout="wide")
st.title("🗂️ Korpus Datenbank erstellen")
st.caption("Aus txt- oder xml-Dateien ein Korpus bauen und nach /korpus "
           "speichern (korpus.csv + metadaten.csv, UTF-8, komma-separiert).")

store = get_store()
korpus_dir = Path(store.project_root) / "korpus"
st.caption(f"Zielordner: `{korpus_dir}`")


def _decode(upload) -> str:
    data = upload.getvalue()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _edit_and_save(corpus_df: pd.DataFrame, state_key: str,
                   id_column: str = "id"):
    st.markdown(f"**{len(corpus_df)} Dokumente** zusammengestellt.")

    # Read-only Überblick: ID + Volltext-Vorschau. Zeigt, dass die Texte
    # tatsächlich geladen wurden. (Die editierbare Tabelle unten enthält nur die
    # Metadaten – der Volltext wird beim Speichern per ID ergänzt und würde die
    # metadaten.csv aufblähen, wenn er dort editierbar wäre.)
    if cb.CONTENT_COLUMN in corpus_df.columns and not corpus_df.empty:
        key_col = id_column if id_column in corpus_df.columns else cb.ID_COLUMN
        cols = [key_col] if key_col in corpus_df.columns else []
        prev = corpus_df[cols + [cb.CONTENT_COLUMN]].copy()
        prev[cb.CONTENT_COLUMN] = (prev[cb.CONTENT_COLUMN].astype(str)
                                   .str.replace(r"\s+", " ", regex=True)
                                   .str.slice(0, 300))
        n_leer = int((prev[cb.CONTENT_COLUMN].str.strip() == "").sum())
        hinweis = (f" — ⚠️ {n_leer} ohne Text: ID-Spalte der CSV muss dem "
                   "Dateinamen (ohne .txt) entsprechen") if n_leer else ""
        st.caption(f"Geladene Volltexte (Vorschau, gekürzt){hinweis}:")
        st.dataframe(prev.rename(columns={cb.CONTENT_COLUMN: "content (Vorschau)"}),
                     use_container_width=True, hide_index=True, height=240)

    meta_view = cb.split_corpus_metadata(corpus_df)
    st.caption("Metadaten prüfen und ggf. bearbeiten (der Volltext wird beim "
               "Speichern per ID ergänzt):")
    edited = st.data_editor(meta_view, use_container_width=True,
                            num_rows="dynamic", key=f"{state_key}_editor")
    with st.expander("Vorschau Volltext (erstes Dokument)"):
        if not corpus_df.empty:
            st.text(str(corpus_df.iloc[0].get(cb.CONTENT_COLUMN, ""))[:1000])
    if st.button("💾 Nach /korpus speichern", key=f"{state_key}_save",
                 type="primary"):
        try:
            kp, mp = cb.write_corpus(corpus_df, korpus_dir,
                                     metadata_df=edited, id_column=id_column)
            st.success(f"Gespeichert: `{kp.name}` ({len(corpus_df)} Texte) "
                       f"und `{mp.name}` in {korpus_dir}")
        except Exception as e:
            show_error(e)


tab_txt, tab_xml = st.tabs(["Aus txt-Dateien", "Aus xml-Dateien"])

# ===========================================================================
# TXT  (Metadaten-CSV + TXT-Dateien zusammenführen)
# ===========================================================================
with tab_txt:
    st.markdown("**Metadaten-CSV (optional)** – verbindet Texte über eine "
                "ID-Spalte (Dateiname = ID).")
    c1, c2 = st.columns([2, 1])
    meta_upload = c1.file_uploader("Metadaten-CSV", type=["csv"],
                                   key="txt_meta_csv")
    sep_label = c2.selectbox("CSV-Trenner", [", (Komma)", "; (Semikolon)",
                                             "\\t (Tab)"], key="txt_sep")
    sep = {", (Komma)": ",", "; (Semikolon)": ";", "\\t (Tab)": "\t"}[sep_label]
    id_column = c1.text_input("ID-Spalte in der CSV", value="id",
                              key="txt_idcol")

    txt_uploads = st.file_uploader("txt-Dateien wählen", type=["txt"],
                                   accept_multiple_files=True, key="txt_up")
    if st.button("Korpus zusammenstellen", key="txt_build", type="primary"):
        if not txt_uploads:
            st.warning("Bitte zuerst txt-Dateien auswählen.")
        else:
            try:
                items = [(u.name, _decode(u)) for u in txt_uploads]
                meta_df = None
                if meta_upload is not None:
                    meta_df = pd.read_csv(io.StringIO(_decode(meta_upload)),
                                          sep=sep, dtype=str)
                df, unmatched = cb.build_from_txt(items, metadata_df=meta_df,
                                                  id_column=id_column)
                st.session_state["corpus_txt"] = df
                st.session_state["corpus_txt_idcol"] = id_column
                if unmatched:
                    st.warning(f"{len(unmatched)} txt-Datei(en) ohne "
                               f"Entsprechung in der CSV: "
                               f"{', '.join(unmatched[:10])}"
                               + (" …" if len(unmatched) > 10 else ""))
            except Exception as e:
                show_error(e)
    if "corpus_txt" in st.session_state:
        _edit_and_save(st.session_state["corpus_txt"], "txt",
                       id_column=st.session_state.get("corpus_txt_idcol", "id"))

# ===========================================================================
# XML  (TEI-orientiert)
# ===========================================================================
with tab_xml:
    st.caption("Der **Body** wird automatisch als `content` übernommen "
               "(`.//tei:text//tei:body`, Fallback `.//tei:text`). Die "
               "Metadaten wählst du unten über die im `teiHeader` vorhandenen "
               "Pfadziele – inkl. Attributwerten wie dem Datum in "
               "`<date when=\"…\">` (z. B. für Nachname, Vorname, "
               "Veröffentlichungsdatum).")

    use_ns = st.checkbox("TEI-Namespace verwenden (tei:)", value=True,
                         key="xml_ns",
                         help="TEI-XML-Dateien ordnen ihre Elemente meist einem "
                              "Namensraum zu (tei:). An (Standard): die Suchpfade "
                              "verwenden das Präfix 'tei:' - für reguläre TEI-Dateien "
                              "richtig. Aus: nur nötig, wenn die Pfade unten keine "
                              "Treffer liefern (Datei ohne Namensraum).")
    id_xpath = st.text_input("XPath für die ID (leer = Dateiname)",
                             value=cb.DEFAULT_ID_XPATH, key="xml_idxp",
                             help="Legt fest, woher die eindeutige Dokument-ID "
                                  "gelesen wird (als XPath-Ausdruck). Standard "
                                  "'./@xml:id' liest das xml:id-Attribut des "
                                  "Wurzelelements; fehlt es, wird der Dateiname als ID "
                                  "verwendet. Feld leer lassen = immer der Dateiname.")

    xml_uploads = st.file_uploader("xml-Dateien wählen", type=["xml"],
                                   accept_multiple_files=True, key="xml_up")

    # 1) Vorhandene Pfadziele ermitteln – vereinigt über mehrere Dateien, damit
    #    auch in der ersten Datei fehlende Felder (z. B. ein anonymer Erstautor)
    #    auswählbar werden. Attributwerte (etwa Datum in <date when="…">) sind
    #    ebenfalls dabei.
    if st.button("🔍 XML-Pfade analysieren", key="xml_scan"):
        if not xml_uploads:
            st.warning("Bitte zuerst xml-Dateien auswählen.")
        else:
            try:
                texts = [_decode(u) for u in xml_uploads[:25]]
                st.session_state["xml_opts"] = cb.discover_xml_paths_multi(
                    texts, use_namespace=use_ns)
                st.session_state.setdefault("xml_n_meta", 3)
            except Exception as e:
                show_error(e)

    opts = st.session_state.get("xml_opts")
    if opts:
        paths = [p for p, _ in opts]
        sample = {p: s for p, s in opts}
        st.markdown("**Metadaten über Pfadziele auswählen** "
                    f"({len(paths)} Pfade gefunden):")

        n_meta = st.session_state.setdefault("xml_n_meta", 3)
        rows = []
        for i in range(n_meta):
            col_p, col_n = st.columns([4, 2])
            sel = col_p.selectbox(
                f"Pfadziel {i + 1}", paths, key=f"xml_path_{i}",
                format_func=lambda p: f"{cb.path_label(p)}  ·  {sample.get(p, '')[:30]}",
                help="Der volle Pfad ist sichtbar – so lassen sich mehrere "
                     "gleichnamige Ziele (z. B. mehrere 'surname' an "
                     "verschiedenen Stellen) anhand des Pfades unterscheiden "
                     "und vergleichen.")
            name = col_n.text_input("Spaltenname", value=cb.path_leaf(sel),
                                    key=f"xml_name_{i}")
            if name.strip() and sel:
                rows.append((name.strip(), sel))

        cadd, cdel = st.columns(2)
        if cadd.button("➕ Metadatum hinzufügen", key="xml_addmeta"):
            st.session_state["xml_n_meta"] = n_meta + 1
            st.rerun()
        if cdel.button("➖ Letztes entfernen", key="xml_delmeta",
                       disabled=n_meta <= 1):
            st.session_state["xml_n_meta"] = max(1, n_meta - 1)
            st.rerun()

        if st.button("Korpus zusammenstellen", key="xml_build",
                     type="primary"):
            try:
                meta_xpaths = {name: path for name, path in rows}
                if not meta_xpaths:
                    st.warning("Bitte mindestens ein Metadatum auswählen.")
                else:
                    items = [(u.name, _decode(u)) for u in xml_uploads]
                    st.session_state["corpus_xml"] = cb.build_from_xml(
                        items, meta_xpaths=meta_xpaths,
                        content_xpaths=None,        # None = Body-Automatik
                        use_namespace=use_ns,
                        id_xpath=id_xpath.strip())
            except Exception as e:
                show_error(e)

    if "corpus_xml" in st.session_state:
        _edit_and_save(st.session_state["corpus_xml"], "xml")
