#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 6: Tag-Topics
===================

Ersetzt den alten Tag-Topic-Explorer (gui_tag_topic_explorer.py):

- TT-Relevanz (Bubbles)   – Relevanz von Tags × Topics über gemeinsame
                            Terme, gewichtet mit TF-IDF-Summen
- Topics/Jahr (Stacked)   – gestapelte Jahresverteilung der Top-Topics
                            (+ gleitender Mittelwert)
- TT-Texte/Jahr (Polynom) – Polynom-Trends der Textzahlen pro Topic
- Tokens vs. Topics       – normalisierter Vergleich Korpusumfang vs.
                            Topic-Präsenz
- TT-Texte-Rang           – Top-Texte pro Topic, mit Metadaten verknüpft
"""

import streamlit as st

from ui_helpers import (get_store, get_schema, show_error, df_with_download,
                        parse_year_range)
from explorer_core.analysis_topics import (tag_topic_bubbles,
                                           stacked_topics_per_year,
                                           tt_texts_polynomial,
                                           tokens_vs_topics,
                                           tt_texts_rank)

from explorer_core.viz_export import save_figure

st.set_page_config(page_title="DTTI erkunden", layout="wide")
st.title("🔖 Document-Tag-Topic-Relationen")
st.caption("Beziehungen zwischen Termset-Tags und Topic-Modell")

store = get_store()
schema = get_schema()


def _require_data(df, label: str, quelle: str):
    """Leere Eingabedatei → verständliche Meldung statt Folge-Fehler.

    Seit die Nachverarbeitung alle Ausgabedateien garantiert erzeugt
    (notfalls leer mit Kopfzeile), zeigt eine leere Tabelle an, dass dort
    keine Jahre/Metadaten zugeordnet werden konnten.
    """
    if df is None or df.empty:
        raise ValueError(
            f"{label} ist leer. Vermutlich konnten bei der Nachverarbeitung "
            f"keine Jahre/Metadaten zugeordnet werden – bitte auf der Seite "
            f"{quelle} den Lauf wiederholen und die [INFO]-/[WARN]-Zeilen "
            "im Log prüfen (Dokument-IDs vs. Metadaten, Jahresspalte).")
    return df


tab_bubble, tab_stack, tab_poly, tab_tok, tab_rank = st.tabs(
    ["TT-Relevanz (Bubbles)", "Topics/Jahr (Stacked)",
     "TT-Texte/Jahr (Polynom)", "Tokens vs. Topics", "TT-Texte-Rang"]
)

# ---------------------------------------------------------------------------
# Tab: TT-Relevanz – Bubble-Chart Tags × Topics
# ---------------------------------------------------------------------------
with tab_bubble:
    st.subheader("Relevanz von Tags × Topics")
    st.caption("Bubble-Größe = Summe der TF-IDF-Werte aller Terme, die "
               "Tag und Topic gemeinsam haben.")
    c1, c2, c3 = st.columns(3)
    bb_topn = c1.number_input("Top-Topics (nach Rang)", 1, 50, 10,
                              key="bb_topn", help='Anzahl der Topics (nach Rang), die im Bubble-Diagramm berücksichtigt werden.')
    bb_minsize = c2.number_input("Min. Bubble-Größe", 1, 200, 20,
                                 key="bb_min", help='Mindestgröße einer Bubble; geringere Tag-Topic-Relevanzen werden ausgeblendet.')
    bb_values = c3.checkbox("Werte in Bubbles anzeigen", key="bb_vals")
    if st.button("Bubbles zeichnen", key="bb_btn", type="primary"):
        try:
            fig, df = tag_topic_bubbles(
                store.load_termset(), store.load_topic_words(),
                store.load_ranks(), store.tfidf_sums(),
                top_n=int(bb_topn), min_size=float(bb_minsize),
                show_values=bb_values)
            save_figure(fig, "tt_relevanz_bubbles", params={
                "Top-Topics": int(bb_topn), "Min. Bubble-Größe": float(bb_minsize),
                "Werte anzeigen": bool(bb_values)}, key="bb")
            with st.expander("Datentabelle anzeigen"):
                df_with_download(df, "tt_relevanz_daten", key="bb_data")
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: Topics/Jahr – gestapelte Jahresverteilung
# ---------------------------------------------------------------------------
with tab_stack:
    st.subheader("Topics pro Jahr (gestapelt)")
    st.caption("Gestapelte Textzahlen der Top-Topics pro Jahr, zusätzlich "
               "der gleitende 5-Jahres-Mittelwert der Gesamtsumme.")
    c1, c2 = st.columns(2)
    stk_topn = c1.number_input("Top-Topics", 1, 50, 10, key="stk_topn", help='Anzahl der häufigsten Topics in der gestapelten Jahresverteilung.')
    stk_years_raw = c2.text_input("Jahresbereich (z. B. 1800-1900, "
                                  "leer = alle)", key="stk_years")
    if st.button("Diagramm zeichnen", key="stk_btn", type="primary"):
        try:
            fig = stacked_topics_per_year(
                _require_data(store.load_counts_per_year(),
                              "Die Topic-Counts-pro-Jahr-Datei",
                              "„DTTI erstellen“ (Tab ‚DTTI nachverarbeiten‘)"),
                store.load_ranks(),
                year_range=parse_year_range(stk_years_raw),
                top_n=int(stk_topn))
            save_figure(fig, "topics_pro_jahr", params={
                "Top-Topics": int(stk_topn),
                "Jahresbereich": stk_years_raw or "alle"}, key="stk")
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: TT-Texte/Jahr – Polynom-Trends
# ---------------------------------------------------------------------------
with tab_poly:
    st.subheader("Texte pro Topic und Jahr (Polynom-Trend)")
    c1, c2 = st.columns(2)
    pl_degree = c1.number_input("Polynomgrad", 1, 12, 6, key="pl_deg", help='Grad des Polynom-Trends je Topic; höher = flexiblere Kurve, überanpassungsanfälliger.')
    pl_topn = c2.number_input("Top-Topics", 1, 50, 10, key="pl_topn", help='Anzahl der Top-Topics, die als Polynom-Trend dargestellt werden.')
    if st.button("Trends zeichnen", key="pl_btn", type="primary"):
        try:
            fig = tt_texts_polynomial(
                _require_data(store.load_counts_per_year(),
                              "Die Topic-Counts-pro-Jahr-Datei",
                              "„DTTI erstellen“ (Tab ‚DTTI nachverarbeiten‘)"),
                store.load_ranks(),
                degree=int(pl_degree), top_n=int(pl_topn))
            save_figure(fig, "tt_texte_polynom", params={
                "Polynomgrad": int(pl_degree), "Top-Topics": int(pl_topn)}, key="pl")
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: Tokens vs. Topics – normalisierter Vergleich
# ---------------------------------------------------------------------------
with tab_tok:
    st.subheader("Tokens vs. Topics")
    st.caption("Korpusumfang (Tokens pro Jahr) und Topic-Präsenz "
               "(Top-Dokumente pro Jahr), beide min-max-normalisiert und "
               "mit gleitendem 5-Jahres-Mittelwert geglättet.")
    tok_years_raw = st.text_input("Jahresbereich (z. B. 1800-1900, "
                                  "leer = alle)", key="tok_years")
    if st.button("Vergleich zeichnen", key="tok_btn", type="primary"):
        try:
            fig = tokens_vs_topics(
                store.load_tokens_year(),
                _require_data(store.load_top10_year_value(),
                              "Das Topic-Ranking pro Jahr",
                              "„Topics nachverarbeiten“"),
                year_range=parse_year_range(tok_years_raw))
            save_figure(fig, "tokens_vs_topics", params={
                "Jahresbereich": tok_years_raw or "alle"}, key="tok")
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: TT-Texte-Rang – Top-Texte pro Topic mit Metadaten
# ---------------------------------------------------------------------------
with tab_rank:
    st.subheader("Top-Texte pro Topic")
    st.caption("Die wichtigsten Texte je Topic (aus der Pipeline), "
               "über mehrere Strategien mit den Metadaten verknüpft.")
    rk_per_topic = st.number_input("Texte pro Topic", 5, 200, 30,
                                   key="rk_n", help='Anzahl der relevantesten Texte je Topic in der Rangliste.')
    if st.button("Rangliste erstellen", key="rk_btn", type="primary"):
        try:
            try:
                meta = store.load_metadata()
            except FileNotFoundError:
                meta = None
                st.info("Keine Metadaten gefunden – Rangliste ohne "
                        "Metadaten-Verknüpfung.")
            df, matched = tt_texts_rank(
                _require_data(store.load_top10_value_per_text(),
                              "Das Topic-Ranking pro Text",
                              "„Topics nachverarbeiten“"),
                meta, schema,
                per_topic=int(rk_per_topic))
            if meta is not None:
                st.caption(f"{matched} von {len(df)} Einträgen mit "
                           "Metadaten verknüpft.")
            df_with_download(df, "tt_texte_rang", key="rk")
        except Exception as e:
            show_error(e)
