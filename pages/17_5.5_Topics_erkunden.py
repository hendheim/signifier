#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 5: Topics
===============

Ersetzt den Tab "Topicverläufe" des alten Korpus-Explorers:

Die Topic-Dokument-Verteilung (Pipeline-Schritt tt_s04_dtti / DTTI) wird über
die Metadaten mit den Erscheinungsjahren verknüpft. Anschließend lassen sich
für ausgewählte Topics zwei Kennzahlen über die Zeit darstellen:

- **Mittlere Ähnlichkeit pro Jahr** – wie stark ist das Topic im Schnitt
  in den Texten eines Jahres vertreten?
- **Texte über Schwelle pro Jahr**  – in wie vielen Texten überschreitet die
  Topic-Ähnlichkeit einen Schwellenwert?

Beide Kennzahlen können roh, geglättet (gleitender Mittelwert) oder als
Polynom-Trend dargestellt werden.
"""

import streamlit as st

from ui_helpers import (get_store, get_schema, show_error, df_with_download,
                        parse_year_range)
from explorer_core.analysis_topics import (topics_with_years,
                                           topic_year_means,
                                           topic_threshold_counts,
                                           natural_key)
from explorer_core.analysis_terms import plot_trends
from explorer_core.viz_export import save_figure

st.set_page_config(page_title="Topics erkunden", layout="wide")
st.title("📈 Topics")
st.caption("Topic-Verläufe über die Zeit")

store = get_store()
schema = get_schema()

# ---------------------------------------------------------------------------
# Daten laden (gecacht in der Session, da das Jahr-Mapping etwas dauern kann)
# ---------------------------------------------------------------------------
try:
    if "topics_year_df" not in st.session_state:
        topics_dist = store.load_topics_dist()
        meta = store.load_metadata()
        st.session_state["topics_year_df"] = topics_with_years(
            topics_dist, meta, schema)
    topics_year_df = st.session_state["topics_year_df"]
except Exception as e:
    show_error(e)
    st.stop()

topic_cols = sorted(
    [c for c in topics_year_df.columns if c not in ("Jahr",)
     and topics_year_df[c].dtype.kind in "fi"],
    key=natural_key)

st.caption(f"{len(topics_year_df)} Texte mit Jahresangabe · "
           f"{len(topic_cols)} Topics verfügbar")

# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------
chosen_topics = st.multiselect("Topics auswählen", topic_cols,
                               default=topic_cols[:3], key="tp_topics")

c1, c2, c3, c4 = st.columns(4)
tp_kind = c1.selectbox("Kennzahl",
                       ["Mittlere Ähnlichkeit pro Jahr",
                        "Texte über Schwelle pro Jahr"], key="tp_kind")
tp_thresh = c2.slider("Schwelle", 0.0, 1.0, 0.2, 0.05, key="tp_thresh",
                      disabled=not tp_kind.startswith("Texte"), help='Schwelle, ab der ein Text als zu einem Topic gehörig gezählt wird (nur bei der Kennzahl Textzahl wirksam).')
tp_mode = c3.selectbox("Darstellung",
                       ["Roh", "Geglättet (gleitender Mittelwert)",
                        "Polynom-Trend"], key="tp_mode")
tp_years_raw = c4.text_input("Jahresbereich (z. B. 1800-1900, leer = alle)",
                             key="tp_years")

c5, c6 = st.columns(2)
tp_window = c5.number_input("Glättungsfenster (Jahre)", 2, 25, 5,
                            key="tp_win",
                            disabled=not tp_mode.startswith("Gegl"), help='Fensterbreite des gleitenden Mittelwerts in Jahren zur Glättung; größer = glatter.')
tp_degree = c6.number_input("Polynomgrad", 1, 10, 3, key="tp_deg",
                            disabled=not tp_mode.startswith("Poly"), help='Grad des Polynoms beim Polynom-Trend; höher = flexibler, aber überanpassungsanfälliger.')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
if st.button("Verläufe zeichnen", key="tp_btn", type="primary"):
    if not chosen_topics:
        st.warning("Bitte mindestens ein Topic auswählen.")
    else:
        try:
            if tp_kind.startswith("Mittlere"):
                df = topic_year_means(topics_year_df, chosen_topics)
                ylabel = "Mittlere Topic-Ähnlichkeit"
            else:
                df = topic_threshold_counts(topics_year_df, chosen_topics,
                                            threshold=float(tp_thresh))
                ylabel = f"Texte mit Ähnlichkeit ≥ {tp_thresh:.2f}"

            year_range = parse_year_range(tp_years_raw)
            if year_range:
                df = df.loc[(df.index >= year_range[0])
                            & (df.index <= year_range[1])]

            smooth = int(tp_window) if tp_mode.startswith("Gegl") else None
            poly = int(tp_degree) if tp_mode.startswith("Poly") else None
            fig = plot_trends(df, "Topic-Verläufe", ylabel,
                              smooth_window=smooth, poly_degree=poly)
            save_figure(fig, "topic_verlaeufe", params={
                "Topics": ", ".join(map(str, chosen_topics)),
                "Art": tp_kind, "Jahresbereich": tp_years_raw or "alle",
                "Darstellung": tp_mode, "Glättungsfenster": smooth,
                "Polynomgrad": poly}, key="tp")
            with st.expander("Datentabelle anzeigen"):
                df_with_download(df.reset_index(), "topic_verlaeufe_daten",
                                 key="tp_data")
        except Exception as e:
            show_error(e)
