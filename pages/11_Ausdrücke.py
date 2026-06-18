#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 1: Ausdrücke
==================

Bündelt alle term-bezogenen Analysen des alten Korpus-Explorers:

- Frequenz            (Tab "Frequenz")
- TF-IDF-Rang         (Tab "TF-IDF-Rang")
- Dokument-Frequenz   (Tab "Dokument-Frequenz")
- Konkordanz / KWIC   (Tab "Konkordanz")
- Kollokationen       (Tab "Kollokation", FREQ & PMI)
- Wortverläufe        (Tab "Wortverläufe", absolut & relativ pro Mio Tokens)

Die Rechenlogik liegt vollständig in ``explorer_core.analysis_terms`` –
diese Datei baut nur die Oberfläche.
"""

import streamlit as st

from ui_helpers import (get_store, get_schema, show_error, df_with_download,
                        fig_with_download, metadata_multiselect, parse_terms,
                        parse_year_range)
from explorer_core.analysis_terms import (
    term_frequencies, filter_tfidf_rank, term_overview, document_frequencies, concordance,
    collocations, collocation_documents, word_trends, plot_trends,
)

from explorer_core.viz_export import save_figure

st.set_page_config(page_title="Ausdrücke", layout="wide")
st.title("🔤 Ausdrücke")
st.caption("Frequenzen, TF-IDF, Konkordanzen, Kollokationen und Wortverläufe")

store = get_store()
schema = get_schema()

tab_overview, tab_docfreq, tab_kwic, tab_kollok, tab_trend = st.tabs(
    ["Frequenz & TF-IDF", "Dokument-Frequenz",
     "Konkordanz", "Kollokation", "Wortverläufe"]
)

# ---------------------------------------------------------------------------
# Tab: Frequenz & TF-IDF – Gesamtsuche (Häufigkeit + mittlerer TF-IDF)
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Gesamtsuche: Häufigkeit und TF-IDF")
    st.caption("freq = Gesamtzahl der Vorkommen im Korpus · "
               "tfidf_mean = mittlerer TF-IDF über alle Dokumente. "
               "Spaltenkopf anklicken zum Umsortieren.")
    col1, col2 = st.columns([3, 1])
    search_ov = col1.text_input(
        "Suche (mehrere Begriffe mit Komma trennen, leer = alle)",
        key="ov_search",
        placeholder="z. B. natur, wald, berg",
    )
    top_n_ov = col2.number_input("Max. Zeilen", 10, 10000, 500, 10,
                                 key="ov_topn")
    if st.button("Anzeigen", key="ov_btn", type="primary"):
        try:
            dtm = store.load_dtm()
            term_cols = store.term_columns(dtm)
            tfidf_avg = store.tfidf_averages()
            df = term_overview(dtm, term_cols, tfidf_avg,
                               search=parse_terms(search_ov) or None,
                               top_n=int(top_n_ov))
            st.caption(f"{len(df)} Ausdrücke gefunden.")
            df_with_download(df, "frequenz_tfidf", key="ov")
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: Dokument-Frequenz – in wie vielen / welchen Dokumenten kommt X vor?
# ---------------------------------------------------------------------------
with tab_docfreq:
    st.subheader("Dokument-Frequenz")
    st.caption("Zeigt, in welchen Dokumenten ein Ausdruck (oder mehrere) "
               "vorkommt – mit Trefferzahl (count), Textlänge in Tokens "
               "(text_laenge) und normalisierter Trefferdichte "
               "(pro_10k_tokens = count / Textlänge × 10 000).")
    terms_df_raw = st.text_input("Ausdrücke (Komma-getrennt)",
                                 key="docfreq_terms",
                                 placeholder="z. B. freiheit, gleichheit")
    c1, c2 = st.columns(2)
    use_regex = c1.checkbox("Als regulären Ausdruck interpretieren",
                            key="docfreq_regex")
    case_sens = c2.checkbox("Groß-/Kleinschreibung beachten",
                            key="docfreq_case")
    display_cols_df = metadata_multiselect("Anzeigespalten (Metadaten)",
                                           key="docfreq_cols")
    if st.button("Dokumente suchen", key="docfreq_btn", type="primary"):
        terms = parse_terms(terms_df_raw)
        if not terms:
            st.warning("Bitte mindestens einen Ausdruck eingeben.")
        else:
            try:
                corpus = store.load_corpus()
                meta = store.load_metadata()
                df = document_frequencies(corpus, meta, schema, terms,
                                          display_cols_df,
                                          use_regex=use_regex,
                                          case_sensitive=case_sens)
                st.caption(f"{len(df)} Dokument-Treffer.")
                df_with_download(df, "dokument_frequenz", key="docfreq")
            except Exception as e:
                show_error(e)

# ---------------------------------------------------------------------------
# Tab: Konkordanz – Keyword in Context (KWIC)
# ---------------------------------------------------------------------------
with tab_kwic:
    st.subheader("Konkordanz (Keyword in Context)")
    c1, c2, c3 = st.columns([3, 1, 1])
    kwic_term = c1.text_input("Suchbegriff", key="kwic_term")
    kwic_context = c2.number_input("Kontext (Zeichen)", 10, 300, 50, 10,
                                   key="kwic_ctx", help='Anzahl der Zeichen links und rechts des Treffers in der Konkordanz (KWIC).')
    kwic_max = c3.number_input("Max. Treffer", 100, 20000, 5000, 100,
                               key="kwic_max", help='Obergrenze der angezeigten Treffer, damit die Tabelle bei häufigen Wörtern handhabbar bleibt.')
    display_cols_kwic = metadata_multiselect("Anzeigespalten (Metadaten)",
                                             key="kwic_cols")
    if st.button("Konkordanz erstellen", key="kwic_btn", type="primary"):
        if not kwic_term.strip():
            st.warning("Bitte einen Suchbegriff eingeben.")
        else:
            try:
                corpus = store.load_corpus()
                meta = store.load_metadata()
                df = concordance(corpus, meta, kwic_term.strip(),
                                 display_cols_kwic,
                                 context=int(kwic_context),
                                 max_hits=int(kwic_max))
                st.caption(f"{len(df)} Belegstellen.")
                df_with_download(df, f"konkordanz_{kwic_term.strip()}",
                                 key="kwic")
            except Exception as e:
                show_error(e)

# ---------------------------------------------------------------------------
# Tab: Kollokation – häufige Nachbarn (FREQ) oder statistisch auffällige (PMI)
# ---------------------------------------------------------------------------
with tab_kollok:
    st.subheader("Kollokationen")
    st.caption("**FREQ** = einfache Häufigkeit im Fenster · "
               "**PMI** = Pointwise Mutual Information (überzufällige "
               "Verbindungen, auch bei seltenen Wörtern)")
    kol_terms_raw = st.text_input("Zielausdrücke (Komma-getrennt)",
                                  key="kol_terms")
    c1, c2, c3, c4, c5 = st.columns(5)
    kol_metric = c1.selectbox("Metrik", ["FREQ", "PMI"], key="kol_metric", help='Bewertung der Kollokate: FREQ = gemeinsame Häufigkeit, PMI = statistische Bindungsstärke (hebt seltene, aber spezifische Paare hervor).')
    kol_window = c2.number_input("Fenster (Wörter)", 1, 20, 5, key="kol_win", help='Fenstergröße in Wörtern links und rechts des Suchworts, in der Kollokate gezählt werden.')
    kol_ngram = c3.selectbox("N-Gramm der Kollokate", [1, 2, 3],
                             key="kol_ngram", help='Länge der Kollokate: 1 = einzelne Wörter, 2 oder 3 = Wortpaare bzw. -tripel.')
    kol_minfreq = c4.number_input("Min. Häufigkeit", 1, 100, 3,
                                  key="kol_minfreq", help='Mindesthäufigkeit, die ein Kollokat erreichen muss, um aufgenommen zu werden (filtert Rauschen).')
    kol_topn = c5.number_input("Top N", 10, 1000, 100, 10, key="kol_topn", help='Anzahl der stärksten Kollokate (nach Metrik sortiert), die angezeigt werden.')

    if st.button("Kollokationen berechnen", key="kol_btn", type="primary"):
        targets = parse_terms(kol_terms_raw)
        if not targets:
            st.warning("Bitte mindestens einen Zielausdruck eingeben.")
        else:
            try:
                corpus = store.load_corpus()
                df = collocations(corpus, targets, window=int(kol_window),
                                  top_n=int(kol_topn),
                                  min_freq=int(kol_minfreq),
                                  ngram=int(kol_ngram), metric=kol_metric)
                st.session_state["kol_result"] = df
                st.session_state["kol_params"] = dict(
                    targets=targets, window=int(kol_window),
                    ngram=int(kol_ngram))
            except Exception as e:
                show_error(e)

    # Ergebnis + Detailansicht (ersetzt den Doppelklick der Tkinter-GUI)
    if "kol_result" in st.session_state:
        df_kol = st.session_state["kol_result"]
        st.caption(f"{len(df_kol)} Kollokationen.")
        df_with_download(df_kol, "kollokationen", key="kol")

        if not df_kol.empty:
            st.markdown("##### Belegdokumente zu einer Kollokation")
            params = st.session_state["kol_params"]
            pair_options = [
                f"{row['Ziel']} ↔ {row['Kollokat']}"
                for _, row in df_kol.iterrows()
            ]
            chosen = st.selectbox("Kollokationspaar wählen", pair_options,
                                  key="kol_pair")
            display_cols_kol = metadata_multiselect(
                "Anzeigespalten (Metadaten)", key="kol_doc_cols")
            if st.button("Dokumente anzeigen", key="kol_docs_btn"):
                target, collocate = (p.strip() for p in chosen.split("↔"))
                try:
                    corpus = store.load_corpus()
                    meta = store.load_metadata()
                    df_docs = collocation_documents(
                        corpus, meta, target, collocate,
                        display_cols_kol,
                        window=params["window"], ngram=params["ngram"])
                    st.caption(f"{len(df_docs)} Dokumente mit dieser "
                               "Kollokation.")
                    df_with_download(df_docs, "kollokation_dokumente",
                                     key="kol_docs")
                except Exception as e:
                    show_error(e)

# ---------------------------------------------------------------------------
# Tab: Wortverläufe – Häufigkeit über die Jahre
# ---------------------------------------------------------------------------
with tab_trend:
    st.subheader("Wortverläufe über die Zeit")
    trend_terms_raw = st.text_input("Ausdrücke (Komma-getrennt)",
                                    key="trend_terms")
    c1, c2, c3, c4 = st.columns(4)
    trend_years_raw = c1.text_input("Jahresbereich (z. B. 1800-1900, "
                                    "leer = alle)", key="trend_years")
    trend_mode = c2.selectbox("Darstellung",
                              ["Roh", "Geglättet (gleitender Mittelwert)",
                               "Polynom-Trend"],
                              key="trend_mode")
    trend_window = c3.number_input("Glättungsfenster (Jahre)", 2, 25, 5,
                                   key="trend_win",
                                   disabled=not trend_mode.startswith("Gegl"), help='Fensterbreite des gleitenden Mittelwerts in Jahren zur Glättung des Verlaufs; größer = glatter.')
    trend_degree = c4.number_input("Polynomgrad", 1, 10, 3,
                                   key="trend_deg",
                                   disabled=not trend_mode.startswith("Poly"), help='Grad des Polynoms beim Polynom-Trend; höher = flexiblere Kurve, neigt aber zu Überanpassung.')
    trend_unit = st.radio("Einheit", ["Absolut", "Relativ (pro Mio. Tokens)"],
                          horizontal=True, key="trend_unit")

    if st.button("Verläufe zeichnen", key="trend_btn", type="primary"):
        terms = parse_terms(trend_terms_raw)
        if not terms:
            st.warning("Bitte mindestens einen Ausdruck eingeben.")
        else:
            try:
                dtm = store.load_dtm()
                term_cols = store.term_columns(dtm)
                df_abs, df_rel, missing = word_trends(
                    dtm, term_cols, schema, terms,
                    year_range=parse_year_range(trend_years_raw))
                if missing:
                    st.info("Nicht im Vokabular: " + ", ".join(missing))

                df_plot = df_abs if trend_unit == "Absolut" else df_rel
                ylabel = ("Häufigkeit (absolut)" if trend_unit == "Absolut"
                          else "Häufigkeit pro Mio. Tokens")
                smooth = (int(trend_window)
                          if trend_mode.startswith("Gegl") else None)
                poly = (int(trend_degree)
                        if trend_mode.startswith("Poly") else None)
                fig = plot_trends(df_plot, "Wortverläufe", ylabel,
                                  smooth_window=smooth, poly_degree=poly)
                _trend_params = {"Ausdrücke": ", ".join(terms),
                                 "Jahresbereich": trend_years_raw or "alle",
                                 "Einheit": trend_unit,
                                 "Darstellung": trend_mode,
                                 "Glättungsfenster": smooth,
                                 "Polynomgrad": poly}
                save_figure(fig, "wortverlaeufe", params=_trend_params, key="trend")
                with st.expander("Datentabelle anzeigen"):
                    df_with_download(df_plot.reset_index(),
                                     "wortverlaeufe_daten", key="trend_data")
            except Exception as e:
                show_error(e)
