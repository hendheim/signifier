#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Topic-Modell erstellen (scikit-learn)
============================================

Erstellt ein Topic-Modell mit scikit-learn. Zwei Eingabewege:

1. Korpus-Text + Chunking (empfohlen fuer Romane): zerlegt jeden Text in
   Wortfenster, vektorisiert (TF-IDF fuer NMF, Zaehlungen fuer LDA) und fittet
   das Modell auf den Segmenten. Das behebt LDA-Rauschen (zu wenige Dokumente)
   und die NMF-Rang-Grenze (min #Dokumente/#Features). Optional werden die
   Segment-Topic-Verteilungen pro Ursprungstext gemittelt.
2. Vorhandene DTM/TF-IDF-Matrix (aus s03_dtm_tfidf).

Die Ausgaben passen direkt in die Topic-Weiterverarbeitung (s02/s03):
- document-topics-distribution_<name>.csv (Zeilen = Dok/Segment-IDs, Spalten = Topics)
- <name>_topic_words.csv (Zeilen = Topics, Spalten = Rang, Zellen = Woerter)
- <name>_hyperparams.txt (alle verwendeten Einstellungen)

Diese Seite gehoert VOR 'Topics & Termset-Topics verarbeiten'.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from ui_helpers import get_store, show_error, df_with_download
from explorer_core import topic_model as tm
from explorer_core import mallet_runner as mr
from explorer_core import topic_metrics as tmet
from explorer_core import corpus_segment as cseg

st.set_page_config(page_title="Topic-Modell erstellen", layout="wide")
st.title("🧠 Topic-Modell erstellen")

store = get_store()
project_root = Path(store.project_root)


def _reference_texts(meta):
    """Rekonstruiert den Referenz-Korpus (Token-Listen) fuer die Kohaerenz."""
    modus = str(meta.get("modus", ""))
    if "korpus" in modus:
        p = meta.get("korpus_path")
        if not p or not Path(p).exists():
            return None
        cdf = pd.read_csv(p)
        cc = meta.get("content_col", "content")
        seg = cseg.segment_corpus(cdf, chunk_words=int(meta.get("chunk_words", 1000)),
                                  id_col=meta.get("id_col", "_id"), content_col=cc,
                                  min_words=int(meta.get("min_words", 50)))
        return [str(t).split() for t in seg[cc]]
    p = meta.get("matrix_path")
    if not p or not Path(p).exists():
        return None
    return tmet.docs_from_matrix(pd.read_csv(p),
                                 feature_start=int(meta.get("feature_start", 10)))


def _pick(label, patterns, key):
    """Dateiauswahl per Glob im Projektordner, mit manueller Pfad-Alternative."""
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


tab_build, tab_metrics = st.tabs(["Topic-Modell erstellen", "Metriken (Qualität)"])

with tab_build:
    method_label = st.radio(
        "Verfahren", ["NMF (auf TF-IDF)", "LDA (auf Frequenzen)", "MALLET (Java, LDA)"],
        horizontal=True,
        help="Drei Wege zum selben Ziel: aus den Texten thematische Wortgruppen "
             "('Topics') gewinnen. NMF rechnet auf TF-IDF und liefert scharfe, gut "
             "interpretierbare Topics; es ist deterministisch (gleicher Startwert "
             "-> gleiches Ergebnis) und ein guter Einstieg. LDA ist das klassische "
             "probabilistische Modell (scikit-learn): jedes Dokument ist eine "
             "Mischung von Topics. MALLET verfolgt dieselbe Idee wie LDA, rechnet "
             "aber per Gibbs-Sampling in Java - oft kohärentere Topics, benötigt "
             "jedoch Java und ein eingerichtetes MALLET.")
    method = ({"NMF": "nmf", "LDA": "lda", "MAL": "mallet"}[method_label[:3]])

    if method == "mallet":
        # MALLET arbeitet immer auf Text (Chunking), nicht auf einer fertigen Matrix
        use_corpus = True
        st.caption("MALLET segmentiert den Korpus-Text und rechnet ueber Java. "
                   "Eine fertige DTM/TF-IDF wird hier nicht verwendet.")
        mallet_path = mr.read_mallet_path(project_root)
        if not mallet_path:
            st.error("MALLET ist nicht eingerichtet. Bitte zuerst die Seite "
                     "'MALLET einrichten' ausfuehren.")
        else:
            st.caption(f"MALLET-Starter: `{mallet_path}`")
    else:
        mallet_path = None
        source_label = st.radio(
            "Eingabe", ["Korpus-Text + Chunking (empfohlen)",
                        "Vorhandene DTM/TF-IDF-Matrix"], horizontal=True,
            help="Chunking zerlegt lange Texte (z. B. ganze Romane) in kleinere "
                 "Segmente. Das ist meist die bessere Wahl: ein umfangreicher Text mischt "
                 "zu viele Themen, viele kürzere Segmente ergeben klarere und "
                 "stabilere Topics. 'Vorhandene Matrix' überspringt diesen Schritt "
                 "und nutzt eine bereits berechnete DTM/TF-IDF.")
        use_corpus = source_label.startswith("Korpus")

    # --- Gemeinsame Modell-Parameter ---
    c1, c2, c3, c4 = st.columns(4)
    n_topics = c1.number_input("Anzahl Topics", 2, 500, 20, key="tm_k",
                               help="Wie viele Themen das Modell finden soll. Es gibt "
                                    "kein 'richtiges' K: zu wenige Topics vermischen "
                                    "Themen, zu viele zersplittern sie. Übliche Praxis: "
                                    "mehrere Werte (z. B. 10/20/40) testen und im Tab "
                                    "'Metriken' Kohärenz und Diversität vergleichen.")
    top_words = c2.number_input("Wörter pro Topic", 5, 500, 100, key="tm_tw",
                                help="Wie viele der wichtigsten Wörter je Topic "
                                     "gespeichert werden. Zum Benennen/Interpretieren "
                                     "reichen meist die Top 10-20; ein höherer Wert "
                                     "(z. B. 100) ist für spätere Auswertungen nützlich "
                                     "und schadet nicht.")
    random_state = c3.number_input("random_state", 0, 10_000, 42, key="tm_rs",
                                   help="Startwert des Zufallsgenerators. Gleicher Wert "
                                        "= exakt reproduzierbares Ergebnis (wichtig fürs "
                                        "wissenschaftliche Arbeiten). Nur ändern, wenn "
                                        "man bewusst eine andere Initialisierung testet.")
    max_iter = c4.number_input("max_iter (0 = Default)", 0, 2000, 0, key="tm_mi",
                               help="Wie oft das Optimierungsverfahren über die Daten "
                                    "läuft. 0 = sinnvolle Standardwerte (NMF 400, "
                                    "LDA 20). Mehr Iterationen = genauer, aber langsamer; "
                                    "meist nur nötig, wenn das Modell noch nicht "
                                    "konvergiert (sich nicht mehr ändert).")

    # --- Methodenspezifische Hyperparameter ---
    extra_params: dict = {}
    mallet_params: dict = {}
    with st.expander("Erweiterte Hyperparameter"):
        if method == "nmf":
            h1, h2 = st.columns(2)
            nmf_init = h1.selectbox("init", ["nndsvda", "nndsvd", "nndsvdar", "random"],
                                    key="tm_init",
                                    help="Wie die Faktoren vor dem Optimieren "
                                         "vorbelegt werden. 'nndsvda' ist ein "
                                         "bewährter, deterministischer Standard "
                                         "(immer gleiches Ergebnis). 'random' startet "
                                         "zufällig - nötig, wenn man mehr Topics als "
                                         "Dokumente extrahieren will. Im Zweifel: nndsvda.")
            nmf_beta = h2.selectbox("beta_loss", ["frobenius", "kullback-leibler"],
                                    key="tm_beta",
                                    help="Fehlermaß, das NMF minimiert. 'frobenius' "
                                         "(Standard) passt gut zu TF-IDF. "
                                         "'kullback-leibler' eignet sich eher für reine "
                                         "Zählungen und erzwingt automatisch den "
                                         "langsameren Solver 'mu'.")
            nmf_l1 = h1.slider("l1_ratio", 0.0, 1.0, 0.0, 0.05, key="tm_l1",
                               help="Art der Regularisierung (Mischung aus L1 und L2). "
                                    "0 = nur L2 (glatt), 1 = nur L1 (erzeugt mehr "
                                    "Nullen -> sparsamere, schärfere Topics). Wirkt nur "
                                    "zusammen mit alpha_W/alpha_H > 0.")
            nmf_aw = h2.number_input("alpha_W", 0.0, 10.0, 0.0, 0.1, key="tm_aw",
                                     help="Stärke der Regularisierung der "
                                          "Dokument-Topic-Faktoren (W). 0 = aus. Größere "
                                          "Werte dämpfen Überanpassung, können Topics "
                                          "aber verwässern - behutsam erhöhen.")
            nmf_ah = h1.number_input("alpha_H", 0.0, 10.0, 0.0, 0.1, key="tm_ah",
                                     help="Wie alpha_W, aber für die Topic-Wort-Faktoren "
                                          "(H). 0 = aus. Höhere Werte glätten die "
                                          "Wortverteilungen der Topics.")
            nmf_tol = h2.number_input("tol", 1e-6, 1e-1, 1e-4, format="%.6f",
                                      key="tm_tol",
                                      help="Abbruchschwelle: Ändert sich das Ergebnis "
                                           "zwischen zwei Iterationen weniger als dieser "
                                           "Wert, stoppt die Optimierung. Kleiner = "
                                           "genauer, aber langsamer.")
            extra_params = {"init": nmf_init, "beta_loss": nmf_beta,
                            "l1_ratio": float(nmf_l1), "alpha_W": float(nmf_aw),
                            "alpha_H": float(nmf_ah), "tol": float(nmf_tol)}
            if nmf_beta != "frobenius":
                extra_params["solver"] = "mu"
        elif method == "lda":
            h1, h2 = st.columns(2)
            lda_lm = h1.selectbox("learning_method", ["batch", "online"], key="tm_lm",
                                  help="'batch' nutzt in jeder Runde alle Dokumente - "
                                       "stabil und für kleine/mittlere Korpora "
                                       "empfohlen. 'online' lernt in Häppchen - "
                                       "schneller bei sehr großen Korpora, dafür etwas "
                                       "schwankender.")
            lda_decay = h2.slider("learning_decay", 0.5, 1.0, 0.7, 0.05, key="tm_decay",
                                  help="Nur bei 'online': wie schnell der Lernschritt "
                                       "mit der Zeit kleiner wird. Werte um 0.7 sind "
                                       "üblich. Bei 'batch' ohne Wirkung.")
            lda_alpha = h1.number_input("doc_topic_prior (alpha, 0 = auto)", 0.0, 10.0,
                                        0.0, 0.01, key="tm_alpha",
                                        help="alpha steuert, wie viele Topics ein "
                                             "Dokument typischerweise enthält. Kleines "
                                             "alpha = wenige, klare Topics je Dokument; "
                                             "großes alpha = Dokumente mischen viele "
                                             "Topics. 0 = automatisch (1/Anzahl Topics).")
            lda_beta = h2.number_input("topic_word_prior (beta, 0 = auto)", 0.0, 10.0,
                                       0.0, 0.01, key="tm_betaprior",
                                       help="beta steuert, wie breit ein Topic über die "
                                            "Wörter streut. Kleines beta = Topics aus "
                                            "wenigen, spezifischen Wörtern; großes beta "
                                            "= breitere Topics. 0 = automatisch "
                                            "(1/Anzahl Topics).")
            extra_params = {"learning_method": lda_lm, "learning_decay": float(lda_decay)}
            if lda_alpha > 0:
                extra_params["doc_topic_prior"] = float(lda_alpha)
            if lda_beta > 0:
                extra_params["topic_word_prior"] = float(lda_beta)
        else:  # MALLET
            h1, h2 = st.columns(2)
            m_iter = h1.number_input("num_iterations", 50, 5000, 1000, 50,
                                     key="tm_m_iter",
                                     help="Anzahl der Gibbs-Sampling-Durchläufe. Mehr "
                                          "Iterationen = stabileres, besser "
                                          "eingependeltes Modell, aber längere "
                                          "Rechenzeit. 1000 ist ein solider Standard; "
                                          "bei großen Korpora ruhig mehr.")
            m_opt = h2.number_input("optimize_interval", 0, 100, 10, key="tm_m_opt",
                                    help="Alle wie viele Iterationen MALLET die Prioren "
                                         "(alpha/beta) automatisch an die Daten anpasst. "
                                         "0 = aus (MALLET-Default). Werte um 10-20 lassen "
                                         "zu, dass manche Topics häufiger sind als andere "
                                         "(meist realistischer); sehr aggressive "
                                         "Optimierung kann das Modell aber auf seltene "
                                         "Topics verschieben (Schöch 2016).")
            m_alpha = h1.number_input("alpha (Summe über Topics)", 0.1, 100.0, 5.0, 0.1,
                                      key="tm_m_alpha",
                                      help="Dirichlet-alpha als Summe über alle Topics: "
                                           "wie stark Dokumente mehrere Topics mischen. "
                                           "MALLET-Default 5.0 (pro Topic also "
                                           "5.0/Anzahl Topics). Größer = gleichmäßigere "
                                           "Mischung je Dokument. Bei aktiver "
                                           "Optimierung nur Startwert.")
            m_beta = h2.number_input("beta", 0.001, 1.0, 0.01, 0.001, format="%.3f",
                                     key="tm_m_beta",
                                     help="Dirichlet-beta pro Wort: Glättung der "
                                          "Topic-Wort-Verteilung. MALLET-Default 0.01 - "
                                          "für natürliche Sprache fast immer ein guter "
                                          "Wert. Kleiner = schärfere, spezifischere "
                                          "Topics.")
            mallet_params = {"num_iterations": int(m_iter), "optimize_interval": int(m_opt),
                             "alpha": float(m_alpha), "beta": float(m_beta)}

    # --- Eingabe-spezifische Optionen ---
    chunk_words = min_words = max_features = min_df = None
    max_df = None
    lowercase = False
    aggregate = True
    corpus_path = matrix_path = None
    content_col = "content"
    id_col = "_id"
    feature_start = 10

    if use_corpus:
        cands = sorted(project_root.glob("korpus/*.csv"))
        if cands:
            names = [str(p.relative_to(project_root)) for p in cands] + ["(anderer Pfad...)"]
            sel = st.selectbox("Korpus-CSV (mit Textspalte)", names, key="tm_corpus_sel")
            corpus_path = (Path(st.text_input("Pfad zur Korpus-CSV",
                                              value="output/processed_corpus/korpus_stop.csv", key="tm_corpus_path"))
                           if sel == "(anderer Pfad...)" else project_root / sel)
        else:
            corpus_path = Path(st.text_input("Pfad zur Korpus-CSV",
                                             value="output/processed_corpus/korpus_stop.csv", key="tm_corpus_path"))
        if not corpus_path.is_absolute():
            corpus_path = project_root / corpus_path

        a1, a2, a3 = st.columns(3)
        content_col = a1.text_input("Textspalte", value="content_stop", key="tm_cc")
        id_col = a2.text_input("ID-Spalte", value="id", key="tm_idc")
        chunk_words = a3.number_input("Wörter pro Segment", 100, 5000, 1000, 50,
                                      key="tm_cw",
                                      help="Länge der Textsegmente in Wörtern. "
                                           "Faustregel: 500-1000. Kürzere Segmente "
                                           "ergeben mehr und feinere Dokumente und oft "
                                           "klarere Topics; sehr kurze Segmente werden "
                                           "aber unzuverlässig (zu wenig Kontext).")
        min_words = st.number_input("Mindestwörter je Segment (kürzere Endstücke verwerfen)",
                                    0, 1000, 50, key="tm_mw",
                                    help="Restsegmente am Textende, die kürzer als "
                                         "dieser Wert sind, werden verworfen - sie "
                                         "enthalten zu wenig Information für ein "
                                         "verlässliches Topic-Profil.")
        if method == "mallet":
            lowercase = st.checkbox(
                "lowercase", value=False, key="tm_mallet_lc",
                help="Aus (Standard): MALLET erhält die Groß-/Kleinschreibung "
                     "(--preserve-case). An: alle Tokens werden beim Import klein "
                     "geschrieben – kann im Deutschen großgeschriebene Substantive/"
                     "Eigennamen mit gleichlautenden Verben verschmelzen.")
        if method != "mallet":
            with st.expander("Vektorisierung (Vokabular)"):
                v1, v2 = st.columns(2)
                max_features = v1.number_input("max_features", 100, 100000, 2000, 100,
                                               key="tm_mf",
                                               help="Obergrenze für das Vokabular: nur "
                                                    "die häufigsten N Wörter werden als "
                                                    "Merkmale verwendet. Begrenzt Rauschen "
                                                    "und Rechenzeit. Mehr = reichere "
                                                    "Topics, aber auch höhere NMF-Rang-"
                                                    "Grenze. 1000-3000 sind typisch.")
                min_df = v2.number_input("min_df (Mindest-Dokumente je Term)", 1, 100, 2,
                                         key="tm_mindf",
                                         help="Wörter ignorieren, die in weniger als so "
                                              "vielen Segmenten vorkommen. Filtert "
                                              "Tippfehler/Einzelfälle heraus. 2 ist ein "
                                              "guter Startwert.")
                max_df = v1.slider("max_df (max. Dokumentanteil)", 0.5, 1.0, 0.95, 0.01,
                                   key="tm_maxdf",
                                   help="Wörter ignorieren, die in mehr als diesem "
                                        "Anteil aller Segmente vorkommen. Entfernt zu "
                                        "allgegenwärtige Wörter, die keine Themen "
                                        "trennen (eine Art automatische Stoppwortliste).")
                lowercase = v2.checkbox("lowercase", value=False, key="tm_lc",
                                        help="Aus (empfohlen): Groß-/Kleinschreibung der "
                                             "bereits lemmatisierten Terme bleibt "
                                             "erhalten. An: alles klein - kann im "
                                             "Deutschen Substantive und Verben "
                                             "verschmelzen.")
        aggregate = st.checkbox("Zusätzlich pro Ursprungstext mitteln "
                                "(Segment-Topics -> Text-Profil)", value=True,
                                key="tm_agg",
                                help="An: zusätzlich zur Topic-Verteilung je Segment "
                                     "wird je Originaltext eine gemittelte "
                                     "Verteilung berechnet - ein Topic-Profil pro Werk. "
                                     "Die Segment-Ebene bleibt dabei erhalten.")
    else:
        hits = sorted(project_root.glob("output/dtm_tfidf*/*.csv"))
        if hits:
            names = [str(p.relative_to(project_root)) for p in hits]
            pref = "tfidf" if method == "nmf" else "dtm"
            idx = next((i for i, n in enumerate(names) if pref in n.lower()), 0)
            matrix_path = project_root / st.selectbox("Eingabematrix (s03_dtm_tfidf)",
                                                      names, index=idx, key="tm_mx")
        else:
            matrix_path = Path(st.text_input("Pfad zur DTM-/TF-IDF-CSV", value="",
                                             key="tm_mxpath"))
        m1, m2 = st.columns(2)
        id_col = m1.text_input("ID-Spalte", value="_id", key="tm_idm")
        feature_start = m2.number_input("Erste Term-Spalte (Index)", 0, 100, 10,
                                        key="tm_fs",
                                        help="Metadaten zuerst, Term-Features ab diesem Index.")

    # --- Berechnen ---
    if st.button("Topic-Modell berechnen", type="primary", key="tm_run"):
        try:
            model_params: dict = {}
            if method == "mallet":
                if not mallet_path:
                    st.warning("MALLET ist nicht eingerichtet (Seite 'MALLET einrichten').")
                    st.stop()
                if not corpus_path or not corpus_path.exists():
                    st.warning("Bitte eine gueltige Korpus-CSV waehlen.")
                    st.stop()
                cdf = pd.read_csv(corpus_path)
                dt, tw, info, agg = mr.fit_mallet_from_corpus(
                    cdf, mallet_path, n_topics=int(n_topics),
                    content_col=content_col.strip(), id_col=id_col.strip(),
                    chunk_words=int(chunk_words), min_words=int(min_words),
                    top_words=int(top_words), random_seed=int(random_state),
                    preserve_case=not lowercase,
                    aggregate=aggregate, **mallet_params)
                st.session_state["tm_agg_result"] = agg
                model_params = {k: info[k] for k in
                                ("engine", "num_iterations", "optimize_interval",
                                 "alpha", "beta", "random_seed", "preserve_case")
                                if k in info}
                st.session_state["tm_meta"] = {
                    "modus": "korpus+chunking+mallet", "methode": "mallet",
                    "korpus": corpus_path.name, "korpus_path": str(corpus_path),
                    "content_col": content_col.strip(), "id_col": id_col.strip(),
                    "chunk_words": int(chunk_words),
                    "min_words": int(min_words), "n_topics": int(n_topics),
                    "top_words": int(top_words),
                    "n_segments": info.get("n_segments"),
                    "n_sources": info.get("n_sources")}
                st.session_state["tm_basename"] = f"mallet_{int(n_topics)}"
                msg = (f"Fertig (MALLET): {dt.shape[1]} Topics ueber "
                       f"{info.get('n_segments')} Segmente aus "
                       f"{info.get('n_sources')} Texten.")
            elif use_corpus:
                if not corpus_path or not corpus_path.exists():
                    st.warning("Bitte eine gueltige Korpus-CSV waehlen.")
                    st.stop()
                cdf = pd.read_csv(corpus_path)
                dt, tw, model, agg, info = tm.fit_topics_from_corpus(
                    cdf, n_topics=int(n_topics), method=method,
                    content_col=content_col.strip(), id_col=id_col.strip(),
                    chunk_words=int(chunk_words), min_words=int(min_words),
                    top_words=int(top_words), max_features=int(max_features),
                    min_df=int(min_df), max_df=float(max_df), lowercase=lowercase,
                    random_state=int(random_state), max_iter=int(max_iter) or None,
                    extra_params=extra_params, aggregate=aggregate)
                st.session_state["tm_agg_result"] = agg
                model_params = model.get_params()
                st.session_state["tm_meta"] = {
                    "modus": "korpus+chunking", "methode": method,
                    "korpus": corpus_path.name, "korpus_path": str(corpus_path),
                    "content_col": content_col.strip(), "id_col": id_col.strip(),
                    "chunk_words": int(chunk_words),
                    "min_words": int(min_words), "max_features": int(max_features),
                    "min_df": int(min_df), "max_df": float(max_df),
                    "lowercase": lowercase, "n_topics": int(n_topics),
                    "top_words": int(top_words), **info}
                st.session_state["tm_basename"] = f"sklearn_{method}_{int(n_topics)}"
                msg = (f"Fertig: {dt.shape[1]} Topics ueber {info['n_segments']} "
                       f"Segmente aus {info['n_sources']} Texten "
                       f"({info['n_features']} Features).")
            else:
                if not matrix_path or not matrix_path.exists():
                    st.warning("Bitte eine gueltige DTM-/TF-IDF-CSV waehlen.")
                    st.stop()
                mdf = pd.read_csv(matrix_path)
                dt, tw, model = tm.fit_topics(
                    mdf, n_topics=int(n_topics), method=method,
                    top_words=int(top_words), id_col=id_col.strip() or None,
                    feature_start=int(feature_start), random_state=int(random_state),
                    max_iter=int(max_iter) or None, extra_params=extra_params)
                st.session_state["tm_agg_result"] = None
                model_params = model.get_params()
                st.session_state["tm_meta"] = {
                    "modus": "matrix", "methode": method,
                    "eingabematrix": matrix_path.name, "matrix_path": str(matrix_path),
                    "feature_start": int(feature_start),
                    "id_col": id_col.strip() or None, "n_topics": int(n_topics),
                    "top_words": int(top_words)}
                st.session_state["tm_basename"] = f"sklearn_{method}_{int(n_topics)}"
                msg = f"Fertig: {dt.shape[1]} Topics ueber {dt.shape[0]} Dokumente."

            st.session_state["tm_doc_topic"] = dt
            st.session_state["tm_topic_word"] = tw
            st.session_state["tm_model_params"] = model_params
            st.success(msg)
            if method == "nmf" and dt.shape[1] < int(n_topics):
                st.warning(
                    f"Es wurden {dt.shape[1]} statt {int(n_topics)} Topics gebildet. "
                    f"NMF mit nndsvd-Init kann hoechstens min(#Dokumente, #Features) "
                    f"Topics erzeugen. Loesung: mehr/kleinere Segmente, hoeheres "
                    f"max_features, init='random' oder LDA.")
        except Exception as e:
            show_error(e)

    # --- Ergebnis + Speichern ---
    if "tm_topic_word" in st.session_state:
        topic_word = st.session_state["tm_topic_word"]
        doc_topic = st.session_state["tm_doc_topic"]
        agg = st.session_state.get("tm_agg_result")

        st.subheader("Topics (Top-Woerter)")
        st.dataframe(topic_word.iloc[:, :min(15, topic_word.shape[1])],
                     use_container_width=True)

        with st.expander("Dokument-/Segment-Topic-Verteilung (Vorschau)"):
            st.dataframe(doc_topic.head(20), use_container_width=True)
        if agg is not None:
            with st.expander("Pro Ursprungstext gemittelt (Vorschau)"):
                st.dataframe(agg.head(20), use_container_width=True)

        st.subheader("Speichern fuer die Weiterverarbeitung")
        basename = st.text_input("Modellname (Dateipraefix)",
                                 value=st.session_state.get("tm_basename", "sklearn"),
                                 key="tm_name")
        target = st.text_input("Zielordner (relativ zum Projekt)",
                               value=f"resources/topic-models/{basename}", key="tm_target")

        def _params():
            p = dict(st.session_state.get("tm_meta", {}))
            prefix = "mallet" if p.get("methode") == "mallet" else "sklearn"
            for k, v in st.session_state.get("tm_model_params", {}).items():
                p[f"{prefix}.{k}"] = v
            return p

        if st.button("💾 Topic-Dateien speichern", key="tm_save"):
            try:
                dt_path, tw_path, hp_path = tm.save_topic_outputs(
                    doc_topic, topic_word, project_root / target, basename=basename,
                    params=_params())
                lines = [f"- `{dt_path.relative_to(project_root)}`",
                         f"- `{tw_path.relative_to(project_root)}`"]
                if hp_path:
                    lines.append(f"- `{hp_path.relative_to(project_root)}` (Hyperparameter)")
                if agg is not None:
                    agg_path = (project_root / target /
                                f"document-topics-distribution_{basename}_pro_text.csv")
                    agg.to_csv(agg_path, encoding="utf-8")
                    lines.append(f"- `{agg_path.relative_to(project_root)}` (pro Text gemittelt)")
                st.success("Gespeichert:\n" + "\n".join(lines))
                st.caption("Auf der Seite 'Topics & Termset-Topics verarbeiten' als "
                           "Document-Topic- bzw. Topic-Words-Eingabe waehlen. Die "
                           "'_pro_text'-Datei nutzt Roman-Profile statt Segmente.")
            except Exception as e:
                show_error(e)

        bn = st.session_state.get("tm_basename", "sklearn")
        df_with_download(doc_topic.reset_index(),
                         f"document-topics-distribution_{bn}", key="tm_dl_dt")
        df_with_download(topic_word.reset_index(), f"{bn}_topic_words", key="tm_dl_tw")
        if agg is not None:
            df_with_download(agg.reset_index(),
                             f"document-topics-distribution_{bn}_pro_text", key="tm_dl_agg")
        st.download_button("⬇️ Hyperparameter (.txt)",
                           data=tm.format_hyperparams(_params()),
                           file_name=f"{bn}_hyperparams.txt", mime="text/plain",
                           key="tm_dl_hp")


with tab_metrics:
    st.caption("Qualitätsmaße für ein Topic-Modell - das zuletzt berechnete "
               "oder ein manuell geladenes.")
    src = st.radio("Modellquelle",
                   ["Zuletzt berechnetes Modell", "Modell manuell laden"],
                   horizontal=True, key="met_src")

    topic_word = None
    meta = {}
    if src.startswith("Zuletzt"):
        if "tm_topic_word" not in st.session_state:
            st.info("In dieser Sitzung wurde noch kein Modell berechnet. Wähle "
                    "'Modell manuell laden' oder berechne zuerst eines im Tab "
                    "'Topic-Modell erstellen'.")
        else:
            topic_word = st.session_state["tm_topic_word"]
            meta = st.session_state.get("tm_meta", {})
    else:
        st.markdown("##### Topic-Wörter-Datei")
        tw_path = _pick(
            "Topic-Words-Matrix (CSV: Zeilen = Topics, Zellen = Wörter)",
            ["resources/topic-models/**/*topic_words*.csv",
             "resources/topic-models/**/*words*.csv",
             "output/**/*topic_words*.csv"], key="met_tw")
        if tw_path and Path(tw_path).exists():
            try:
                tw = pd.read_csv(tw_path)
                first = tw.columns[0]
                if str(first).lower() in ("index", "topic", "unnamed: 0") \
                        or pd.api.types.is_integer_dtype(tw[first]):
                    tw = tw.drop(columns=[first])
                topic_word = tw
                st.caption(f"{topic_word.shape[0]} Topics geladen.")
            except Exception as e:
                show_error(e)

        st.markdown("##### Referenz-Korpus für die Kohärenz")
        st.caption("Nur für die Kohärenz nötig - die Diversität funktioniert "
                   "auch ohne. Quelle sollte zum geladenen Modell passen.")
        ref_mode = st.radio("Referenzquelle",
                            ["Korpus-CSV (wird segmentiert)", "DTM-/TF-IDF-Matrix"],
                            horizontal=True, key="met_refmode")
        if ref_mode.startswith("Korpus"):
            ref_corpus = _pick("Korpus-CSV (mit Textspalte)",
                               ["korpus/*.csv", "resources/**/*.csv"],
                               key="met_refcorp")
            rc1, rc2 = st.columns(2)
            ref_cc = rc1.text_input("Textspalte", value="content", key="met_cc")
            ref_idc = rc2.text_input("ID-Spalte", value="_id", key="met_idc")
            rc3, rc4 = st.columns(2)
            ref_cw = rc3.number_input("Wörter pro Segment", 100, 5000, 1000, 50,
                                      key="met_cw")
            ref_mw = rc4.number_input("Mindestwörter je Segment", 0, 1000, 50,
                                      key="met_mw")
            if ref_corpus:
                meta = {"modus": "korpus", "korpus_path": str(ref_corpus),
                        "content_col": ref_cc.strip(), "id_col": ref_idc.strip(),
                        "chunk_words": int(ref_cw), "min_words": int(ref_mw)}
        else:
            ref_matrix = _pick("DTM-/TF-IDF-Matrix (CSV)",
                               ["output/dtm_tfidf*/*.csv"], key="met_refmat")
            ref_fs = st.number_input("Erste Term-Spalte (Index)", 0, 100, 10,
                                     key="met_fs")
            if ref_matrix:
                meta = {"modus": "matrix", "matrix_path": str(ref_matrix),
                        "feature_start": int(ref_fs)}

    if topic_word is not None:
        topic_lists = tmet.topic_word_lists_from_df(topic_word)
        met_topk = st.slider("Top-Wörter je Topic für die Maße", 5, 50, 10,
                             key="met_topk")
        div = tmet.topic_diversity(topic_lists, top_k=int(met_topk))
        st.metric("Topic-Diversität", f"{div:.3f}",
                  help="Anteil eindeutiger Wörter über alle Topic-Top-Listen "
                       "(Dieng et al. 2020). Höher = weniger redundante Topics.")
        st.markdown("##### Kohärenz")
        st.caption("Die Kohärenz nutzt den Referenz-Korpus (wird dafür "
                   "gelesen/segmentiert).")
        measure = st.selectbox("Maß", ["C_v (gensim)", "c_npmi (gensim)",
                                        "NPMI (eigenständig, ohne gensim)"],
                               key="met_measure",
                               help="C_v gilt als Forschungsstandard (Röder et al. "
                                    "2015); NPMI als robuste Gegenprobe ohne gensim.")
        if st.button("Kohärenz berechnen", key="met_btn", type="primary"):
            try:
                ref = _reference_texts(meta)
                if not ref:
                    st.warning("Referenz-Korpus konnte nicht rekonstruiert werden "
                               "(Quelle/Pfad fehlt?).")
                else:
                    if measure.startswith("NPMI"):
                        res = tmet.coherence_npmi(topic_lists, ref, top_k=int(met_topk))
                    else:
                        meas = "c_v" if measure.startswith("C_v") else "c_npmi"
                        res = tmet.coherence_gensim(topic_lists, ref, measure=meas,
                                                    top_k=int(met_topk))
                    st.metric(f"Mittlere Kohärenz ({res.get('measure')})",
                              f"{res['mean']:.4f}")
                    per = res["per_topic"]
                    dfm = pd.DataFrame({"Topic": list(range(len(per))),
                                        "Kohärenz": per}).sort_values(
                                            "Kohärenz", ascending=False)
                    df_with_download(dfm, "topic_kohaerenz", key="met_dl")
                    st.caption("Hinweis (Hoyle et al. 2021): automatische Kohärenz "
                               "ist ein Proxy - mit Word-Intrusion/Lektüre absichern.")
            except Exception as e:
                show_error(e)
