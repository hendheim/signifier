#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Wortvektoren
===================

Analysen auf Basis des trainierten Word2Vec-Modells (Pipeline-Schritt s07),
in zwei Top-Tabs gegliedert:

- **Wortvektoren** (Pipeline): Embeddings (aehnlichste Woerter),
  Embeddings-Vergleich (gemeinsame Nachbarn), Netzwerk (semantisches Netzwerk).
- **Metriken** (Qualitaet): Abdeckung/OOV, Wortpaar-Korrelation, Stabilitaet,
  diachrone Drift (explorer_core.wvm_metrics).

Das Modell wird ueber den ModelStore nur einmal geladen (lazy, gecacht).
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from ui_helpers import (get_store, get_models, show_error, df_with_download,
                        fig_with_download, parse_terms)
from explorer_core.analysis_vectors import (most_similar, compare_embeddings,
                                            semantic_network)
from explorer_core.viz_export import save_figure
from explorer_core import wvm_metrics as wm

st.set_page_config(page_title="Wortvektoren", layout="wide")
st.title("🧭 Wortvektoren")
st.caption("Semantische Ähnlichkeiten aus dem Word2Vec-Modell")

store = get_store()
models = get_models()


def _load_kv2(path_str: str):
    """Laedt eine zweite KeyedVectors-Datei (gensim) fuer Vergleich/Drift."""
    from gensim.models import KeyedVectors
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(store.project_root) / p
    if not p.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {p}")
    return KeyedVectors.load(str(p))


def _parse_pairs(text: str):
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (",", ";", "\t"):
            if sep in line:
                parts = [x.strip() for x in line.split(sep)]
                break
        else:
            parts = line.split()
        if len(parts) >= 3:
            try:
                pairs.append((parts[0], parts[1], float(parts[2])))
            except ValueError:
                continue
    return pairs


def _word_list(raw: str):
    return [w for chunk in raw.splitlines()
            for w in chunk.replace(",", " ").split()]


tab_pipeline, tab_metrics = st.tabs(["Wortvektoren", "Metriken (Qualität)"])

# ===========================================================================
# TOP-TAB: Wortvektoren (bisherige Pipeline)
# ===========================================================================
with tab_pipeline:
    tab_emb, tab_cmp, tab_net = st.tabs(
        ["Embeddings", "Embeddings-Vergleich", "Netzwerk"])

    # ----- Embeddings – ähnlichste Wörter -----
    with tab_emb:
        st.subheader("Ähnlichste Wörter")
        c1, c2 = st.columns([3, 1])
        emb_word = c1.text_input("Begriff", key="emb_word",
                                 placeholder="z. B. natur")
        emb_topn = c2.number_input("Top N", 5, 200, 20, 5, key="emb_topn",
                                   help="Anzahl der ähnlichsten Wörter (nächste "
                                        "Nachbarn im Vektorraum), die angezeigt werden.")
        if st.button("Ähnliche Wörter suchen", key="emb_btn", type="primary"):
            if not emb_word.strip():
                st.warning("Bitte einen Begriff eingeben.")
            else:
                try:
                    kv = models.load()
                    df = most_similar(kv, emb_word.strip(), top_n=int(emb_topn))
                    df_with_download(df, f"embeddings_{emb_word.strip()}", key="emb")
                except KeyError:
                    st.warning(f"»{emb_word.strip()}« ist nicht im Vokabular "
                               "des Modells.")
                except Exception as e:
                    show_error(e)

    # ----- Embeddings-Vergleich – gemeinsame Nachbarn -----
    with tab_cmp:
        st.subheader("Embeddings-Vergleich")
        st.caption("Findet Wörter, die sowohl dem Zentralbegriff als auch den "
                   "Vergleichsbegriffen ähnlich sind (Schwelle = minimale "
                   "Kosinus-Ähnlichkeit).")
        cmp_central = st.text_input("Zentralbegriff", key="cmp_central")
        cmp_others_raw = st.text_input("Vergleichsbegriffe (Komma-getrennt)",
                                       key="cmp_others")
        c1, c2 = st.columns(2)
        cmp_topn = c1.number_input("Nachbarn pro Begriff", 10, 500, 50, 10,
                                   key="cmp_topn",
                                   help="Anzahl der nächsten Nachbarn je Begriff, "
                                        "die in den Vergleich einfließen.")
        cmp_thresh = c2.slider("Ähnlichkeits-Schwelle", 0.0, 1.0, 0.3, 0.05,
                               key="cmp_thresh",
                               help="Mindest-Kosinusähnlichkeit, ab der ein Nachbar "
                                    "in den Vergleich aufgenommen wird.")
        if st.button("Vergleichen", key="cmp_btn", type="primary"):
            others = parse_terms(cmp_others_raw)
            if not cmp_central.strip() or not others:
                st.warning("Bitte Zentralbegriff und mindestens einen "
                           "Vergleichsbegriff eingeben.")
            else:
                try:
                    kv = models.load()
                    overview, details = compare_embeddings(
                        kv, cmp_central.strip(), [w for w in others],
                        top_n=int(cmp_topn), threshold=float(cmp_thresh))
                    st.markdown("##### Übersicht")
                    df_with_download(overview, "embeddings_vergleich",
                                     key="cmp_overview")
                    for word, df_det in details.items():
                        with st.expander(f"Gemeinsame Nachbarn mit »{word}« "
                                         f"({len(df_det)})"):
                            df_with_download(df_det, f"vergleich_{word}",
                                             key=f"cmp_{word}")
                except KeyError as e:
                    st.warning(f"Begriff nicht im Vokabular: {e}")
                except Exception as e:
                    show_error(e)

    # ----- Netzwerk – semantisches Netzwerk -----
    with tab_net:
        st.subheader("Semantisches Netzwerk")
        net_words_raw = st.text_input("Begriffe (Komma-getrennt)", key="net_words")
        c1, c2, c3 = st.columns(3)
        net_topn = c1.number_input("Nachbarn pro Begriff", 2, 30, 8, key="net_topn",
                                   help="Anzahl der nächsten Nachbarn je Begriff, "
                                        "die als Knoten ins Netzwerk kommen.")
        net_thresh = c2.slider("Kanten-Schwelle (Ähnlichkeit)", 0.0, 1.0, 0.3,
                               0.05, key="net_thresh",
                               help="Mindest-Kosinusähnlichkeit, ab der eine Kante "
                                    "gezeichnet wird; höher = sparsameres Netz.")
        net_res = c3.selectbox("Bildgröße", ["Klein", "Mittel", "Groß"],
                               key="net_res")
        if st.button("Netzwerk zeichnen", key="net_btn", type="primary"):
            words = [w for w in parse_terms(net_words_raw)]
            if not words:
                st.warning("Bitte mindestens einen Begriff eingeben.")
            else:
                try:
                    kv = models.load()
                    fig = semantic_network(kv, words, top_n=int(net_topn),
                                           threshold=float(net_thresh),
                                           resolution=net_res)
                    save_figure(fig, "netzwerk", params={
                        "Begriffe": ", ".join(words),
                        "Nachbarn pro Begriff": int(net_topn),
                        "Kanten-Schwelle": float(net_thresh),
                        "Bildgröße": net_res}, key="net")
                except Exception as e:
                    show_error(e)

# ===========================================================================
# TOP-TAB: Metriken (Qualität)
# ===========================================================================
with tab_metrics:
    st.caption("Maße zur Prüfung des Modells. Hinweis: Standard-Ähnlichkeits-"
               "Benchmarks bilden modernes Deutsch ab und passen nur begrenzt zu "
               "historischen Texten - Nachbarschaft, Abdeckung und Stabilität "
               "sind aussagekräftiger als absolute Korrelationswerte.")
    try:
        kv = models.load()
    except Exception as e:
        show_error(e)
        st.stop()

    s = wm.model_summary(kv)
    mc1, mc2 = st.columns(2)
    mc1.metric("Vokabulargröße", f"{s['vocab_size']:,}".replace(",", "."))
    mc2.metric("Vektordimension", s["vector_size"])

    m_cov, m_pairs, m_stab, m_drift = st.tabs(
        ["Abdeckung / OOV", "Wortpaar-Korrelation", "Stabilität", "Diachrone Drift"])

    with m_cov:
        st.caption("Prüfe, welche deiner Interessenswörter im Modell vorkommen.")
        cov_raw = st.text_area("Wörter (eines pro Zeile oder komma-getrennt)",
                               value="", key="wm_cov_words")
        if st.button("Abdeckung prüfen", key="wm_cov_btn"):
            cov = wm.vocabulary_coverage(kv, _word_list(cov_raw))
            st.metric("Abdeckung", f"{cov['coverage']*100:.0f} %")
            st.write(f"Im Vokabular: {len(cov['in_vocab'])} / {cov['n']}")
            if cov["oov"]:
                st.warning("Nicht im Vokabular (OOV): " + ", ".join(cov["oov"]))

    with m_pairs:
        st.caption("Zeilen der Form 'wort1, wort2, score' (menschliches "
                   "Ähnlichkeitsurteil). Berechnet Spearman zwischen Modell-"
                   "Cosinus und Score.")
        up = st.file_uploader("Optional: CSV mit Spalten w1,w2,score",
                              type=["csv"], key="wm_pairs_csv")
        pairs_text = st.text_area("oder Paare direkt eingeben",
                                  value="koenig, koenigin, 0.9\nmann, frau, 0.85",
                                  key="wm_pairs_text")
        if st.button("Korrelation berechnen", key="wm_pairs_btn"):
            try:
                if up is not None:
                    dfp = pd.read_csv(up)
                    pairs = [(str(r.iloc[0]), str(r.iloc[1]), float(r.iloc[2]))
                             for _, r in dfp.iterrows()]
                else:
                    pairs = _parse_pairs(pairs_text)
                res = wm.evaluate_pairs(kv, pairs)
                if res["spearman"] is None:
                    st.warning(f"Zu wenige Paare im Vokabular "
                               f"(verwendet: {res['n_used']}, OOV: {res['n_oov']}).")
                else:
                    st.metric("Spearman-Korrelation", f"{res['spearman']:.3f}")
                    st.caption(f"verwendete Paare: {res['n_used']} | "
                               f"übersprungen (OOV): {res['n_oov']} | "
                               f"p = {res['pvalue']:.3g}")
                    st.caption("Vorbehalt: bei historischem Deutsch sind absolute "
                               "Werte mit Vorsicht zu lesen (Benchmark-Mismatch).")
            except Exception as e:
                show_error(e)

    with m_stab:
        st.caption("Vergleicht die Top-N-Nachbarn dieses Modells mit einem "
                   "zweiten Modell (z. B. anderer Seed/Lauf). Hohe Jaccard-"
                   "Überlappung = stabile Nachbarschaften.")
        path2 = st.text_input("Pfad zum zweiten Modell (KeyedVectors)",
                              value="", key="wm_stab_path")
        topn = st.number_input("Top-N Nachbarn", 5, 100, 10, key="wm_stab_n")
        words_s = st.text_area("Prüfwörter (leer = Stichprobe aus dem Vokabular)",
                               value="", key="wm_stab_words")
        if st.button("Stabilität berechnen", key="wm_stab_btn"):
            try:
                kv2 = _load_kv2(path2)
                words = _word_list(words_s) or list(kv.index_to_key[:200])
                res = wm.neighbor_stability(kv, kv2, words, topn=int(topn))
                if res["mean_jaccard"] is None:
                    st.warning("Keine gemeinsamen Prüfwörter in beiden Modellen.")
                else:
                    st.metric("Mittlere Jaccard-Überlappung",
                              f"{res['mean_jaccard']:.3f}")
                    st.caption(f"verglichene Wörter: {res['n_words']}")
            except Exception as e:
                show_error(e)

    with m_drift:
        st.caption("Bedeutungsverschiebung zwischen zwei Periodenmodellen "
                   "(Hamilton et al. 2016): orthogonale Procrustes-Ausrichtung, "
                   "dann Cosinus-Drift je Wort (0 = stabil, groß = verschoben).")
        base_path = st.text_input("Basis-Modell (frühere Periode)", value="",
                                  key="wm_drift_base")
        other_path = st.text_input("Vergleichs-Modell (spätere Periode)",
                                   value="", key="wm_drift_other")
        words_d = st.text_area("Wörter für die Drift-Messung", value="",
                               key="wm_drift_words")
        if st.button("Drift berechnen", key="wm_drift_btn"):
            try:
                kv_base = _load_kv2(base_path)
                kv_other = _load_kv2(other_path)
                words = _word_list(words_d)
                if not words:
                    st.warning("Bitte Wörter für die Drift-Messung angeben.")
                else:
                    drift = wm.semantic_drift(kv_base, kv_other, words)
                    if not drift:
                        st.warning("Keine der Wörter in beiden Modellen vorhanden.")
                    else:
                        dfd = (pd.DataFrame({"Wort": list(drift),
                                             "Drift": list(drift.values())})
                               .sort_values("Drift", ascending=False))
                        df_with_download(dfd, "wvm_drift", key="wm_drift_dl")
            except Exception as e:
                show_error(e)
