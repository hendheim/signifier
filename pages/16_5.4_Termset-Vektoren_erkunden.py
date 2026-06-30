#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 3: Termset
================

Analysen auf Basis der Termset-Datei (getaggte Begriffsliste, z. B. aus
``tt_s03_dtti.py``):

- Cluster       – UMAP-Projektion + agglomeratives Clustering der
                  Termset-Wörter im Embedding-Raum
- Wortwolke     – Termset-Wörter, Größe ∝ TF-IDF-Summe, Farbe = Tag
- Dendrogramme  – hierarchische Cluster, je Vor-Cluster ein Dendrogramm
"""

import streamlit as st

from ui_helpers import (get_store, get_models, show_error, df_with_download)
from explorer_core.analysis_vectors import (compute_termset_clusters,
                                            draw_termset_clusters,
                                            termset_wordcloud,
                                            termset_dendrograms)

from explorer_core.viz_export import save_figure

st.set_page_config(page_title="Termset-Vektoren erkunden", layout="wide")
st.title("🏷️ Termset")
st.caption("Cluster, Wortwolke und Dendrogramme der getaggten Begriffsliste")

store = get_store()
models = get_models()


@st.cache_data(show_spinner="Cluster werden berechnet …")
def _cached_cluster_compute(model_path, df_terms, k, n_neighbors, min_dist, _kv):
    """Cacht den teuren Cluster-/UMAP-Teil. ``_kv`` (führender Unterstrich) wird
    von Streamlit NICHT gehasht; der Cache-Schlüssel ist der Modellpfad plus
    Termset-Inhalt und die Hyperparameter. So zahlt man die einmalige
    UMAP-Numba-Kompilierung nur einmal, und identische Parameter liefern sofort."""
    return compute_termset_clusters(_kv, df_terms, k=k, n_neighbors=n_neighbors,
                                    min_dist=min_dist)

tab_cluster, tab_cloud, tab_dendro = st.tabs(
    ["Cluster", "Wortwolke", "Dendrogramme"]
)

# ---------------------------------------------------------------------------
# Tab: Cluster – UMAP + agglomeratives Clustering
# ---------------------------------------------------------------------------
with tab_cluster:
    st.subheader("Termset-Cluster im Embedding-Raum")
    st.caption("Die Termset-Wörter werden mit UMAP auf 2D projiziert und "
               "agglomerativ geclustert (Kosinus-Distanz, average linkage).")
    c1, c2, c3, c4, c5 = st.columns(5)
    cl_k = c1.number_input("Anzahl Cluster (k)", 2, 30, 5, key="cl_k", help='In wie viele Cluster die Termset-Wörter im Vektorraum eingeteilt werden (agglomerativ auf Kosinus-Distanzen).')
    cl_neighbors = c2.number_input("UMAP n_neighbors", 2, 100, 15,
                                   key="cl_nb", help='UMAP-Nachbarschaftsgröße: klein betont lokale, groß globale Struktur. Muss kleiner als die Wortzahl sein.')
    cl_mindist = c3.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.05,
                           key="cl_md", help='UMAP-Mindestabstand der Punkte: klein = enge Cluster, groß = gleichmäßigere Verteilung.')
    cl_res = c4.selectbox("Bildgröße", ["Klein", "Mittel", "Groß"],
                          key="cl_res")
    cl_labels = c5.checkbox("Wortlabels anzeigen", value=True,
                            key="cl_labels")
    if st.button("Cluster berechnen", key="cl_btn", type="primary"):
        try:
            kv = models.load()
            df_terms = store.load_termset()
            # Teurer Teil (UMAP + Clustering) gecacht; Zeichnen bleibt billig,
            # sodass Bildgröße/Labels ohne Neuberechnung anpassbar sind.
            terms, labels, coords, proj = _cached_cluster_compute(
                str(models.model_path), df_terms, int(cl_k),
                int(cl_neighbors), float(cl_mindist), kv)
            fig, df_clusters, clusters = draw_termset_clusters(
                terms, labels, coords, proj, resolution=cl_res,
                show_labels=cl_labels)
            save_figure(fig, "termset_cluster", params={
                "Anzahl Cluster (k)": int(cl_k), "UMAP n_neighbors": int(cl_neighbors),
                "UMAP min_dist": float(cl_mindist), "Bildgröße": cl_res,
                "Wortlabels": bool(cl_labels)}, key="cl")
            st.markdown("##### Cluster-Zuordnung")
            df_with_download(df_clusters, "termset_cluster_zuordnung",
                             key="cl_table")
            with st.expander("Cluster als Wortlisten"):
                for cid, words in sorted(clusters.items()):
                    st.markdown(f"**Cluster {cid}** ({len(words)} Wörter): "
                                + ", ".join(words))
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: Wortwolke – Termset × TF-IDF
# ---------------------------------------------------------------------------
with tab_cloud:
    st.subheader("Wortwolke des Termsets")
    st.caption("Wortgröße = TF-IDF-Summe über das Korpus · "
               "Wortfarbe = Tag aus dem Termset")
    c1, c2, c3 = st.columns(3)
    wc_cmap = c1.selectbox("Farbschema (Tags)",
                           ["tab10", "tab20", "Set2", "Dark2", "Paired"],
                           key="wc_cmap")
    wc_whole = c2.checkbox("Nur ganze Wörter zählen", value=True,
                           key="wc_whole")
    wc_title = c3.text_input("Titel", value="Wortwolke", key="wc_title")
    if st.button("Wortwolke erstellen", key="wc_btn", type="primary"):
        try:
            df_terms = store.load_termset()
            tfidf_avg = store.tfidf_averages()
            fig = termset_wordcloud(df_terms, tfidf_avg, cmap=wc_cmap,
                                    whole_word=wc_whole,
                                    title=wc_title or "Wortwolke")
            save_figure(fig, "wortwolke", params={
                "Farbschema": wc_cmap, "Nur ganze Wörter": bool(wc_whole),
                "Titel": wc_title or "Wortwolke"}, key="wc")
        except RuntimeError as e:
            # wordcloud ist optional – Hinweis mit Installationsbefehl
            st.warning(str(e))
        except Exception as e:
            show_error(e)

# ---------------------------------------------------------------------------
# Tab: Dendrogramme – je Vor-Cluster ein Dendrogramm
# ---------------------------------------------------------------------------
with tab_dendro:
    st.subheader("Dendrogramme")
    st.caption("Das Termset wird zunächst grob vorgeclustert (Ward), "
               "anschließend wird für jedes Teilcluster ein eigenes "
               "Dendrogramm gezeichnet – so bleiben die Diagramme lesbar.")
    c1, c2 = st.columns(2)
    dd_k = c1.number_input("Anzahl Vor-Cluster", 1, 20, 3, key="dd_k", help='Das Termset wird zunächst grob in k Cluster (Ward) geteilt; je Teilcluster entsteht ein eigenes, lesbares Dendrogramm.')
    dd_method = c2.selectbox("Linkage-Methode",
                             ["average", "complete", "single", "ward"],
                             key="dd_method", help='Verknüpfungsregel: average = mittlerer, complete = größter, single = kleinster Gruppenabstand; ward minimiert die Varianz (euklidisch).')
    if st.button("Dendrogramme zeichnen", key="dd_btn", type="primary"):
        try:
            kv = models.load()
            df_terms = store.load_termset()
            figs = termset_dendrograms(kv, df_terms, k=int(dd_k),
                                       method=dd_method)
            for cid, fig in figs:
                st.markdown(f"##### Dendrogramm – Cluster {cid}")
                save_figure(fig, f"dendrogramm_cluster_{cid}",
                            params={"Anzahl Vor-Cluster (k)": int(dd_k),
                                    "Linkage-Methode": dd_method},
                            key=f"dd_{cid}")
        except Exception as e:
            show_error(e)
