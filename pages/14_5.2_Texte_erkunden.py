#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 4: Texte
==============

Zwei Ansichten auf die Dokument-Ähnlichkeit (Pipeline-Schritt s04, Kosinus-
Matrix):

- **Streudiagramm**: 2D-Projektion der Kosinus-Matrix mit vier Verfahren in
  eigenen Subtabs – PCA, MDS, t-SNE und UMAP. Jeder Subtab bietet
  Hyperparameter, optionales hierarchisches Clustering (aus der Kosinus-Matrix)
  sowie interaktive (Plotly) und statische (Matplotlib) Darstellung. Alle
  Streudiagramme haben dieselbe Darstellung (keine Achsen/Beschriftung/Rahmen,
  weißer Hintergrund).
- **Dendrogramme**: agglomeratives Clustering der Texte (precomputed
  Kosinus-Distanzen).

Zu jeder gespeicherten Grafik werden die verwendeten Hyperparameter als
gleichnamige ``.txt`` mit ausgegeben.
"""

import streamlit as st

from ui_helpers import (get_store, get_schema, show_error, df_with_download,
                        metadata_multiselect)
from explorer_core.analysis_texts import (reduce_from_cosine,
                                          text_scatter_plotly,
                                          text_scatter_matplotlib,
                                          text_dendrograms)
from explorer_core.viz_export import save_figure, format_hyperparams

st.set_page_config(page_title="Texte erkunden", layout="wide")
st.title("📚 Texte")
st.caption("Ähnlichkeitslandkarte und Dendrogramme der Dokumente "
           "(aus der Kosinus-Matrix)")

store = get_store()
schema = get_schema()


@st.cache_data(show_spinner="Projektion wird berechnet …")
def _cached_reduce(method, cosine_df, meta, cluster_k, cluster_method, hp_items):
    """Cacht die 2D-Projektion. Vor allem UMAP ist beim ersten Aufruf teuer
    (einmalige Numba-Kompilierung); identische Streudiagramme (gleiche Methode,
    Daten und Hyperparameter) werden danach nicht neu gerechnet, sondern aus dem
    Cache geliefert."""
    return reduce_from_cosine(cosine_df, meta, method=method,
                              cluster_k=cluster_k, cluster_method=cluster_method,
                              hp=dict(hp_items))


# ---------------------------------------------------------------------------
# Gemeinsame Streudiagramm-Logik für alle vier Verfahren (PCA/MDS/TSNE/UMAP)
# ---------------------------------------------------------------------------
def _scatter_subtab(method: str, facet_options, label_meta_cols):
    """Rendert einen vollständigen Streudiagramm-Subtab für ein Verfahren."""
    mlabel = method.upper()

    with st.expander("⚙️ Parameter", expanded=True):
        c1, c2, c3 = st.columns(3)
        marker_size = c1.number_input("Punktgröße", 2, 30, 8,
                                      key=f"tx_ms_{method}")
        do_cluster = c2.checkbox("Texte clustern", key=f"tx_do_{method}")

        # --- methodenspezifische Hyperparameter ---
        hp: dict = {}
        if method == "umap":
            hp["n_neighbors"] = c3.number_input(
                "UMAP n_neighbors", 2, 200, 15, key=f"tx_nb_{method}", help='Zahl der Nachbarn, die UMAP zur Schätzung der lokalen Struktur heranzieht. **Kleine** Werte (z. B. 5–15) betonen feine, lokale Nachbarschaften; **große** Werte (50–100) bewahren stärker die globale Gesamtstruktur. Muss kleiner als die Anzahl der Texte sein.')
            hp["min_dist"] = st.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.05,
                                       key=f"tx_md_{method}", help='Mindestabstand, den Punkte in der 2D-Projektion zueinander haben dürfen. **Kleine** Werte (≈0) packen Punkte dicht in Cluster zusammen; **große** Werte (bis 1) verteilen sie gleichmäßiger und betonen die grobe Anordnung.')
            st.caption("ℹ️ Der **erste** UMAP-Aufruf nach App-Start kompiliert "
                       "einmalig (~30 s); danach ist das Ergebnis gecacht und "
                       "erscheint sofort. PCA/MDS/t-SNE sind ohne Wartezeit.")
        elif method == "pca":
            hp["svd_solver"] = c3.selectbox(
                "PCA svd_solver", ["auto", "full", "randomized"],
                key=f"tx_svd_{method}", help='Verfahren zur Singulärwertzerlegung. **auto** wählt automatisch; **full** rechnet exakt (genau, aber langsamer bei vielen Texten); **randomized** approximiert schnell und eignet sich für große Matrizen. Das Ergebnis ist bei allen praktisch gleich.')
            hp["whiten"] = st.checkbox("PCA whiten", value=False,
                                       key=f"tx_wh_{method}", help='Das sogenannte Weißen skaliert die beiden Hauptkomponenten auf gleiche Varianz. Macht die Achsen gleich gewichtet (oft rundere Punktwolke), verändert aber die relativen Abstände. Für reine Visualisierung meist **aus**.')
        elif method == "mds":
            hp["n_init"] = c3.number_input("MDS n_init", 1, 20, 4,
                                           key=f"tx_ni_{method}", help='Anzahl der Neustarts mit zufälliger Startanordnung. MDS kann in lokalen Minima landen; mehr Läufe (höhere Werte) liefern ein stabileres, besseres Ergebnis, kosten aber mehr Rechenzeit. Das beste Ergebnis wird behalten.')
            m1, m2 = st.columns(2)
            hp["max_iter"] = m1.number_input("MDS max_iter", 50, 2000, 300,
                                             step=50, key=f"tx_mi_{method}", help='Maximale Iterationen pro Lauf, in denen MDS die Anordnung optimiert. Höhere Werte erlauben bessere Konvergenz (genauere Abstandstreue), dauern aber länger.')
            hp["metric"] = m2.checkbox("MDS metrisch", value=True,
                                       key=f"tx_me_{method}", help='**An** = metrisches MDS: bildet die Kosinus-Distanzen möglichst maßstabsgetreu ab. **Aus** = nicht-metrisches MDS: bewahrt nur die *Rangordnung* der Distanzen – robuster, wenn nur die Reihenfolge der Ähnlichkeiten verlässlich ist.')
        elif method == "tsne":
            hp["perplexity"] = c3.number_input("t-SNE perplexity", 2.0, 100.0,
                                               30.0, key=f"tx_pp_{method}", help='Grob die erwartete Nachbarschaftsgröße je Punkt (typisch 5–50). **Klein** betont sehr lokale Grüppchen, **groß** berücksichtigt mehr globale Struktur. Muss kleiner als die Anzahl der Texte sein (wird sonst automatisch begrenzt).')
            t1, t2 = st.columns(2)
            hp["learning_rate"] = t1.number_input(
                "t-SNE learning_rate", 10.0, 1000.0, 200.0,
                key=f"tx_lr_{method}", help='Schrittweite der Optimierung. **Zu klein** → Punkte bleiben in einem dichten Klumpen; **zu groß** → Struktur zerfällt. Werte um 200 sind ein üblicher Startpunkt.')
            hp["n_iter"] = t2.number_input("t-SNE n_iter", 250, 5000, 1000,
                                           step=250, key=f"tx_it_{method}", help='Anzahl der Optimierungsschritte. Mehr Iterationen geben dem Layout Zeit, sich zu stabilisieren (mind. ~250); sehr hohe Werte bringen kaum noch Verbesserung, kosten aber Zeit.')

        cc4, cc5 = st.columns(2)
        cluster_k = cc4.number_input("Anzahl Cluster", 2, 50, 5,
                                     key=f"tx_k_{method}",
                                     disabled=not do_cluster, help='In wie viele Gruppen die Texte beim hierarchischen Clustering eingeteilt werden. Das Clustering läuft auf den Kosinus-Distanzen – unabhängig vom Projektionsverfahren – und färbt die Punkte entsprechend ein.')
        cluster_method = cc5.selectbox("Cluster-Methode",
                                       ["average", "complete", "ward"],
                                       key=f"tx_cm_{method}",
                                       disabled=not do_cluster, help='Verknüpfungsregel beim agglomerativen Clustering. **average**: mittlerer Abstand zwischen Gruppen (ausgewogen). **complete**: größter Abstand (kompakte Cluster). **ward**: minimiert die Varianz – nur euklidisch, daher werden die Distanzen vorher per MDS eingebettet.')

        color_options = ["(keine)"]
        if do_cluster:
            color_options.append("Cluster")
        color_options += facet_options
        color_default = color_options.index("Cluster") if do_cluster else 0
        color_by = st.selectbox("Punkte einfärben nach", color_options,
                                index=color_default, key=f"tx_color_{method}")

        hover_cols = metadata_multiselect(
            "Hover-Spalten (interaktive Ansicht)", key=f"tx_hover_{method}")

        render_mode = st.radio(
            "Darstellung", ["Interaktiv (Plotly)", "Statisch (Matplotlib)"],
            horizontal=True, key=f"tx_render_{method}")

        label_by = "(keine)"
        if render_mode.startswith("Statisch"):
            label_by = st.selectbox(
                "Punkte beschriften mit (statisch)",
                ["(keine)"] + label_meta_cols, key=f"tx_label_{method}",
                help="Beschriftet jeden Punkt mit dem Wert dieser Spalte. "
                     "Bei sehr vielen Punkten automatisch übersprungen.")

    if st.button(f"Streudiagramm berechnen ({mlabel})",
                 key=f"tx_btn_{method}", type="primary"):
        try:
            cosine_df = store.load_cosine()
            meta = store.load_metadata()
            df = _cached_reduce(
                method, cosine_df, meta,
                int(cluster_k) if do_cluster else None,
                cluster_method, tuple(sorted(hp.items())))
            st.session_state[f"tx_df_{method}"] = df
        except Exception as e:
            show_error(e)

    if f"tx_df_{method}" in st.session_state:
        df = st.session_state[f"tx_df_{method}"]
        color_col = None
        if color_by == "Cluster":
            if "cluster_label" in df.columns:
                color_col = "cluster_label"
            else:
                st.info("Für die Cluster-Färbung zuerst 'Texte clustern' "
                        "aktivieren und neu berechnen.")
        elif color_by not in ("(keine)", "Cluster"):
            color_col = color_by

        # Hyperparameter für die TXT-Beilage
        params = {"Methode": mlabel, "Anzahl Texte": len(df),
                  "Punktgröße": int(marker_size),
                  "Clustering": (f"{cluster_method}, k={int(cluster_k)}"
                                 if do_cluster else "aus"),
                  "Färbung": color_by}
        params.update({f"hp.{k}": v for k, v in hp.items()})
        base = f"texte_streudiagramm_{method}"

        try:
            if render_mode.startswith("Interaktiv"):
                fig = text_scatter_plotly(df, color_col, hover_cols,
                                          marker_size=int(marker_size))
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(
                    "⬇️ Hyperparameter (TXT)",
                    format_hyperparams(params, base).encode("utf-8"),
                    file_name=f"{base}.txt", mime="text/plain",
                    key=f"tx_hp_{method}")
            else:
                label_col = label_by if label_by != "(keine)" else None
                fig = text_scatter_matplotlib(
                    df, color_col, marker_size=int(marker_size),
                    label_column=label_col)
                save_figure(fig, base, params=params, key=f"tx_fig_{method}")

            with st.expander("Datentabelle (Koordinaten + Metadaten)"):
                df_with_download(df, f"texte_{method}", key=f"tx_table_{method}")
        except RuntimeError as e:
            st.warning(str(e))
        except Exception as e:
            show_error(e)


tab_scatter, tab_dendro = st.tabs(["Streudiagramm", "Dendrogramme"])

# ===========================================================================
# Tab: Streudiagramm – vier Verfahren in eigenen Subtabs
# ===========================================================================
with tab_scatter:
    try:
        meta0 = store.load_metadata()
        facet_options = schema.facet_columns(meta0)
        label_meta_cols = [c for c in meta0.columns
                           if not str(c).lower().startswith(("content", "text"))]
    except FileNotFoundError:
        facet_options = []
        label_meta_cols = []

    sub_pca, sub_mds, sub_tsne, sub_umap = st.tabs(
        ["Streudiagramm-PCA", "Streudiagramm-MDS",
         "Streudiagramm-TSNE", "Streudiagramm-UMAP"])
    with sub_pca:
        _scatter_subtab("pca", facet_options, label_meta_cols)
    with sub_mds:
        _scatter_subtab("mds", facet_options, label_meta_cols)
    with sub_tsne:
        _scatter_subtab("tsne", facet_options, label_meta_cols)
    with sub_umap:
        _scatter_subtab("umap", facet_options, label_meta_cols)

# ===========================================================================
# Tab: Dendrogramme (agglomeratives Clustering der Texte)
# ===========================================================================
with tab_dendro:
    st.subheader("Dendrogramme der Texte")
    st.caption("Ein globales Dendrogramm der Dokumente mit farblich "
               "indizierten k Clustern. Grundlage sind vorberechnete "
               "Kosinus-Distanzen (1 − Ähnlichkeit, metric='precomputed').")

    try:
        meta_dd = store.load_metadata()
        label_choices = ["(Dokument-ID)"] + [
            c for c in meta_dd.columns
            if c not in ("doc_id",) and not str(c).lower().startswith(("content", "text"))
        ]
    except FileNotFoundError:
        meta_dd = None
        label_choices = ["(Dokument-ID)"]

    d1, d2, d3 = st.columns(3)
    dd_k = d1.number_input("Anzahl Cluster", 1, 50, 3, key="dd_tx_k", help='In wie viele Cluster die Texte eingeteilt und im Dendrogramm farblich indiziert werden (Schnitt durch den Baum bei k Gruppen).')
    dd_linkage = d2.selectbox(
        "linkage", ["average", "complete", "ward", "single"], key="dd_tx_link", help='Verknüpfungsregel des agglomerativen Clusterings: average = mittlerer, complete = größter, single = kleinster Gruppenabstand. ward minimiert die Varianz (nur euklidisch) und bettet die Distanzmatrix vorher per MDS ein; alle anderen arbeiten direkt auf den vorberechneten Kosinus-Distanzen.')
    dd_label = d3.selectbox("Beschriftung", label_choices, key="dd_tx_label")
    dd_metric = "precomputed"

    if st.button("Dendrogramme zeichnen", key="dd_tx_btn", type="primary"):
        try:
            cosine_df = store.load_cosine()
            label_map = None
            if dd_label != "(Dokument-ID)" and meta_dd is not None \
                    and "doc_id" in meta_dd.columns and dd_label in meta_dd.columns:
                label_map = {
                    str(d): str(v) for d, v in
                    zip(meta_dd["doc_id"], meta_dd[dd_label])
                }
            st.session_state["tx_dendros"] = text_dendrograms(
                cosine_df, label_map=label_map, k=int(dd_k),
                linkage_method=dd_linkage, metric=dd_metric)
            st.session_state["tx_dendro_params"] = {
                "Anzahl Cluster (k)": int(dd_k), "linkage": dd_linkage,
                "metric": dd_metric, "Beschriftung": dd_label}
        except Exception as e:
            show_error(e)

    if "tx_dendros" in st.session_state:
        figs = st.session_state["tx_dendros"]
        if not figs:
            st.info("Kein Dendrogramm – es werden mindestens 2 Texte benötigt.")
        else:
            params = st.session_state.get("tx_dendro_params", {})
            for cid, fig in figs:
                save_figure(fig, "texte_dendrogramm", params=params,
                            key=f"dd_tx_{cid}")
