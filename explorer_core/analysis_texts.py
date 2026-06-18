#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.analysis_texts
============================

UI-freie Analysen auf Textebene: UMAP-Projektion der Kosinus-Matrix und
Streudiagramme der Texte (interaktiv mit Plotly, statisch mit Matplotlib).
Extrahiert aus den Tabs "Streudiagramm 1" und "Streudiagramm 2" des
Korpus-Explorers. Die Hover-/Legenden-Spalten sind nicht mehr hartkodiert
(früher fest ``author_surname``/``title``), sondern frei wählbar.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCATTER_METHODS = ("pca", "mds", "tsne", "umap")


def _prepare_distance(cosine_df: pd.DataFrame, metadata: pd.DataFrame):
    """Gemeinsame Vorbereitung: Ähnlichkeits-/Distanzmatrix + ausgerichtete
    Metadaten. Distanzen = 1 − Kosinus-Ähnlichkeit (symmetrisch, 0-Diagonale)."""
    sim = cosine_df.values.astype(float)
    doc_ids = [str(d) for d in cosine_df.index.tolist()]

    meta = metadata.copy()
    if "doc_id" not in meta.columns:
        raise ValueError("Metadaten enthalten keine doc_id-Spalte.")
    meta = meta[meta["doc_id"].isin(doc_ids)]
    meta = meta.set_index("doc_id").reindex(doc_ids).reset_index()

    distance = 1.0 - sim
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    distance = (distance + distance.T) / 2.0  # Symmetrie erzwingen
    return sim, distance, meta, doc_ids


def _cluster_from_distance(distance: np.ndarray, k: int, method: str):
    """Agglomeratives Clustering auf den vorberechneten Distanzen (0-basierte
    Labels). ``ward`` über MDS-Einbettung (euklidisch), sonst direkt auf den
    Kosinus-Distanzen via scipy. Gibt None zurück, wenn k < 2."""
    n = len(distance)
    k_eff = min(int(k), n)
    if k_eff < 2:
        return None
    if method == "ward":
        from sklearn.manifold import MDS
        from sklearn.cluster import AgglomerativeClustering
        mds = MDS(n_components=min(50, n - 1), dissimilarity="precomputed",
                  random_state=42)
        embedded = mds.fit_transform(distance)
        return AgglomerativeClustering(
            n_clusters=k_eff, linkage="ward").fit_predict(embedded)
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
    condensed = squareform(distance, checks=False)
    Z = scipy_linkage(condensed, method=method)
    return fcluster(Z, k_eff, criterion="maxclust") - 1


def _project(method: str, sim: np.ndarray, distance: np.ndarray,
             hp: dict) -> np.ndarray:
    """2D-Projektion mit der gewählten Methode.

    - **PCA**  arbeitet auf den Ähnlichkeitsprofilen (Zeilen der Kosinus-Matrix)
      als Merkmalsvektoren.
    - **MDS/TSNE/UMAP** arbeiten auf den vorberechneten Kosinus-Distanzen
      (``metric/dissimilarity='precomputed'``).
    """
    method = method.lower()
    n = len(distance)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, svd_solver=hp.get("svd_solver", "auto"),
                   whiten=bool(hp.get("whiten", False)),
                   random_state=42).fit_transform(sim)
    if method == "mds":
        from sklearn.manifold import MDS
        kw = dict(n_components=2, dissimilarity="precomputed",
                  n_init=int(hp.get("n_init", 4)),
                  max_iter=int(hp.get("max_iter", 300)),
                  metric=bool(hp.get("metric", True)), random_state=42)
        try:
            return MDS(normalized_stress="auto", **kw).fit_transform(distance)
        except TypeError:
            return MDS(**kw).fit_transform(distance)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = float(hp.get("perplexity", 30.0))
        # perplexity muss < n_samples sein (Faustregel < (n-1)/3).
        perp = max(2.0, min(perp, (n - 1) / 3.0)) if n > 6 else max(1.0, min(perp, n - 1.5))
        common = dict(n_components=2, metric="precomputed", init="random",
                      perplexity=perp, learning_rate=hp.get("learning_rate", "auto"),
                      random_state=42)
        n_iter = int(hp.get("n_iter", 1000))
        try:  # sklearn >=1.5: max_iter; älter: n_iter
            return TSNE(max_iter=n_iter, **common).fit_transform(distance)
        except TypeError:
            return TSNE(n_iter=n_iter, **common).fit_transform(distance)
    if method == "umap":
        import umap
        nn = min(max(2, int(hp.get("n_neighbors", 15))), n - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=nn,
                            min_dist=float(hp.get("min_dist", 0.1)),
                            metric="precomputed", random_state=42)
        return reducer.fit_transform(distance)
    raise ValueError(f"Unbekannte Reduktionsmethode: {method!r}")


def reduce_from_cosine(cosine_df: pd.DataFrame, metadata: pd.DataFrame,
                       method: str = "umap", *,
                       cluster_k: Optional[int] = None,
                       cluster_method: str = "average",
                       hp: Optional[dict] = None) -> pd.DataFrame:
    """2D-Projektion der Kosinus-Matrix mit PCA/MDS/TSNE/UMAP, optional
    geclustert.

    Returns
    -------
    DataFrame mit ``DIM-1``/``DIM-2``, allen Metadaten und – falls
    ``cluster_k`` gesetzt – ``cluster`` (0-basiert) und ``cluster_label``
    (geordnete Kategorie ab 1).
    """
    sim, distance, meta, doc_ids = _prepare_distance(cosine_df, metadata)
    coords = _project(method, sim, distance, hp or {})

    df = pd.DataFrame({"DIM-1": coords[:, 0], "DIM-2": coords[:, 1]})
    df = df.join(meta.reset_index(drop=True))

    if cluster_k:
        labels = _cluster_from_distance(distance, cluster_k, cluster_method)
        if labels is not None:
            df["cluster"] = labels
            n_clusters = int(df["cluster"].max()) + 1
            ordered = [f"Cluster {i + 1}" for i in range(n_clusters)]
            df["cluster_label"] = pd.Categorical(
                "Cluster " + (df["cluster"] + 1).astype(str),
                categories=ordered, ordered=True,
            )
    return df


def umap_from_cosine(cosine_df: pd.DataFrame, metadata: pd.DataFrame,
                     n_neighbors: int = 15, min_dist: float = 0.1,
                     cluster_k: Optional[int] = None,
                     cluster_method: str = "average") -> pd.DataFrame:
    """Rückwärtskompatibel: UMAP-Projektion über ``reduce_from_cosine``."""
    return reduce_from_cosine(
        cosine_df, metadata, method="umap",
        cluster_k=cluster_k, cluster_method=cluster_method,
        hp={"n_neighbors": n_neighbors, "min_dist": min_dist})


def text_scatter_plotly(umap_df: pd.DataFrame, color_column: Optional[str],
                        hover_columns: List[str], marker_size: int = 8):
    """Interaktives Plotly-Streudiagramm der Texte (einheitliche Darstellung:
    keine Achsen/Beschriftung/Rahmen, weißer Hintergrund)."""
    import plotly.express as px

    df = umap_df.copy()

    def hover_text(row) -> str:
        parts = [str(row.get("doc_id", "N/A"))]
        for col in hover_columns:
            if col in df.columns:
                parts.append(f"{col}: {row.get(col, 'N/A')}")
        return "<br>".join(parts)

    df["hover_text"] = df.apply(hover_text, axis=1)

    if color_column and color_column in df.columns:
        col = df[color_column]
        if isinstance(col.dtype, pd.CategoricalDtype):
            cats = [str(c) for c in col.cat.categories]
        else:
            cats = sorted(col.dropna().astype(str).unique().tolist())
        df["_color"] = col.astype(str).fillna("?")
        fig = px.scatter(df, x="DIM-1", y="DIM-2", color="_color",
                         hover_name="hover_text", opacity=0.7,
                         category_orders={"_color": cats})
        fig.update_layout(legend_title_text=color_column)
    else:
        fig = px.scatter(df, x="DIM-1", y="DIM-2", hover_name="hover_text",
                         opacity=0.7)

    fig.update_traces(marker=dict(size=marker_size))
    # Einheitliche Darstellung (Aufgabe 3)
    axis_off = dict(visible=False, showgrid=False, zeroline=False,
                    showticklabels=False, title_text="")
    fig.update_xaxes(**axis_off)
    fig.update_yaxes(**axis_off)
    fig.update_layout(
        height=750, title=None,
        paper_bgcolor="white", plot_bgcolor="white",
        # Schriftfarbe explizit dunkel setzen – sonst rendert die Legende
        # (Cluster 1, Cluster 2, …) je nach Plotly-Template weiß auf weiß und
        # ist unlesbar.
        font=dict(color="#222222"),
        legend=dict(font=dict(color="#222222"),
                    title=dict(font=dict(color="#222222"))),
        margin=dict(l=10, r=10, t=10, b=10))
    return fig


def text_scatter_matplotlib(umap_df: pd.DataFrame, color_column: Optional[str],
                            marker_size: int = 8,
                            label_column: Optional[str] = None,
                            max_labels: int = 200) -> plt.Figure:
    """Statisches Matplotlib-Streudiagramm der Texte (PNG-Export).

    Einheitliche Darstellung (Aufgabe 3): kein Rahmen, keine Achsen, keine
    Achsenbeschriftungen, weißer Hintergrund. ``label_column`` (optional)
    beschriftet die Punkte mit einer Metadaten-Spalte; bei mehr als
    ``max_labels`` Punkten wird die Beschriftung übersprungen.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if color_column and color_column in umap_df.columns:
        col = umap_df[color_column]
        if isinstance(col.dtype, pd.CategoricalDtype):
            unique_cats = [str(c) for c in col.cat.categories]
        else:
            unique_cats = sorted(col.dropna().astype(str).unique())
        categories = col.astype(str).fillna("?")
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, min(len(unique_cats), 20))))
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_cats)}
        for cat in unique_cats:
            mask = categories == cat
            ax.scatter(umap_df.loc[mask, "DIM-1"], umap_df.loc[mask, "DIM-2"],
                       c=[color_map[cat]], label=str(cat)[:35],
                       s=marker_size * 10, alpha=0.7)
        if len(unique_cats) <= 20:
            ax.legend(loc="upper right", fontsize=8)
    else:
        ax.scatter(umap_df["DIM-1"], umap_df["DIM-2"], s=marker_size * 10, alpha=0.7)

    if label_column and label_column in umap_df.columns and len(umap_df) <= max_labels:
        for _, row in umap_df.iterrows():
            txt = row[label_column]
            if pd.isna(txt):
                continue
            ax.annotate(str(txt)[:40], (row["DIM-1"], row["DIM-2"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_axis_off()  # keine Achsen, Ticks, Beschriftungen, kein Rahmen
    fig.tight_layout()
    return fig


def text_dendrograms(cosine_df: pd.DataFrame,
                     label_map: Optional[dict] = None,
                     k: int = 3,
                     linkage_method: str = "average",
                     metric: str = "precomputed") -> List[Tuple[int, "plt.Figure"]]:
    """Globales Dendrogramm der Texte mit farblich indizierten k Clustern.

    Die Texte besitzen keine rohen Merkmalsvektoren, sondern nur die
    vorberechnete Kosinus-*Ähnlichkeits*-Matrix (Pipeline-Schritt s04).
    Deshalb werden hier IMMER zuerst Kosinus-*Distanzen* aus der Matrix
    berechnet (``1 − Ähnlichkeit``) und das Clustering läuft auf diesen
    vorberechneten Distanzen (``metric='precomputed'``). Der frühere
    ``cosine``/``euclidean``-Pfad rechnete versehentlich auf den *Zeilen*
    der Ähnlichkeitsmatrix (= Ähnlichkeitsprofile, keine Dokumentvektoren)
    und lieferte dadurch unbrauchbare Distanzen.

    Der ``metric``-Parameter bleibt aus Kompatibilitätsgründen erhalten,
    wird aber ignoriert: Für Texte ist nur ``precomputed`` sinnvoll.
    ``linkage='ward'`` benötigt euklidische Daten und wird daher über eine
    MDS-Einbettung der Distanzmatrix realisiert.

    Rückgabe: ``[(0, fig)]`` – eine einelementige Liste, damit die
    ``for cid, fig in figs:``-Schleife im Dashboard unverändert funktioniert.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    from scipy.cluster.hierarchy import (linkage as scipy_linkage, dendrogram,
                                         set_link_color_palette)
    from scipy.spatial.distance import squareform

    ids = [str(d) for d in cosine_df.index.tolist()]
    n = len(ids)
    if n < 2:
        raise ValueError("Zu wenige Texte für ein Dendrogramm (mindestens 2 nötig).")

    sim = cosine_df.values.astype(float)
    labels = [(label_map or {}).get(i, i) for i in ids]

    # 1. Kosinus-Distanzen aus der Kosinus-Matrix vorberechnen (precomputed).
    distance_matrix = 1.0 - sim
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, None)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0  # Symmetrie erzwingen

    # 2. Linkage-Baum auf den vorberechneten Distanzen.
    if linkage_method == "ward":
        # ward verlangt euklidische Merkmale -> MDS-Einbettung der Distanzen.
        from sklearn.manifold import MDS
        mds = MDS(n_components=min(50, n - 1), dissimilarity="precomputed",
                  random_state=42)
        embedded = mds.fit_transform(distance_matrix)
        Z = scipy_linkage(embedded, method="ward")
    else:
        condensed = squareform(distance_matrix, checks=False)
        Z = scipy_linkage(condensed, method=linkage_method)

    # 3. Robuste Farbschwelle: Mitte zwischen dem (k-1)-ten und k-ten größten
    #    Verschmelzungsschritt – so entstehen zuverlässig genau k Farbgruppen
    #    (die alte Variante setzte die Schwelle exakt auf eine Merge-Höhe und
    #    war dadurch je nach Gleichstand um ein Cluster daneben).
    k_eff = max(2, min(int(k), n))
    if 1 < k_eff < n:
        hi = Z[-(k_eff - 1), 2]
        lo = Z[-k_eff, 2]
        color_threshold = (hi + lo) / 2.0 if hi > lo else hi * 0.999
    else:
        color_threshold = 0.0

    # 4. Zeichnen mit fester Farbpalette; oberhalb der Schwelle neutral grau.
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#393b79"]
    fig_height = max(6, n * 0.3)
    fig = plt.figure(figsize=(10, fig_height))
    ax = fig.add_subplot(111)

    set_link_color_palette(palette)
    try:
        dn = dendrogram(
            Z,
            labels=labels,
            orientation="right",
            leaf_font_size=9,
            color_threshold=color_threshold,
            above_threshold_color="lightgrey",
            ax=ax,
        )
    finally:
        set_link_color_palette(None)  # globalen scipy-Zustand zurücksetzen

    # 5. Legende aus den TATSÄCHLICH gezeichneten Blattfarben aufbauen, damit
    #    Legende und Diagramm garantiert übereinstimmen (genau das fehlte).
    leaf_colors = dn["leaves_color_list"]
    order: List[str] = []
    for c in leaf_colors:
        if c != "lightgrey" and c not in order:
            order.append(c)
    sizes = Counter(c for c in leaf_colors if c != "lightgrey")
    if order:
        handles = [plt.Line2D([0], [0], color=c, lw=4) for c in order]
        legend_labels = [f"Cluster {i + 1} (n={sizes[c]})" for i, c in enumerate(order)]
        ax.legend(handles, legend_labels, loc="lower right", fontsize=8,
                  title=f"{len(order)} Cluster", framealpha=0.9)

    ax.set_title(f"Dendrogramm der Texte – {n} Dokumente, k={len(order) or k_eff} "
                 f"(Kosinus-Distanz, precomputed)")
    ax.set_xlabel("Kosinus-Distanz (1 − Ähnlichkeit)")
    fig.tight_layout()

    return [(0, fig)]