#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.analysis_vectors
==============================

UI-freie Analysen auf dem Word2Vec-Modell und Termsets. Extrahiert aus den
Blöcken "Wort-Vektor-Modell" und "Termset" des Korpus-Explorers:
ähnlichste Wörter, Embedding-Vergleich, semantisches Netzwerk,
UMAP-Clustering eines Termsets, Wortwolke und Dendrogramme.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------------------
# Embeddings: ähnlichste Wörter & Vergleich
# ----------------------------------------------------------------------------

def most_similar(kv, word: str, top_n: int = 20) -> pd.DataFrame:
    """Ähnlichste Wörter zu einem Wort (Legacy-Tab 'Embeddings')."""
    word = word.strip()
    if not word:
        raise ValueError("Bitte ein Wort angeben.")
    if word not in kv:
        raise ValueError(f"'{word}' ist nicht im Modell enthalten.")
    similar = kv.most_similar(word, topn=top_n)
    df = pd.DataFrame(similar, columns=["word", "similarity"])
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def compare_embeddings(kv, central: str, comparisons: List[str],
                       top_n: int = 50, threshold: float = 0.3
                       ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Vergleicht ein zentrales Wort mit mehreren Ausdrücken über
    gemeinsame Nachbarn (Legacy-Tab 'Embeddings Vergleich').

    Returns
    -------
    (uebersicht, details): Übersichtstabelle und pro Vergleichswort eine
    Tabelle der gemeinsamen Nachbarn.
    """
    from scipy.spatial.distance import cosine

    central = central.strip()
    comps = [w.strip() for w in comparisons if w.strip()]
    if not central or not comps:
        raise ValueError("Bitte Zentral- und Vergleichsausdrücke angeben.")
    if central not in kv:
        raise ValueError(f"'{central}' ist nicht im Modell enthalten.")

    central_vec = kv[central]
    central_neighbors = dict(kv.most_similar(central, topn=top_n))

    rows_main, details = [], {}
    for wort in comps:
        if wort not in kv:
            rows_main.append((wort, np.nan, 0))
            details[wort] = pd.DataFrame(
                columns=["nachbar", f"sim_{central}", f"sim_{wort}"])
            continue
        score = 1 - cosine(central_vec, kv[wort])
        wort_neighbors = dict(kv.most_similar(wort, topn=top_n))
        gemeinsame = []
        for gw in set(central_neighbors) & set(wort_neighbors):
            sim_a, sim_b = central_neighbors[gw], wort_neighbors[gw]
            if sim_a >= threshold and sim_b >= threshold:
                gemeinsame.append((gw, round(sim_a, 4), round(sim_b, 4)))
        gemeinsame.sort(key=lambda x: -(x[1] + x[2]))
        details[wort] = pd.DataFrame(
            gemeinsame, columns=["nachbar", f"sim_{central}", f"sim_{wort}"])
        rows_main.append((wort, round(float(score), 4), len(gemeinsame)))

    uebersicht = pd.DataFrame(rows_main, columns=["vergleich", "score", "anzahl"])
    return uebersicht, details


# ----------------------------------------------------------------------------
# Semantisches Netzwerk
# ----------------------------------------------------------------------------

_NETWORK_SIZES = {
    "Klein": ((14, 14), 300, 10),
    "Mittel": ((20, 16), 400, 12),
    "Groß": ((28, 20), 500, 14),
}


def semantic_network(kv, words: List[str], top_n: int = 8,
                     threshold: float = 0.3, resolution: str = "Klein"
                     ) -> plt.Figure:
    """Semantisches Netzwerk aus Word2Vec-Nachbarn (Legacy-Tab 'Netzwerk')."""
    import networkx as nx

    words = [w.strip() for w in words if w.strip()]
    words = [w for w in words if w in kv]
    if not words:
        raise ValueError("Keines der Wörter ist im Modell enthalten.")
    if not 0 <= threshold <= 1:
        raise ValueError("Kosinus-Schwelle muss zwischen 0 und 1 liegen.")

    G = nx.Graph()
    for word in words:
        G.add_node(word, is_seed=True)
    for word in words:
        for neighbor, sim in kv.most_similar(word, topn=top_n):
            if sim >= threshold:
                G.add_node(neighbor, is_seed=False)
                G.add_edge(word, neighbor, weight=sim)
    for w1, w2 in itertools.combinations(words, 2):
        sim = float(kv.similarity(w1, w2))
        if sim >= threshold:
            G.add_edge(w1, w2, weight=sim)

    if G.number_of_edges() == 0:
        raise ValueError("Keine Verbindungen über der Schwelle gefunden.")

    figsize, node_size, font_size = _NETWORK_SIZES.get(resolution, _NETWORK_SIZES["Klein"])
    pos = nx.spring_layout(G, seed=42, k=0.4)
    fig, ax = plt.subplots(figsize=figsize)
    seeds = [n for n in G.nodes() if G.nodes[n].get("is_seed")]
    others = [n for n in G.nodes() if not G.nodes[n].get("is_seed")]
    nx.draw_networkx_nodes(G, pos, nodelist=seeds, node_color="coral",
                           node_size=node_size * 1.5, ax=ax, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=others, node_color="lightblue",
                           node_size=node_size, ax=ax, alpha=0.7)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)
    ax.set_title(f"Netzwerk: {', '.join(words[:5])}{'…' if len(words) > 5 else ''}"
                 f" | Top-N: {top_n} | Schwelle: {threshold}")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Termset-Hilfen
# ----------------------------------------------------------------------------

def termset_words(df_terms: pd.DataFrame, kv=None) -> List[str]:
    """Alle Termset-Wörter (Groß-/Kleinschreibung erhalten); optional auf das
    Modell-Vokabular gefiltert.

    Hinweis: Da das Modell jetzt case-sensitiv ist, müssen die Termset-Einträge
    in der passenden Schreibweise vorliegen (Nomen groß), sonst werden sie beim
    Abgleich gegen das Vokabular nicht gefunden.
    """
    all_terms = set()
    for c in df_terms.columns:
        all_terms.update(df_terms[c].dropna().astype(str).str.strip().tolist())
    terms = sorted(t for t in all_terms if t)
    if kv is not None:
        terms = [t for t in terms if t in kv]
    return terms


# ----------------------------------------------------------------------------
# UMAP-Clustering eines Termsets
# ----------------------------------------------------------------------------

_CLUSTER_SIZES = {
    "Klein": ((12, 8), 80, 8),
    "Mittel": ((16, 12), 100, 10),
    "Groß": ((20, 16), 120, 12),
}


def cluster_termset(kv, df_terms: pd.DataFrame, k: int = 5,
                    n_neighbors: int = 15, min_dist: float = 0.1,
                    resolution: str = "Klein", show_labels: bool = True
                    ) -> Tuple[plt.Figure, pd.DataFrame, Dict[int, List[str]]]:
    """UMAP-Streudiagramm mit agglomerativem Clustering eines Termsets.

    Robust gemacht:
    - Das agglomerative Clustering läuft auf Kosinus-Distanzen; die
      sklearn-Argumentbezeichnung (``metric`` ab 1.2, früher ``affinity``)
      wird abgefangen.
    - Für die 2D-Projektion wird UMAP versucht; schlägt der Import oder der
      Lauf fehl (häufige umap/numba/numpy-Kompatibilitätsklemme), wird
      deterministisch auf PCA zurückgefallen, statt abzubrechen. Die
      tatsächlich verwendete Methode steht im Diagrammtitel.
    """
    from sklearn.cluster import AgglomerativeClustering

    terms = termset_words(df_terms, kv)
    if len(terms) < 3:
        raise ValueError("Zu wenige Termset-Wörter im Modell (mindestens 3 nötig).")

    k = min(max(1, k), len(terms))
    vectors = np.array([kv[t] for t in terms])

    # --- Clustering auf Kosinus-Distanz (sklearn-API-robust) ----------------
    try:
        clustering = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average")
        labels = clustering.fit_predict(vectors)
    except TypeError:
        # ältere sklearn-Versionen: 'affinity' statt 'metric'
        clustering = AgglomerativeClustering(
            n_clusters=k, affinity="cosine", linkage="average")
        labels = clustering.fit_predict(vectors)

    # --- 2D-Projektion: UMAP, sonst PCA-Fallback ----------------------------
    proj_method = "UMAP"
    coords = None
    try:
        import umap
        nn = min(max(2, n_neighbors), len(terms) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist,
                            metric="cosine", random_state=42)
        coords = reducer.fit_transform(vectors)
    except Exception:
        # UMAP nicht verfügbar oder Laufzeitfehler (numba/numpy) -> PCA
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
        proj_method = "PCA (UMAP nicht verfügbar)"

    # --- Zeichnen ------------------------------------------------------------
    figsize, marker_size, font_size = _CLUSTER_SIZES.get(resolution, _CLUSTER_SIZES["Klein"])
    fig, ax = plt.subplots(figsize=figsize)
    # Farbpalette, die auch für k > 10 verschiedene Farben liefert.
    if k <= 10:
        palette = plt.cm.tab10(np.linspace(0, 1, 10))
    elif k <= 20:
        palette = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        palette = plt.cm.hsv(np.linspace(0, 1, k, endpoint=False))
    for i in range(k):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[i % len(palette)]],
                   label=f"Cluster {i + 1}", s=marker_size, alpha=0.7)
    if show_labels:
        for i, t in enumerate(terms):
            ax.annotate(t, (coords[i, 0], coords[i, 1]), fontsize=font_size, alpha=0.8)
    ax.set_title(f"Termset-Clustering (k={k}, n={len(terms)}, Projektion: {proj_method})")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()

    clusters: Dict[int, List[str]] = defaultdict(list)
    for t, l in zip(terms, labels):
        clusters[int(l)].append(t)

    df_result = pd.DataFrame({"term": terms, "cluster": labels + 1,
                              "x": coords[:, 0], "y": coords[:, 1]})
    return fig, df_result, dict(clusters)


# ----------------------------------------------------------------------------
# Wortwolke (Termset × TF-IDF)
# ----------------------------------------------------------------------------

def termset_wordcloud(df_terms: pd.DataFrame, tfidf_avg: pd.DataFrame,
                      cmap: str = "tab10", whole_word: bool = True,
                      title: str = "Wortwolke") -> plt.Figure:
    """Wortwolke aus TF-IDF-Werten, gefärbt nach Termset-Tags
    (Legacy-Tab 'Wortwolke'). Benötigt das optionale Paket ``wordcloud``."""
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise RuntimeError("Paket 'wordcloud' fehlt: pip install wordcloud") from exc
    import matplotlib.colors as mcolors

    avg = tfidf_avg.rename(columns={"term": "word"})[["word", "tfidf_avg"]]

    word_infos = []
    for tag in df_terms.columns:
        for word in df_terms[tag].dropna():
            word = str(word).strip()
            if whole_word:
                val = avg.loc[avg["word"] == word, "tfidf_avg"]
            else:
                val = avg.loc[avg["word"].str.contains(word, case=True, na=False,
                                                       regex=False), "tfidf_avg"]
            if not val.empty:
                word_infos.append({"word": word, "tag": tag, "tfidf": float(val.values[0])})

    if not word_infos:
        raise ValueError("Keine Überschneidung zwischen Termset und TF-IDF.")

    df_combined = pd.DataFrame(word_infos)
    tags = df_combined["tag"].unique()
    
    # -----------------------------------------------------------------
    # AKTUALISIERTER BEREICH: Moderne Colormap-Zuweisung
    # -----------------------------------------------------------------
    try:
        colormap = mpl.colormaps[cmap]
    except Exception:
        colormap = mpl.colormaps["tab10"]
    
    # colormap(i / max(1, len(tags)-1)) sorgt dafür, dass die Werte 
    # sauber zwischen 0.0 und 1.0 für die Palette normalisiert werden.
    num_tags = max(1, len(tags) - 1)
    tag_colors = {tag: mcolors.rgb2hex(colormap(i / num_tags)) for i, tag in enumerate(tags)}
    # -----------------------------------------------------------------

    def color_func(word, *args, **kwargs):
        row = df_combined[df_combined["word"] == word]
        return tag_colors.get(row.iloc[0]["tag"], "black") if not row.empty else "black"

    sizes = df_combined.groupby("word")["tfidf"].max().to_dict()
    sizes_scaled = {w: np.log(v + 1.0) for w, v in sizes.items()}

    wc = WordCloud(width=1200, height=600, background_color="white",
                    prefer_horizontal=1.0)
    wc.generate_from_frequencies(sizes_scaled)
    wc.recolor(color_func=color_func)

    fig = plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Dendrogramme pro Cluster
# ----------------------------------------------------------------------------

def termset_dendrograms(kv, df_terms: pd.DataFrame, k: int = 3,
                        method: str = "average") -> List[Tuple[int, plt.Figure]]:
    """Hierarchisches Clustering eines Termsets, je Cluster ein Dendrogramm
    (Legacy-Tab 'Dendrogramme'). Gibt (Cluster-ID, Figur)-Paare zurück."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering

    # Termset-Einträge können Zusätze in Klammern haben: "wort (anm.)" → "wort"
    all_terms: Dict[str, str] = {}
    for tag in df_terms.columns:
        for entry in df_terms[tag].dropna().astype(str):
            word = re.sub(r"\s*\(.*?\)\s*$", "", entry.strip())
            if word and word in kv:
                all_terms[word] = entry.strip()

    if len(all_terms) < 2:
        raise ValueError("Zu wenige Termset-Wörter im Modell (mindestens 2 nötig).")

    words = list(all_terms)
    labels = [all_terms[w] for w in words]
    vectors = np.array([kv[w] for w in words])

    k_eff = min(max(1, k), len(words))
    if k_eff == 1:
        cluster_labels = np.zeros(len(words), dtype=int)
    else:
        cluster_labels = AgglomerativeClustering(
            n_clusters=k_eff, linkage="ward").fit_predict(vectors)

    figures = []
    for cluster_id in range(k_eff):
        idx = np.where(cluster_labels == cluster_id)[0]
        if len(idx) < 2:
            continue
        cluster_label_list = [labels[i] for i in idx]
        cluster_vecs = vectors[idx]
        if method == "ward":
            Z = linkage(cluster_vecs, method="ward")
        else:
            Z = linkage(pdist(cluster_vecs, metric="cosine"), method=method)
        fig_height = max(6, len(cluster_label_list) * 0.3)
        fig = plt.figure(figsize=(10, fig_height))
        dendrogram(Z, labels=cluster_label_list, orientation="right", leaf_font_size=9)
        plt.title(f"Dendrogramm — Cluster {cluster_id}")
        fig.tight_layout()
        figures.append((cluster_id, fig))
    return figures
