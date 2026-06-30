#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.topic_model
==========================

UI-freie Topic-Modellierung mit **scikit-learn** auf den bereits vorhandenen
Term-Matrizen aus ``s03_dtm_tfidf`` (DTM = Zählungen, TF-IDF = gewichtet).

- **NMF** auf TF-IDF: scharfe, interpretierbare Topics (deterministisch).
- **LDA** auf Zählungen (DTM): klassisches probabilistisches Topic-Modell.

Die Ausgaben sind exakt im Format, das die nachgelagerten Skripte erwarten:

- ``document-topics-distribution_<name>.csv`` – Zeilen = Dokument-IDs (Index),
  Spalten = Topics, Werte = Anteile (Zeilensumme 1). Wird von
  ``tt_s02_topics`` mit ``header=0, index_col=0`` gelesen.
- ``<name>_topic_words.csv`` – Zeilen = Topic-IDs (Index), Spalten =
  Rangpositionen, Zellen = Top-Wörter je Topic. Wird von ``s03`` mit
  ``index_col=0`` gelesen.

Eingabe ist die von ``s03_dtm_tfidf`` erzeugte CSV: Metadatenspalten zuerst,
Term-Features ab ``feature_start`` (Pipeline-Konvention: 10), ID in ``id_col``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Paketkontext
    from .corpus_segment import segment_corpus, aggregate_by_source, SOURCE_COLUMN
except Exception:  # Skript-/Testkontext
    from corpus_segment import segment_corpus, aggregate_by_source, SOURCE_COLUMN

METHODS = ("nmf", "lda")


def _feature_block(df: pd.DataFrame, id_col: Optional[str],
                   feature_start: Optional[int]
                   ) -> Tuple[List[str], List[str], np.ndarray]:
    """Zerlegt die Matrix-CSV in (Dokument-IDs, Vokabular, Feature-Matrix X)."""
    if feature_start is not None:
        feats = df.iloc[:, feature_start:]
    else:
        drop = [c for c in (id_col,) if c and c in df.columns]
        feats = df.drop(columns=drop)

    # Dokument-IDs bestimmen
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    elif feature_start:
        ids = df.iloc[:, 0].astype(str).tolist()
    else:
        ids = [str(i) for i in df.index]

    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.loc[:, feats.notna().any(axis=0)].fillna(0.0)
    if feats.shape[1] == 0:
        raise ValueError("Keine numerischen Term-Spalten gefunden – stimmt "
                         "`feature_start`/`id_col`?")
    vocab = [str(c) for c in feats.columns]
    X = feats.to_numpy(dtype=float)
    return ids, vocab, X


def _fit_on_matrix(X, vocab: List[str], ids: List[str], n_topics: int,
                   method: str, top_words: int, random_state: int,
                   max_iter: Optional[int], extra_params: Optional[dict]
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Kern: fittet NMF/LDA auf einer (dichten oder dünnen) Matrix X."""
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unbekannte Methode {method!r} (nmf|lda).")

    n_docs, n_feats = X.shape
    k = max(1, int(n_topics))
    if method == "nmf":
        init = str((extra_params or {}).get("init", "nndsvda"))
        if init.startswith("nndsvd"):
            # nndsvd-Initialisierung erlaubt höchstens rang-viele Komponenten
            k = min(k, n_docs, n_feats)
    # LDA und NMF mit init='random': keine dokumentbasierte Begrenzung

    base = dict(n_components=k, random_state=random_state)
    if method == "nmf":
        from sklearn.decomposition import NMF
        base.setdefault("init", "nndsvda")
        base["max_iter"] = int(max_iter) if max_iter else 400
        if extra_params:
            base.update(extra_params)
        model = NMF(**base)
    else:
        from sklearn.decomposition import LatentDirichletAllocation
        base["learning_method"] = "batch"
        base["max_iter"] = int(max_iter) if max_iter else 20
        if extra_params:
            base.update(extra_params)
        model = LatentDirichletAllocation(**base)

    W = model.fit_transform(X)          # Dokument-Topic
    H = model.components_               # Topic-Wort

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    doc_topic = pd.DataFrame(
        W / row_sums, index=pd.Index(ids, name="id"),
        columns=[str(i) for i in range(k)])

    top = min(int(top_words), len(vocab))
    rows = [[vocab[j] for j in np.argsort(H[t])[::-1][:top]] for t in range(k)]
    topic_word = pd.DataFrame(
        rows, index=pd.Index(range(k), name="Topic"),
        columns=[str(i) for i in range(1, top + 1)])
    return doc_topic, topic_word, model


def fit_topics(matrix_df: pd.DataFrame, n_topics: int = 20,
               method: str = "nmf", top_words: int = 100,
               id_col: Optional[str] = "id",
               feature_start: Optional[int] = 10,
               random_state: int = 42, max_iter: Optional[int] = None,
               extra_params: Optional[dict] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fittet ein Topic-Modell auf einer **vorhandenen DTM/TF-IDF-Matrix**.

    ``method='nmf'`` erwartet eine TF-IDF-Matrix, ``method='lda'`` eine
    Zähl-DTM. Über ``extra_params`` lassen sich Hyperparameter des sklearn-
    Schätzers setzen (sie überschreiben die Defaults).
    """
    ids, vocab, X = _feature_block(matrix_df, id_col, feature_start)
    return _fit_on_matrix(X, vocab, ids, n_topics, method, top_words,
                          random_state, max_iter, extra_params)


def vectorize_texts(texts: List[str], method: str = "nmf",
                    max_features: Optional[int] = 2000, min_df: int = 2,
                    max_df: float = 0.95, lowercase: bool = False):
    """Vektorisiert Texte: TF-IDF für NMF, Zählungen (DTM) für LDA.

    Gibt ``(X, vocab, vectorizer)`` zurück (X ist eine dünne Matrix).
    """
    kwargs = dict(max_features=max_features, min_df=min_df, max_df=max_df,
                  lowercase=lowercase)
    if method.lower() == "nmf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(**kwargs)
    else:
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(**kwargs)
    X = vec.fit_transform(texts)
    return X, list(vec.get_feature_names_out()), vec


def fit_topics_from_corpus(corpus_df: pd.DataFrame, n_topics: int = 20,
                           method: str = "nmf", *, content_col: str = "content",
                           id_col: str = "id", chunk_words: int = 1000,
                           min_words: int = 50, top_words: int = 100,
                           max_features: Optional[int] = 2000, min_df: int = 2,
                           max_df: float = 0.95, lowercase: bool = False,
                           random_state: int = 42,
                           max_iter: Optional[int] = None,
                           extra_params: Optional[dict] = None,
                           aggregate: bool = True):
    """Komplettweg ab Korpus-Text **mit Chunking** – direkt im Topic-Modelling.

    Schritte: Texte in Wortfenster (``chunk_words``) segmentieren →
    vektorisieren (TF-IDF/Zählungen) → NMF/LDA fitten. Optional wird die
    Segment-Topic-Verteilung pro Ursprungstext gemittelt.

    Returns
    -------
    ``(doc_topic, topic_word, model, doc_topic_aggregated|None, info)`` –
    ``doc_topic`` auf Segmentebene; ``info`` enthält ``n_segments``,
    ``n_sources``, ``n_features``.
    """
    seg = segment_corpus(corpus_df, chunk_words=chunk_words, id_col=id_col,
                         content_col=content_col, min_words=min_words)
    if seg.empty:
        raise ValueError("Keine Segmente erzeugt – ist die Inhaltsspalte leer?")
    texts = seg[content_col].astype(str).tolist()
    ids = seg[id_col].astype(str).tolist()

    X, vocab, vec = vectorize_texts(texts, method=method,
                                    max_features=max_features, min_df=min_df,
                                    max_df=max_df, lowercase=lowercase)
    if X.shape[1] == 0:
        raise ValueError("Leeres Vokabular – min_df/max_df zu streng?")

    doc_topic, topic_word, model = _fit_on_matrix(
        X, vocab, ids, n_topics, method, top_words, random_state, max_iter,
        extra_params)

    agg = None
    if aggregate:
        source_ids = pd.Series(seg[SOURCE_COLUMN].values, index=seg[id_col].values)
        agg = aggregate_by_source(doc_topic, source_ids)

    info = {"n_segments": len(seg),
            "n_sources": int(seg[SOURCE_COLUMN].nunique()),
            "n_features": X.shape[1]}
    return doc_topic, topic_word, model, agg, info


def format_hyperparams(params: dict, title: str = "Topic-Modell – Hyperparameter") -> str:
    """Hyperparameter als lesbaren TXT-Block formatieren (für die Speicherung)."""
    from datetime import datetime
    lines = [f"# {title}",
             f"erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for key in sorted(params, key=str):
        lines.append(f"{key}: {params[key]}")
    return "\n".join(lines) + "\n"


def save_topic_outputs(doc_topic: pd.DataFrame, topic_word: pd.DataFrame,
                       out_dir: Path, basename: str = "sklearn",
                       params: Optional[dict] = None
                       ) -> Tuple[Path, Path, Optional[Path]]:
    """Schreibt beide Dateien im erwarteten Format (UTF-8, komma-separiert).

    Wird ``params`` übergeben, wird zusätzlich ``<name>_hyperparams.txt`` mit
    den verwendeten Einstellungen abgelegt. Gibt
    ``(doc_topic_path, topic_word_path, hyperparams_path|None)`` zurück.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_path = out_dir / f"document-topics-distribution_{basename}.csv"
    tw_path = out_dir / f"{basename}_topic_words.csv"
    doc_topic.to_csv(dt_path, encoding="utf-8")   # index=_id als Spalte 0
    topic_word.to_csv(tw_path, encoding="utf-8")  # index=Topic als Spalte 0

    hp_path: Optional[Path] = None
    if params:
        hp_path = out_dir / f"{basename}_hyperparams.txt"
        hp_path.write_text(format_hyperparams(params), encoding="utf-8")
    return dt_path, tw_path, hp_path
