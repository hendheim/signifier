#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.topic_metrics
===========================

Qualitätsmaße für Topic-Modelle (research-orientiert):

- ``topic_diversity``   : Anteil eindeutiger Wörter über alle Topic-Top-Listen
                          (Dieng et al. 2020) – hoch = wenig redundante Topics.
- ``coherence_npmi``    : eigenständige NPMI-Kohärenz über Dokument-Ko-Okkurrenz
                          im Referenz-Korpus (ohne externe Abhängigkeit, testbar).
- ``coherence_gensim``  : C_v bzw. c_npmi über ``gensim.CoherenceModel``
                          (Forschungsstandard, Röder et al. 2015; benötigt gensim).
- Hilfen: ``topic_word_lists_from_df`` (aus unserer Topic-Wort-Tabelle),
  ``docs_from_matrix`` (Referenz-Dokumente aus einer DTM rekonstruieren).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import pandas as pd


def topic_word_lists_from_df(topic_word: pd.DataFrame,
                             top_k: Optional[int] = None) -> List[List[str]]:
    """Wandelt unsere Topic-Wort-Tabelle (Zeile=Topic, Zellen=Wörter) in
    Listen von Wörtern je Topic um."""
    lists = []
    for _, row in topic_word.iterrows():
        words = [str(w) for w in row.dropna().tolist() if str(w) != "nan"]
        lists.append(words[:top_k] if top_k else words)
    return lists


def topic_diversity(topic_lists: Sequence[Sequence[str]], top_k: int = 25) -> float:
    """Anteil eindeutiger Wörter über die Top-k-Wörter aller Topics."""
    tops = [w for t in topic_lists for w in list(t)[:top_k]]
    return (len(set(tops)) / len(tops)) if tops else 0.0


def coherence_npmi(topic_lists: Sequence[Sequence[str]],
                   ref_texts: Sequence[Sequence[str]], top_k: int = 10,
                   eps: float = 1e-12) -> Dict:
    """NPMI-Kohärenz über boolesche Dokument-Ko-Okkurrenz im Referenz-Korpus.

    Pro Topic der Mittelwert der paarweisen NPMI-Werte der Top-k-Wörter; das
    Modellmaß ist der Mittelwert über die Topics. Eigenständig (kein gensim) –
    als robuste Gegenprobe und für Umgebungen ohne gensim.
    """
    N = len(ref_texts)
    if N == 0:
        return {"per_topic": [], "mean": float("nan")}
    needed = {w for t in topic_lists for w in list(t)[:top_k]}
    doc_sets: Dict[str, set] = {w: set() for w in needed}
    for i, doc in enumerate(ref_texts):
        for w in (set(doc) & needed):
            doc_sets[w].add(i)

    per_topic = []
    for t in topic_lists:
        words = [w for w in list(t)[:top_k] if doc_sets.get(w)]
        scores = []
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                di, dj = doc_sets[words[a]], doc_sets[words[b]]
                p_ij = len(di & dj) / N
                if p_ij <= 0:
                    scores.append(-1.0)
                    continue
                p_i, p_j = len(di) / N, len(dj) / N
                npmi = math.log(p_ij / (p_i * p_j) + eps) / (-math.log(p_ij + eps))
                scores.append(npmi)
        per_topic.append(sum(scores) / len(scores) if scores else float("nan"))

    valid = [x for x in per_topic if not math.isnan(x)]
    mean = sum(valid) / len(valid) if valid else float("nan")
    return {"per_topic": per_topic, "mean": mean, "measure": "npmi"}


def coherence_gensim(topic_lists: Sequence[Sequence[str]],
                     ref_texts: Sequence[Sequence[str]], measure: str = "c_v",
                     top_k: Optional[int] = None) -> Dict:
    """C_v bzw. c_npmi über gensim (Röder et al. 2015). Benötigt gensim."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    topics = [list(t)[:top_k] if top_k else list(t) for t in topic_lists]
    topics = [t for t in topics if len(t) >= 2]
    dictionary = Dictionary(list(ref_texts))
    cm = CoherenceModel(topics=topics, texts=list(ref_texts),
                        dictionary=dictionary, coherence=measure)
    return {"per_topic": cm.get_coherence_per_topic(),
            "mean": cm.get_coherence(), "measure": measure}


def docs_from_matrix(df: pd.DataFrame, feature_start: int = 10
                     ) -> List[List[str]]:
    """Rekonstruiert Referenz-Dokumente (Term-Präsenz je Dokument) aus einer
    DTM/TF-IDF-CSV – für die Kohärenz, wenn kein Rohtext vorliegt."""
    feats = df.iloc[:, feature_start:].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    vocab = [str(c) for c in feats.columns]
    docs = []
    for _, row in feats.iterrows():
        vals = row.to_numpy()
        docs.append([vocab[j] for j in range(len(vocab)) if vals[j] > 0])
    return docs
