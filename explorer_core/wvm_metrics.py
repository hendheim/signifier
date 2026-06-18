#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.wvm_metrics
=========================

Qualitätsmetriken für das Wort-Vektor-Modell (Word2Vec / gensim
``KeyedVectors``) – research-orientiert und bewusst mit den Vorbehalten für
ein historisches Korpus:

- ``model_summary``        : Vokabulargröße, Vektordimension.
- ``vocabulary_coverage``  : Abdeckung/OOV-Rate für Interessenswörter.
- ``nearest_neighbors``    : Cosinus-Nachbarn (qualitativ aussagekräftigstes Maß).
- ``evaluate_pairs``       : Spearman-Korrelation Modell-Cosinus vs. menschliche
                             Ähnlichkeitsurteile (Vorbehalt: moderne Test-Sets
                             passen schlecht zu historischem Deutsch).
- ``neighbor_stability``   : Jaccard-Überlappung der Top-N-Nachbarn zweier
                             Modelle (Seed-/Trainings-Stabilität).
- ``procrustes_align`` /
  ``semantic_drift``       : diachrone Bedeutungsverschiebung (Hamilton et al.
                             2016): zwei Periodenmodelle ausrichten, Cosinus-
                             Drift je Wort.

Die Mathematik nutzt nur numpy/scipy; als ``kv`` genügt ein Objekt mit
``index_to_key``, ``key_to_index``, ``vectors`` (und ``vector_size``) – also
gensims ``KeyedVectors`` ebenso wie ein schlankes Test-Objekt.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def _vector(kv, word) -> np.ndarray:
    idx = kv.key_to_index[word]
    return np.asarray(kv.vectors[idx], dtype=float)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + _EPS))


def model_summary(kv) -> Dict[str, int]:
    return {"vocab_size": len(kv.index_to_key),
            "vector_size": int(getattr(kv, "vector_size", kv.vectors.shape[1]))}


def vocabulary_coverage(kv, words: Sequence[str]) -> Dict:
    words = [w for w in (w.strip() for w in words) if w]
    in_vocab = [w for w in words if w in kv.key_to_index]
    oov = [w for w in words if w not in kv.key_to_index]
    rate = len(in_vocab) / len(words) if words else 0.0
    return {"n": len(words), "in_vocab": in_vocab, "oov": oov,
            "coverage": rate}


def nearest_neighbors(kv, word: str, topn: int = 10) -> List[Tuple[str, float]]:
    if word not in kv.key_to_index:
        raise KeyError(f"'{word}' ist nicht im Vokabular.")
    V = np.asarray(kv.vectors, dtype=float)
    norms = np.linalg.norm(V, axis=1) + _EPS
    i = kv.key_to_index[word]
    sims = (V @ V[i]) / (norms * norms[i])
    order = np.argsort(-sims)
    out = [(kv.index_to_key[j], float(sims[j])) for j in order if j != i]
    return out[:topn]


def evaluate_pairs(kv, pairs: Sequence[Tuple[str, str, float]]) -> Dict:
    """Spearman-Korrelation zwischen Modell-Cosinus und menschlichem Urteil."""
    from scipy.stats import spearmanr
    model_s, human_s, oov = [], [], 0
    for w1, w2, human in pairs:
        if w1 in kv.key_to_index and w2 in kv.key_to_index:
            model_s.append(_cosine(_vector(kv, w1), _vector(kv, w2)))
            human_s.append(float(human))
        else:
            oov += 1
    if len(model_s) < 2:
        return {"spearman": None, "pvalue": None, "n_used": len(model_s),
                "n_oov": oov}
    rho, p = spearmanr(model_s, human_s)
    return {"spearman": float(rho), "pvalue": float(p),
            "n_used": len(model_s), "n_oov": oov}


def neighbor_stability(kv_a, kv_b, words: Sequence[str], topn: int = 10) -> Dict:
    """Mittlere Jaccard-Überlappung der Top-N-Nachbarn zweier Modelle."""
    jacc = []
    for w in words:
        if w in kv_a.key_to_index and w in kv_b.key_to_index:
            na = {x for x, _ in nearest_neighbors(kv_a, w, topn)}
            nb = {x for x, _ in nearest_neighbors(kv_b, w, topn)}
            if na or nb:
                jacc.append(len(na & nb) / len(na | nb))
    return {"mean_jaccard": float(np.mean(jacc)) if jacc else None,
            "n_words": len(jacc)}


def procrustes_align(kv_base, kv_other) -> Tuple[np.ndarray, List[str]]:
    """Orthogonale Procrustes-Ausrichtung von ``kv_other`` auf ``kv_base``.

    Liefert (Rotationsmatrix R, gemeinsames Vokabular) mit ``other @ R ≈ base``.
    """
    from scipy.linalg import orthogonal_procrustes
    shared = [w for w in kv_base.index_to_key if w in kv_other.key_to_index]
    if len(shared) < 2:
        raise ValueError("Zu wenig gemeinsames Vokabular für die Ausrichtung.")
    A = np.vstack([_vector(kv_other, w) for w in shared])  # zu rotieren
    B = np.vstack([_vector(kv_base, w) for w in shared])   # Ziel
    R, _ = orthogonal_procrustes(A, B)
    return R, shared


def semantic_drift(kv_base, kv_other, words: Sequence[str],
                   R: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Diachrone Drift je Wort = 1 - Cosinus(Basis, ausgerichtetes Anderes)."""
    if R is None:
        R, _ = procrustes_align(kv_base, kv_other)
    drift = {}
    for w in words:
        if w in kv_base.key_to_index and w in kv_other.key_to_index:
            b = _vector(kv_base, w)
            o = _vector(kv_other, w) @ R
            drift[w] = 1.0 - _cosine(b, o)
    return drift
