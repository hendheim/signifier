#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.token_index
=========================

Einmalig aufgebauter, numerischer Token-Index über das Korpus.

Motivation: Die Kollokations- und Dokument-Frequenz-Analysen tokenisierten
bislang bei *jedem* Aufruf das komplette Korpus neu (bei ~45 Mio. Tokens
zweistellige Sekunden pro Klick). Dieser Index tokenisiert genau einmal
(mit derselben Regex wie ``analysis_terms.tokenize``) und hält das Ergebnis
kompakt als ``int32``-ID-Array plus Vokabular:

- ``vocab``   : Token → int-ID (Reihenfolge = erstes Vorkommen im Korpus)
- ``id2tok``  : ID → Token (numpy-Objekt-Array für Fancy-Indexing)
- ``ids``     : alle Tokens des Korpus als flaches ``int32``-Array
- ``offsets`` : Dokumentgrenzen, ``ids[offsets[k]:offsets[k+1]]`` = Dokument k

Speicherbedarf bei 45 Mio. Tokens: ~180 MB für ``ids`` plus Vokabular –
unkritisch gegenüber dem wiederholten Tokenisieren.

Die Kollokationsfunktionen hier sind für ``ngram=1`` (Standardfall)
vollständig vektorisiert; für ``ngram=2/3`` delegieren sie an die
unveränderte Zähllogik in ``analysis_terms`` (gespeist aus dem Index statt
aus erneuter Tokenisierung), damit die Semantik exakt gleich bleibt.

Wie der Rest von ``explorer_core`` frei von Streamlit/Tkinter.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from .analysis_terms import (
    _TOKEN_RE,
    _collocations_from_tokens,
    _collocation_documents_from_tokens,
)

# Chunk-Größe für die Fenster-Gather-Operationen (Positionen pro Block),
# begrenzt den Spitzen-Speicherbedarf bei sehr häufigen Zielwörtern.
_CHUNK = 200_000


class TokenIndex:
    """Kompakter Token-Index (IDs + Dokumentgrenzen) über das ganze Korpus."""

    __slots__ = ("vocab", "id2tok", "ids", "offsets", "doc_ids")

    def __init__(self, vocab: Dict[str, int], ids: np.ndarray,
                 offsets: np.ndarray, doc_ids: List[str]):
        self.vocab = vocab
        self.id2tok = np.array(list(vocab.keys()), dtype=object)
        self.ids = ids
        self.offsets = offsets
        self.doc_ids = doc_ids

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, doc_ids: List[str], texts: List[str]) -> "TokenIndex":
        """Tokenisiert alle Texte einmal (Regex wie ``analysis_terms.tokenize``)."""
        vocab: Dict[str, int] = {}
        parts: List[np.ndarray] = []
        offsets = np.empty(len(texts) + 1, dtype=np.int64)
        offsets[0] = 0
        for k, text in enumerate(texts):
            toks = _TOKEN_RE.findall(text) if isinstance(text, str) else []
            arr = np.fromiter(
                (vocab.setdefault(t, len(vocab)) for t in toks),
                dtype=np.int32, count=len(toks))
            parts.append(arr)
            offsets[k + 1] = offsets[k] + arr.size
        ids = (np.concatenate(parts) if parts
               else np.empty(0, dtype=np.int32))
        return cls(vocab, ids, offsets, [str(d) for d in doc_ids])

    # ------------------------------------------------------------------
    # Abgeleitete Größen
    # ------------------------------------------------------------------

    def token_counts(self) -> pd.Series:
        """Tokenzahl je Dokument (Index = doc_id), z. B. für ``text_laenge``."""
        return pd.Series(np.diff(self.offsets), index=self.doc_ids)

    def iter_doc_tokens(self) -> Iterator[List[str]]:
        """Token-Listen je Dokument (rekonstruiert, ohne Neu-Tokenisierung)."""
        for k in range(len(self.doc_ids)):
            seg = self.ids[self.offsets[k]:self.offsets[k + 1]]
            yield self.id2tok[seg].tolist()

    # ------------------------------------------------------------------
    # Interna
    # ------------------------------------------------------------------

    def _window_context(self, pos: np.ndarray, window: int):
        """Gather der Fenster-Nachbarn (ohne Zentrum), dokumentgrenzen-treu.

        Liefert (gathered_ids, valid_mask) mit Form (len(pos), 2*window).
        """
        rel = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])
        d = np.searchsorted(self.offsets, pos, side="right") - 1
        lo = self.offsets[d]
        hi = self.offsets[d + 1]
        idx = pos[:, None] + rel[None, :]
        valid = (idx >= lo[:, None]) & (idx < hi[:, None])
        gathered = self.ids[np.clip(idx, 0, max(0, self.ids.size - 1))]
        return gathered, valid


# ----------------------------------------------------------------------------
# Kollokationen auf dem Index
# ----------------------------------------------------------------------------

def collocations(index: TokenIndex, targets: List[str], window: int = 5,
                 top_n: int = 100, min_freq: int = 3, ngram: int = 1,
                 metric: str = "FREQ") -> pd.DataFrame:
    """Wie ``analysis_terms.collocations``, aber ohne Neu-Tokenisierung.

    ``ngram=1`` läuft vektorisiert auf dem ID-Array; ``ngram>1`` nutzt die
    Originallogik aus ``analysis_terms`` (identische Ergebnisse), gespeist
    aus dem Index. Bei Score-Gleichstand ist die Reihenfolge deterministisch
    (Kollokat alphabetisch) – die Werte selbst sind identisch zur Altfassung.
    """
    targets = [t.strip() for t in targets if t.strip()]
    if not targets:
        raise ValueError("Bitte Zielausdrücke angeben.")

    if ngram != 1:
        return _collocations_from_tokens(
            index.iter_doc_tokens(), targets, window=window, top_n=top_n,
            min_freq=min_freq, ngram=ngram, metric=metric)

    ids = index.ids
    total_tokens = int(ids.size)
    n_vocab = len(index.vocab)
    freq_w = np.bincount(ids, minlength=n_vocab) if metric != "FREQ" else None

    rows = []
    eps = 1e-12
    for t in targets:
        tid = index.vocab.get(t)
        if tid is None:
            continue
        pos = np.flatnonzero(ids == np.int32(tid))
        ct = int(pos.size)
        if ct == 0:
            continue

        counts = np.zeros(n_vocab, dtype=np.int64)
        for c0 in range(0, ct, _CHUNK):
            gathered, valid = index._window_context(pos[c0:c0 + _CHUNK], window)
            counts += np.bincount(gathered[valid], minlength=n_vocab)

        cand = np.flatnonzero(counts >= min_freq)
        if cand.size == 0:
            continue
        ctw = counts[cand]
        if metric == "FREQ":
            score = ctw.astype(float)
        else:  # PMI, Formel identisch zu analysis_terms.collocations
            total = max(1, total_tokens)
            pw = freq_w[cand] / total
            pt = max(1, ct) / total
            ptw = ctw / total
            score = np.log2(np.maximum(eps, ptw) / np.maximum(eps, pt * pw))

        toks = index.id2tok[cand]
        rows.extend((t, cw, int(c), float(s))
                    for cw, c, s in zip(toks, ctw, score))

    if not rows:
        return pd.DataFrame(columns=["target", "collocate", "freq", "score"])

    res = pd.DataFrame(rows, columns=["target", "collocate", "freq", "score"])
    sort_col = "freq" if metric == "FREQ" else "score"
    res = res.sort_values(["target", sort_col, "collocate"],
                          ascending=[True, False, True], kind="mergesort")
    return (res.groupby("target", group_keys=False).head(top_n)
               .reset_index(drop=True))


def collocation_documents(index: TokenIndex, target: str, collocate: str,
                          window: int = 5, ngram: int = 1) -> pd.DataFrame:
    """Dokumente mit einem Kollokationspaar (Spalten: doc_id, freq).

    Metadaten-Spalten hängt der Aufrufer bei Bedarf über
    ``analysis_terms.join_display_metadata`` an.
    """
    target, collocate = target.strip(), collocate.strip()

    if ngram != 1:
        return _collocation_documents_from_tokens(
            index.doc_ids, index.iter_doc_tokens(), target, collocate,
            window=window, ngram=ngram)

    tid = index.vocab.get(target)
    cid = index.vocab.get(collocate)
    if tid is None or cid is None:
        return pd.DataFrame(columns=["doc_id", "freq"])

    ids = index.ids
    pos = np.flatnonzero(ids == np.int32(tid))
    if pos.size == 0:
        return pd.DataFrame(columns=["doc_id", "freq"])

    doc_counts = np.zeros(len(index.doc_ids), dtype=np.int64)
    for c0 in range(0, pos.size, _CHUNK):
        p = pos[c0:c0 + _CHUNK]
        gathered, valid = index._window_context(p, window)
        per_pos = ((gathered == np.int32(cid)) & valid).sum(axis=1)
        d = np.searchsorted(index.offsets, p, side="right") - 1
        np.add.at(doc_counts, d, per_pos)

    hit = np.flatnonzero(doc_counts > 0)
    if hit.size == 0:
        return pd.DataFrame(columns=["doc_id", "freq"])
    df = pd.DataFrame({"doc_id": [index.doc_ids[i] for i in hit],
                       "freq": doc_counts[hit]})
    return (df.sort_values("freq", ascending=False, kind="mergesort")
              .reset_index(drop=True))


__all__ = ["TokenIndex", "collocations", "collocation_documents"]
