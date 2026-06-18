#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.corpus_segment
============================

Zerlegt ein Korpus (eine Zeile = ein langer Text/Roman) in **Segmente**, damit
Topic-Modelle sinnvoll arbeiten: viele kurze „Dokumente" statt weniger sehr
langer. Das behebt sowohl das LDA-Rauschen (zu wenige Dokumente) als auch die
NMF-Rang-Grenze (``min(#Dokumente, #Features)``) und bildet die thematische
Binnen­differenzierung großer Romane ab.

Workflow: Korpus (idealerweise bereits lemmatisiert) → ``segment_corpus`` →
als CSV speichern → bestehende ``s03_dtm_tfidf`` + Topic-Modell darauf laufen
lassen. Mit ``aggregate_by_source`` lassen sich Segment-Topic-Verteilungen
anschließend pro Ursprungstext mitteln.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

ID_COLUMN = "_id"
CONTENT_COLUMN = "content"
SOURCE_COLUMN = "source_id"


def _word_chunks(text: str, chunk_words: int) -> List[str]:
    """Teilt ``text`` in Stücke von höchstens ``chunk_words`` Wörtern."""
    words = (text or "").split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_words])
            for i in range(0, len(words), chunk_words)]


def segment_corpus(df: pd.DataFrame, chunk_words: int = 1000,
                   id_col: str = ID_COLUMN, content_col: str = CONTENT_COLUMN,
                   min_words: int = 50) -> pd.DataFrame:
    """Zerlegt jeden Text in Wortfenster und erzeugt ein Segment-Korpus.

    Jedes Segment wird eine eigene Zeile mit
    - ``_id`` = ``<ursprungs-id>_<laufende_nummer>`` (4-stellig),
    - ``source_id`` = ursprüngliche Dokument-ID (für die spätere Aggregation),
    - allen übrigen Metadatenspalten (geerbt vom Ursprungstext),
    - ``content`` = Segmenttext.

    Segmente mit weniger als ``min_words`` Wörtern (typisch das kurze Ende
    eines Textes) werden verworfen, sofern der Text mehr als ein Segment hat.

    Returns
    -------
    DataFrame im selben Spaltenschema (``_id`` + Metadaten + ``content``), nur
    mit zusätzlicher Spalte ``source_id`` und vielen mehr Zeilen.
    """
    if id_col not in df.columns:
        raise ValueError(f"ID-Spalte '{id_col}' fehlt. Vorhanden: "
                         f"{list(df.columns)}")
    if content_col not in df.columns:
        raise ValueError(f"Inhaltsspalte '{content_col}' fehlt.")

    meta_cols = [c for c in df.columns if c not in (id_col, content_col)]
    rows = []
    for _, src in df.iterrows():
        sid = str(src[id_col])
        chunks = _word_chunks(str(src[content_col]), chunk_words)
        if not chunks:
            continue
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1 and len(chunk.split()) < min_words:
                continue  # zu kurzes Endstück verwerfen
            row = {id_col: f"{sid}_{i:04d}", SOURCE_COLUMN: sid}
            for c in meta_cols:
                row[c] = src[c]
            row[content_col] = chunk
            rows.append(row)

    cols = [id_col, SOURCE_COLUMN] + meta_cols + [content_col]
    return pd.DataFrame(rows, columns=cols)


def aggregate_by_source(doc_topic: pd.DataFrame,
                        source_ids: Optional[pd.Series] = None,
                        sep: str = "_") -> pd.DataFrame:
    """Mittelt eine Segment-Topic-Verteilung pro Ursprungstext.

    ``doc_topic`` ist die Segment-Verteilung (Index = Segment-IDs). Die
    Zuordnung Segment→Ursprung kommt entweder aus ``source_ids`` (Series,
    gleich indiziert wie ``doc_topic``) oder – als Fallback – durch Abschneiden
    des letzten ``sep``-Abschnitts der Segment-ID.
    """
    if source_ids is not None:
        groups = source_ids.reindex(doc_topic.index)
    else:
        groups = pd.Series(
            [str(ix).rsplit(sep, 1)[0] for ix in doc_topic.index],
            index=doc_topic.index)
    out = doc_topic.groupby(groups).mean()
    out.index.name = ID_COLUMN
    return out
