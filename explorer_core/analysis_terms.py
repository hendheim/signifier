#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.analysis_terms
============================

UI-freie Analysefunktionen rund um Ausdrücke/Terme. Extrahiert aus den
``build_tab_*``-Funktionen des Korpus-Explorers ("Ausdrücke"-Block):
Frequenz, TF-IDF-Rang, Dokument-Frequenz, Konkordanz (KWIC),
Kollokationen (FREQ/PMI) und Wortverläufe.

Alle Funktionen nehmen DataFrames/Parameter entgegen und geben DataFrames
bzw. Matplotlib-Figuren zurück – keine Widgets, keine Dialoge. Die
Berechnungslogik ist 1:1 aus den Legacy-Tabs übernommen (Funktionsparität).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .schema import MetadataSchema

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text) -> List[str]:
    """Einfache Tokenisierung (case-sensitiv, wie im Kollokations-Tab)."""
    if not isinstance(text, str):
        return []
    return list(_TOKEN_RE.findall(text))


def _iter_ngrams(tokens: List[str], n: int):
    if n <= 1:
        yield from tokens
    else:
        buf = deque(maxlen=n)
        for t in tokens:
            buf.append(t)
            if len(buf) == n:
                yield " ".join(buf)


# ----------------------------------------------------------------------------
# Frequenz & TF-IDF-Rang
# ----------------------------------------------------------------------------

def term_frequencies(dtm: pd.DataFrame, term_cols: List[str],
                     search: Optional[List[str]] = None, top_n: int = 500) -> pd.DataFrame:
    """Gesamtfrequenz pro Term aus der DTM (Legacy-Tab 'Frequenz')."""
    if not term_cols:
        raise ValueError("Keine Term-Spalten in der DTM gefunden.")
    sums = dtm[term_cols].sum().sort_values(ascending=False)
    df = pd.DataFrame({
        "rank": np.arange(1, len(sums) + 1),
        "term": sums.index,
        "freq": sums.values.astype(int),
    })
    if search:
        wanted = {t.strip() for t in search if t.strip()}
        return df[df["term"].astype(str).isin(wanted)]
    return df.head(top_n)


def filter_tfidf_rank(tfidf_avg: pd.DataFrame,
                      search: Optional[List[str]] = None, top_n: int = 500) -> pd.DataFrame:
    """Filtert die TF-IDF-Durchschnittstabelle (Legacy-Tab 'TF-IDF-Rang')."""
    if search:
        wanted = {t.strip() for t in search if t.strip()}
        return tfidf_avg[tfidf_avg["term"].astype(str).isin(wanted)]
    return tfidf_avg.head(top_n)


def term_overview(dtm: pd.DataFrame, term_cols: List[str],
                  tfidf_avg: pd.DataFrame,
                  search: Optional[List[str]] = None, top_n: int = 500) -> pd.DataFrame:
    """Gesamtsuche: kombiniert Gesamtfrequenz (DTM) und mittleren TF-IDF
    (Durchschnitt über alle Dokumente) je Ausdruck in EINER Tabelle.

    Spalten: rank, term, freq (Summe aller Vorkommen im Korpus),
    tfidf_mean (mittlerer TF-IDF über alle Dokumente). Ausdrücke außerhalb
    der TF-IDF-Top-N erhalten tfidf_mean = NaN. Standardsortierung nach freq;
    in der Anzeige lässt sich die Tabelle interaktiv nach tfidf_mean sortieren.
    """
    if not term_cols:
        raise ValueError("Keine Term-Spalten in der DTM gefunden.")
    freq = dtm[term_cols].sum()
    freq_df = pd.DataFrame({
        "term": freq.index.astype(str),
        "freq": freq.values.astype(int),
    })
    tf = tfidf_avg[["term", "tfidf_avg"]].copy()
    tf["term"] = tf["term"].astype(str)
    tf = tf.rename(columns={"tfidf_avg": "tfidf_mean"})

    df = freq_df.merge(tf, on="term", how="outer")
    df["freq"] = df["freq"].fillna(0).astype(int)
    df = df.sort_values("freq", ascending=False).reset_index(drop=True)

    if search:
        wanted = {t.strip() for t in search if t.strip()}
        out = df[df["term"].isin(wanted)].copy()
    else:
        out = df.head(top_n).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Dokument-Frequenz
# ----------------------------------------------------------------------------

def document_frequencies(corpus: pd.DataFrame, metadata: pd.DataFrame,
                         schema: MetadataSchema, terms: List[str],
                         display_cols: List[str],
                         use_regex: bool = False,
                         case_sensitive: bool = False,
                         normalize: bool = True,
                         scale: int = 10000) -> pd.DataFrame:
    """Trefferanzahl pro Dokument für eine Liste von Ausdrücken
    (Legacy-Tab 'Dokument-Frequenz').

    Bei ``normalize=True`` (Standard) werden zusätzlich die Textlänge
    (Tokenzahl je Dokument) und die normalisierte Trefferdichte
    ``count / Textlänge × scale`` ausgegeben – so sind Texte
    unterschiedlicher Länge vergleichbar (Default: Treffer pro 10 000 Tokens).
    """
    terms = [t.strip() for t in terms if t.strip()]
    if not terms:
        raise ValueError("Bitte mindestens einen Ausdruck angeben.")

    flags = 0 if case_sensitive else re.IGNORECASE
    counts = []
    texts = corpus["text"].fillna("").astype(str)
    for term in terms:
        pattern = re.compile(term if use_regex else re.escape(term), flags)
        counts.append(texts.apply(lambda x: len(pattern.findall(x))))

    result = corpus[["doc_id"]].copy()
    result["count"] = sum(counts)
    if normalize:
        result["text_laenge"] = texts.apply(lambda x: len(tokenize(x))).astype(int)

    # Metadaten anhand der ID-Spalte des Schemas joinen
    meta = metadata.copy()
    join_cols = [c for c in display_cols if c in meta.columns]
    if "doc_id" in meta.columns and join_cols:
        result = result.merge(meta[["doc_id"] + join_cols].drop_duplicates("doc_id"),
                              on="doc_id", how="left")

    result = result[result["count"] > 0].sort_values("count", ascending=False)

    norm_col = None
    if normalize:
        norm_col = (f"pro_{scale // 1000}k_tokens" if scale % 1000 == 0
                    else f"pro_{scale}_tokens")
        laenge = result["text_laenge"].replace(0, np.nan)
        result[norm_col] = (result["count"] / laenge * scale).round(2)

    ordered = ["doc_id"] + join_cols + ["count"]
    if normalize:
        ordered += ["text_laenge", norm_col]
    return result[[c for c in ordered if c in result.columns]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Konkordanz (KWIC)
# ----------------------------------------------------------------------------

def concordance(corpus: pd.DataFrame, metadata: pd.DataFrame, term: str,
                display_cols: List[str], context: int = 50,
                max_hits: int = 5000) -> pd.DataFrame:
    """KWIC-Konkordanz mit Kontextfenster (Legacy-Tab 'Konkordanz')."""
    term = term.strip()
    if not term:
        raise ValueError("Bitte einen Suchbegriff angeben.")

    # Case-sensitiv: "Hund" trifft nicht "hund" (passend zum gecaseten Korpus).
    pattern = re.compile(f"(.{{0,{context}}})({re.escape(term)})(.{{0,{context}}})")

    meta_lookup: Dict[str, dict] = {}
    if "doc_id" in metadata.columns:
        cols = [c for c in display_cols if c in metadata.columns]
        for _, m in metadata.iterrows():
            meta_lookup[str(m["doc_id"])] = {c: m.get(c, "") for c in cols}

    rows = []
    for _, r in corpus.iterrows():
        text = str(r.get("text", ""))
        doc_id = str(r.get("doc_id", ""))
        meta_vals = meta_lookup.get(doc_id, {})
        for m in pattern.finditer(text):
            row = {"doc_id": doc_id,
                   **meta_vals,
                   "left": m.group(1).replace("\n", " "),
                   "match": m.group(2),
                   "right": m.group(3).replace("\n", " ")}
            rows.append(row)
            if len(rows) >= max_hits:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Kollokationen
# ----------------------------------------------------------------------------

def collocations(corpus: pd.DataFrame, targets: List[str], window: int = 5,
                 top_n: int = 100, min_freq: int = 3, ngram: int = 1,
                 metric: str = "FREQ") -> pd.DataFrame:
    """Kollokationsanalyse mit FREQ- oder PMI-Score (Legacy-Tab 'Kollokation')."""
    targets = [t.strip() for t in targets if t.strip()]
    if not targets:
        raise ValueError("Bitte Zielausdrücke angeben.")

    total_tokens = 0
    freq_w: Counter = Counter()
    freq_tw: Dict[str, Counter] = defaultdict(Counter)
    freq_t: Counter = Counter()

    for _, r in corpus.iterrows():
        tokens = tokenize(r.get("text", ""))
        if not tokens:
            continue
        ngrams = list(_iter_ngrams(tokens, ngram))
        total_tokens += len(ngrams)
        for w in ngrams:
            freq_w[w] += 1
        for i, w in enumerate(tokens):
            if w in targets:
                freq_t[w] += 1
                lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
                if ngram == 1:
                    ctx = tokens[lo:i] + tokens[i + 1:hi]
                else:
                    ctx = (list(_iter_ngrams(tokens[lo:i], ngram))
                           + list(_iter_ngrams(tokens[i + 1:hi], ngram)))
                for cw in ctx:
                    freq_tw[w][cw] += 1

    rows = []
    eps = 1e-12
    for t in targets:
        ct = max(1, freq_t[t])
        for cw, ctw in freq_tw[t].items():
            if ctw < min_freq:
                continue
            if metric == "FREQ":
                score = float(ctw)
            else:  # PMI
                pw = freq_w[cw] / max(1, total_tokens)
                pt = ct / max(1, total_tokens)
                ptw = ctw / max(1, total_tokens)
                score = math.log2(max(eps, ptw) / max(eps, pt * pw))
            rows.append((t, cw, int(ctw), float(score)))

    if not rows:
        return pd.DataFrame(columns=["target", "collocate", "freq", "score"])

    res = pd.DataFrame(rows, columns=["target", "collocate", "freq", "score"])
    sort_col = "freq" if metric == "FREQ" else "score"
    return (res.sort_values(["target", sort_col], ascending=[True, False])
               .groupby("target", group_keys=False).head(top_n)
               .reset_index(drop=True))


def collocation_documents(corpus: pd.DataFrame, metadata: pd.DataFrame,
                          target: str, collocate: str, display_cols: List[str],
                          window: int = 5, ngram: int = 1) -> pd.DataFrame:
    """Dokumente, in denen ein Kollokationspaar vorkommt (Klick-Detail im Legacy-Tab)."""
    target, collocate = target.strip(), collocate.strip()

    meta_lookup: Dict[str, dict] = {}
    if "doc_id" in metadata.columns:
        cols = [c for c in display_cols if c in metadata.columns]
        for _, m in metadata.iterrows():
            meta_lookup[str(m["doc_id"])] = {c: m.get(c, "") for c in cols}

    rows = []
    for _, d in corpus.iterrows():
        tokens = tokenize(d.get("text", ""))
        positions = [i for i, w in enumerate(tokens) if w == target]
        if not positions:
            continue
        count = 0
        for i in positions:
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            if ngram == 1:
                ctx = tokens[lo:i] + tokens[i + 1:hi]
            else:
                ctx = (list(_iter_ngrams(tokens[lo:i], ngram))
                       + list(_iter_ngrams(tokens[i + 1:hi], ngram)))
            count += sum(1 for c in ctx if c == collocate)
        if count > 0:
            doc_id = str(d.get("doc_id", ""))
            rows.append({"doc_id": doc_id, **meta_lookup.get(doc_id, {}), "freq": count})

    if not rows:
        return pd.DataFrame(columns=["doc_id", "freq"])
    return pd.DataFrame(rows).sort_values("freq", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Wortverläufe
# ----------------------------------------------------------------------------

def word_trends(dtm: pd.DataFrame, term_cols: List[str], schema: MetadataSchema,
                terms: List[str], year_range: Optional[Tuple[int, int]] = None
                ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Aggregiert Frequenzen pro Jahr (absolut + relativ pro Mio. Tokens).

    Returns
    -------
    (df_abs, df_rel, missing): Jahres-DataFrames + nicht gefundene Begriffe.
    """
    terms_input = [t.strip() for t in terms if t.strip()]
    valid = [t for t in terms_input if t in term_cols]
    missing = [t for t in terms_input if t not in term_cols]
    if not valid:
        raise ValueError("Keiner der Begriffe ist als Spalte in der DTM vorhanden.")

    years = schema.get_year_series(dtm)
    if years is None:
        raise ValueError("Keine Jahr-Spalte in der DTM gefunden.")

    work = dtm.copy()
    work["_year"] = years
    work = work.dropna(subset=["_year"])
    if year_range:
        lo, hi = year_range
        work = work[(work["_year"] >= lo) & (work["_year"] <= hi)]

    df_abs = pd.DataFrame({t: work.groupby("_year")[t].sum() for t in valid})
    df_abs.index.name = "year"

    total_per_year = work.groupby("_year")[term_cols].sum().sum(axis=1)
    df_rel = pd.DataFrame(index=df_abs.index)
    for t in valid:
        df_rel[t] = (df_abs[t] / total_per_year) * 1_000_000  # pro Mio. Tokens

    return df_abs, df_rel, missing


def plot_trends(df: pd.DataFrame, title: str, ylabel: str,
                smooth_window: Optional[int] = None,
                poly_degree: Optional[int] = None) -> plt.Figure:
    """Linien-Plot für Verläufe; optional geglättet oder als Polynom-Fit.

    Wird sowohl für Wort- als auch für Topic-Verläufe verwendet.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in df.columns:
        if poly_degree:
            x = df.index.astype(float).values
            y = df[col].values.astype(float)
            mask = ~np.isnan(y)
            xc, yc = x[mask], y[mask]
            if len(xc) > poly_degree:
                coeffs = np.polyfit(xc, yc, poly_degree)
                xx = np.linspace(xc.min(), xc.max(), 200)
                ax.plot(xx, np.polyval(coeffs, xx), label=col)
        elif smooth_window and smooth_window > 1:
            smoothed = df[col].rolling(window=smooth_window, center=True,
                                       min_periods=1).mean()
            ax.plot(df.index, smoothed, label=col)
        else:
            ax.plot(df.index, df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Jahr")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
    fig.tight_layout()
    return fig
