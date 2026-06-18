#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.analysis_topics
=============================

UI-freie Analysen rund um Topic-Modelle. Extrahiert aus dem Topics-Tab des
Korpus-Explorers und sämtlichen Visualisierungs-Tabs des Tag-Topic-Explorers:
Topicverläufe, Tag-Topic-Bubble-Chart, gestapelte Jahres-Balken,
Tokens-vs.-Topics-Vergleich und TT-Texts-Ranglisten.

Wichtigste Flexibilisierung gegenüber den Legacy-GUIs:
- Jahr-Mapping läuft über das Metadatenschema statt über fest kodierte
  Spalten (früher ``Jahr_final``/``year_final`` direkt im Code).
- Der harte Jahresfilter (>= 1840 bzw. >= 1800) ist durch ``min_year`` aus
  dem Schema ersetzt (Standard: kein Filter).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .schema import MetadataSchema, find_column


def natural_key(s: str):
    """Sortierschlüssel für natürliche Sortierung: 'Topic_2' < 'Topic_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def normalize_document_id(doc_id: str) -> str:
    """Normalisiert eine Dokument-ID für das Matching (Pfad/Endung/Präfix weg)."""
    doc_id = str(doc_id)
    doc_id = Path(doc_id).stem
    doc_id = re.sub(r"^(doc_?|text_?|id_?)", "", doc_id, flags=re.IGNORECASE)
    return doc_id.strip()


# ----------------------------------------------------------------------------
# Topicverläufe (Document-Topic-Verteilung über die Zeit)
# ----------------------------------------------------------------------------

def topics_with_years(topics_dist: pd.DataFrame, metadata: pd.DataFrame,
                      schema: MetadataSchema,
                      min_year: Optional[int] = None) -> pd.DataFrame:
    """Mappt Dokument-IDs der Topic-Verteilung auf Jahre aus den Metadaten.

    Verwendet normalisierte IDs (robust gegenüber 'doc_…'/'.txt'-Varianten)
    und die Jahr-Logik des Schemas statt fest kodierter Spalten.
    """
    meta = metadata.copy()
    if "doc_id" not in meta.columns:
        # Robustheit: doc_id ggf. selbst herleiten (sonst kommt sie aus
        # DataStore.load_metadata, das ensure_doc_id bereits aufruft).
        meta = schema.ensure_doc_id(meta)
    years = schema.get_year_series(meta)
    if years is None:
        raise ValueError("Keine Jahr-Spalte in den Metadaten gefunden.")
    meta["_year"] = years
    meta["_id_norm"] = meta["doc_id"].map(normalize_document_id)
    year_map = dict(zip(meta["_id_norm"], meta["_year"]))

    df = topics_dist.copy()
    df["_id_norm"] = df.index.astype(str).map(normalize_document_id)
    df["Jahr"] = df["_id_norm"].map(year_map)
    df = df.dropna(subset=["Jahr"])
    df["Jahr"] = df["Jahr"].astype(int)

    threshold_year = min_year if min_year is not None else schema.min_year
    if threshold_year is not None:
        df = df[df["Jahr"] >= int(threshold_year)]

    if df.empty:
        raise ValueError("Keine Daten nach dem Jahr-Mapping – passen die IDs "
                         "in Metadaten und Topic-Verteilung zusammen?")
    return df.drop(columns=["_id_norm"])


def topic_year_means(topics_year_df: pd.DataFrame, topics: List[str]) -> pd.DataFrame:
    """Durchschnittliche Topic-Ähnlichkeit pro Jahr."""
    topics = [t for t in topics if t in topics_year_df.columns]
    return topics_year_df.groupby("Jahr")[topics].mean().fillna(0.0)


def topic_threshold_counts(topics_year_df: pd.DataFrame, topics: List[str],
                           threshold: float = 0.2) -> pd.DataFrame:
    """Anzahl Texte pro Jahr mit Topic-Ähnlichkeit >= Schwelle."""
    topics = [t for t in topics if t in topics_year_df.columns]
    result = pd.DataFrame(index=sorted(topics_year_df["Jahr"].unique()))
    for topic in topics:
        result[topic] = topics_year_df.groupby("Jahr")[topic].apply(
            lambda x: int((x >= threshold).sum()))
    return result.fillna(0)


# ----------------------------------------------------------------------------
# Topic-Ranking-Matching (aus dem Tag-Topic-Explorer übernommen)
# ----------------------------------------------------------------------------

def get_ranked_topics(df_ranks: pd.DataFrame, available_topics: set,
                      top_n: int) -> List[str]:
    """Top-N Topics nach Rang, mit flexiblem Spalten- und Namens-Matching."""
    df = df_ranks.copy()
    topic_col = find_column(df, ["Topic", "topic", "TOPIC", "topic_id", "Topic_ID"])
    if topic_col is None and len(df.columns):
        topic_col = df.columns[0]
    rank_col = find_column(df, ["TFIDF-Positions-Rang", "Rang", "rank", "Rank", "position"])
    if rank_col is None:
        for col in df.columns:
            if "rang" in str(col).lower() or "rank" in str(col).lower():
                rank_col = col
                break
    if topic_col is None or rank_col is None:
        return []

    df["_topic"] = df[topic_col].astype(str).str.strip()
    df["_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.dropna(subset=["_rank"]).sort_values("_rank")

    available = {str(t).strip() for t in available_topics}
    result = []
    for topic in df["_topic"]:
        if topic in available:
            result.append(topic)
        elif topic.split("(")[0].strip() in available:
            result.append(topic.split("(")[0].strip())
        elif topic.split() and topic.split()[0] in available:
            result.append(topic.split()[0])
        if len(result) >= top_n:
            break
    return result[:top_n]


def get_ranked_topics_for_counts(df_counts: pd.DataFrame, df_ranks: pd.DataFrame,
                                 top_n: int) -> List[str]:
    """Wie ``get_ranked_topics``, aber mit zusätzlichen Namens-Varianten und
    Fallback auf die Spaltensummen (Legacy-Logik 1:1 übernommen)."""
    topics_in_df = set(df_counts.columns.astype(str))

    rank_col = find_column(df_ranks, ["TFIDF-Positions-Rang", "Rang", "rank", "Rank"])
    if rank_col is None:
        for col in df_ranks.columns:
            if "rang" in str(col).lower() or "rank" in str(col).lower():
                rank_col = col
                break
    topic_col = find_column(df_ranks, ["Topic", "topic", "TOPIC"])

    def fallback() -> List[str]:
        numeric = df_counts.apply(pd.to_numeric, errors="coerce")
        return numeric.sum(axis=0, skipna=True).sort_values(ascending=False).head(top_n).index.tolist()

    if not rank_col or not topic_col:
        return fallback()

    r = df_ranks.copy()
    r["_topic"] = r[topic_col].astype(str)
    r["_rank"] = pd.to_numeric(r[rank_col], errors="coerce")
    r = r.dropna(subset=["_rank"]).sort_values("_rank")

    ranked: List[str] = []
    for topic in r["_topic"]:
        if len(ranked) >= top_n:
            break
        topic_str = str(topic).strip()
        if topic_str in topics_in_df:
            ranked.append(topic_str)
            continue
        if f"Topic {topic_str}" in topics_in_df:
            ranked.append(f"Topic {topic_str}")
            continue
        if topic_str.startswith("Topic ") and topic_str.replace("Topic ", "", 1) in topics_in_df:
            ranked.append(topic_str.replace("Topic ", "", 1))
            continue
        match = re.search(r"\d+", topic_str)
        if match:
            num = match.group()
            for variant in (f"Topic {num}", num, f"topic_{num}", f"topic{num}"):
                if variant in topics_in_df:
                    ranked.append(variant)
                    break
    return ranked or fallback()


# ----------------------------------------------------------------------------
# Tag-Topic-Bubble-Chart
# ----------------------------------------------------------------------------

def tag_topic_bubbles(df_tags: pd.DataFrame, df_topic_words: pd.DataFrame,
                      df_ranks: pd.DataFrame, tfidf_sums: pd.Series,
                      top_n: int = 10, min_size: float = 20,
                      max_size: Optional[float] = None,
                      show_values: bool = False
                      ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Bubble-Chart der Tag-Topic-Relevanz (TF-IDF-Summen gemeinsamer Terme)."""
    available = set(df_topic_words.index.astype(str))
    top_topics = get_ranked_topics(df_ranks, available, top_n)
    if not top_topics:
        raise ValueError("Keine Topics gefunden – Rankings und Topic-Words prüfen.")

    tag_dict = {col: df_tags[col].dropna().astype(str).str.strip().tolist()
                for col in df_tags.columns}
    topic_map = {str(t): df_topic_words.loc[t].dropna().astype(str).str.strip().tolist()
                 for t in df_topic_words.index}
    tfidf_dict = tfidf_sums.to_dict()

    rows = []
    for topic in top_topics:
        topic_words = set(topic_map[topic])
        for tag, expressions in tag_dict.items():
            common = topic_words.intersection(expressions)
            value = sum(tfidf_dict.get(w, 0.0) for w in common)
            rows.append({"Topic": topic, "Tag": tag, "value": value})
    df_result = pd.DataFrame(rows)
    df_result["Topic_Label"] = df_result["Topic"].str.split("(", n=1).str[0].str.strip()

    n_topics, n_tags = len(top_topics), len(tag_dict)
    fig_width = max(10, n_topics * 1.2)
    fig_height = max(6, n_tags * 0.4)
    if max_size is None:  # "auto" wie im Legacy-Tab
        cell = min(fig_width / n_topics, fig_height / n_tags) * 72 * 0.6
        max_size = min(max(min_size * 2, cell ** 2), 2000)

    values = df_result["value"].clip(lower=0).values
    if values.max() > values.min():
        normalized = (values - values.min()) / (values.max() - values.min())
    else:
        normalized = np.zeros_like(values)
    sizes = min_size + normalized * (max_size - min_size)
    sizes = np.where(values == 0, min_size * 0.3, sizes)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter = ax.scatter(df_result["Topic_Label"], df_result["Tag"], s=sizes,
                         c=values, cmap="YlOrRd", alpha=0.7,
                         edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, shrink=0.8, label="TF-IDF Summe")
    if show_values:
        for x, y, v in zip(df_result["Topic_Label"], df_result["Tag"], values):
            if v > 0:
                ax.annotate(f"{v:.0f}", (x, y), ha="center", va="center", fontsize=6)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Tag")
    ax.set_title(f"Top {n_topics} Topics – Tag-Topic-Relevanz")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, n_topics - 0.5)
    ax.set_ylim(-0.5, n_tags - 0.5)
    ax.margins(x=0.08, y=0.08)
    fig.tight_layout()
    return fig, df_result


# ----------------------------------------------------------------------------
# Stacked Bar: Topic-Texte pro Jahr
# ----------------------------------------------------------------------------

def stacked_topics_per_year(df_counts: pd.DataFrame, df_ranks: pd.DataFrame,
                            year_range: Optional[Tuple[int, int]] = None,
                            top_n: int = 10) -> plt.Figure:
    """Gestapeltes Balkendiagramm: Anzahl Texte pro Topic über die Zeit."""
    df = df_counts.drop(columns=["Anzahl Topics"], errors="ignore")
    df.index = df.index.astype(int)
    lo, hi = year_range if year_range else (int(df.index.min()), int(df.index.max()))
    df = df.reindex(range(lo, hi + 1), fill_value=0)

    topics = get_ranked_topics(df_ranks, set(df.columns.astype(str)), top_n)
    if not topics:
        raise ValueError("Keine Topics gefunden – Rankings prüfen.")
    df_plot = df[topics]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(topics)))
    df_plot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.85)
    rolling = df_plot.sum(axis=1).rolling(window=5, center=True, min_periods=1).mean()
    ax.plot(range(len(df_plot)), rolling.values, color="black", linestyle="--",
            linewidth=1.5, alpha=0.7, label="Gleitender MW (5J)")
    ticks = [i for i, y in enumerate(df_plot.index) if y % 10 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([df_plot.index[i] for i in ticks], rotation=45, ha="right")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl Texte")
    ax.set_title(f"Top {len(topics)} Topics – Texte pro Jahr")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
    fig.tight_layout()
    return fig


def tt_texts_polynomial(df_counts: pd.DataFrame, df_ranks: pd.DataFrame,
                        degree: int = 6, top_n: int = 10) -> plt.Figure:
    """Polynomiale Trendlinien der Topic-Counts pro Jahr (Top-N nach Rang)."""
    df = df_counts.drop(columns=["Anzahl Topics"], errors="ignore")
    df.index = df.index.astype(int)
    df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1), fill_value=0)

    ranked = get_ranked_topics_for_counts(df, df_ranks, top_n)
    if not ranked:
        raise ValueError("Keine Topics gefunden – Rankings prüfen.")

    x = df.index.values.astype(float)
    degree = max(1, min(degree, len(x) - 1))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ranked)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, topic in enumerate(ranked):
        if topic not in df.columns:
            continue
        y = df[topic].values.astype(float)
        if len(np.unique(y)) < 2:
            continue
        coeffs = np.polyfit(x, y, degree)
        label = topic.replace("(", "\n(") if "(" in topic else topic
        ax.plot(x, np.polyval(coeffs, x), label=label, color=colors[i], linewidth=1.2)
    ax.set_xticks([int(j) for j in x if int(j) % 10 == 0])
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl TT-Texts pro Topic")
    ax.set_title(f"TT-Texts/Jahr (Polynom Grad {degree}) – Top-{len(ranked)} nach Rang")
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Tokens vs. Topics
# ----------------------------------------------------------------------------

def tokens_vs_topics(df_tokens: pd.DataFrame, df_topdocs: pd.DataFrame,
                     year_range: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """Normalisierte Zeitreihen: Token-Anzahl vs. Topic-Relevanz pro Jahr."""
    year_col_t = find_column(df_tokens, ["year", "Jahr", "Year"])
    tokens_col = find_column(df_tokens, ["anzahl_tokens", "tokens", "count"])
    year_col_d = find_column(df_topdocs, ["Jahr", "year", "Year"])
    value_col = find_column(df_topdocs, ["Wert", "value", "Value"])
    if not all([year_col_t, tokens_col, year_col_d, value_col]):
        raise ValueError("Erforderliche Spalten (Jahr/Tokens/Wert) nicht gefunden.")

    def normalize(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    tok = df_tokens.rename(columns={year_col_t: "year", tokens_col: "tokens"}).sort_values("year")
    top = df_topdocs.rename(columns={year_col_d: "year", value_col: "value"}).sort_values("year")
    tok["norm"] = normalize(tok["tokens"].rolling(5, center=True, min_periods=1).mean())
    top["norm"] = normalize(top["value"].rolling(5, center=True, min_periods=1).mean())

    lo = year_range[0] if year_range else int(min(tok["year"].min(), top["year"].min()))
    hi = year_range[1] if year_range else int(max(tok["year"].max(), top["year"].max()))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tok["year"], tok["norm"], label="Tokens (normalisiert)",
            linestyle="--", linewidth=1.2)
    ax.plot(top["year"], top["norm"], label="TopDocs (normalisiert)",
            linestyle="-", linewidth=1.5)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Normalisierter Wert (0–1)")
    ax.set_title("Vergleich: Tokens vs. Topic-Verteilung")
    ax.set_xticks(np.arange(lo, hi + 1, 10))
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# TT-Texts-Rang mit Metadaten-Matching
# ----------------------------------------------------------------------------

def match_text_to_metadata(text_value: str, metadata_df: pd.DataFrame,
                           schema: MetadataSchema) -> Optional[str]:
    """Mappt einen freien Text-Wert auf eine Dokument-ID der Metadaten.

    Die Legacy-Version nutzte fest ``author_surname``/``title``/``year`` –
    hier kommen stattdessen die Anzeige-Spalten und Jahr-Spalten des Schemas
    zum Einsatz, damit das Matching mit beliebigen Korpora funktioniert.
    """
    text_str = str(text_value).strip().lower()
    if not text_str or text_str == "nan":
        return None

    # Strategie 1: direkte ID-Übereinstimmung
    if "doc_id" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            row_id = str(row["doc_id"]).strip()
            if row_id.lower() == text_str or row_id == str(text_value):
                return row_id

    # Textuelle Anzeige-Spalten (z. B. Autor, Titel) + Jahre aus dem Schema
    display = [c for c in schema.display_columns(metadata_df, n=3)
               if c in metadata_df.columns]
    text_cols = [c for c in display if metadata_df[c].dtype == "object"]
    years = schema.get_year_series(metadata_df)

    def year_str(idx) -> str:
        if years is None or pd.isna(years.loc[idx]):
            return ""
        return str(int(years.loc[idx]))

    # Strategie 2: alle textuellen Felder + Jahr im String enthalten
    for idx, row in metadata_df.iterrows():
        vals = [str(row.get(c, "")).strip().lower() for c in text_cols
                if pd.notna(row.get(c))]
        vals = [v for v in vals if len(v) >= 3]
        if not vals:
            continue
        all_match = all(v in text_str or v[:20] in text_str for v in vals)
        y = year_str(idx)
        if all_match and (not y or y in text_str):
            return str(row["doc_id"])

    # Strategie 3: lockerer – mindestens zwei Signale
    for idx, row in metadata_df.iterrows():
        vals = [str(row.get(c, "")).strip().lower() for c in text_cols
                if pd.notna(row.get(c))]
        vals = [v for v in vals if len(v) >= 3]
        hits = sum(1 for v in vals if v[:15] in text_str)
        y = year_str(idx)
        if y and y in text_str:
            hits += 1
        if hits >= 2:
            return str(row["doc_id"])

    # Strategie 4: normalisierte ID
    normalized_text = normalize_document_id(text_str).lstrip("0") or "0"
    if "doc_id" in metadata_df.columns:
        for _, row in metadata_df.iterrows():
            row_id = str(row["doc_id"]).strip()
            normalized_id = normalize_document_id(row_id.lower()).lstrip("0") or "0"
            if normalized_text == normalized_id:
                return row_id
    return None


def tt_texts_rank(df: pd.DataFrame, metadata_df: Optional[pd.DataFrame],
                  schema: MetadataSchema, per_topic: int = 30
                  ) -> Tuple[pd.DataFrame, int]:
    """Top-Texte pro Topic mit Rang und optionalem Metadaten-Mapping.

    Returns
    -------
    (df_result, matched_count)
    """
    if df.shape[1] < 2:
        raise ValueError("Die Datei benötigt mindestens 2 Spalten (Text, Wert).")

    text_col, value_col = df.columns[0], df.columns[1]
    topic_col = find_column(df, ["Topic", "topic", "topic_label"])

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    if topic_col:
        work = (work.sort_values([topic_col, value_col], ascending=[True, False])
                    .groupby(topic_col, group_keys=False).head(per_topic))
        work["rank"] = work.groupby(topic_col)[value_col].rank(
            method="min", ascending=False).astype(int)
    else:
        work = work.sort_values(value_col, ascending=False).head(per_topic)
        work["rank"] = work[value_col].rank(method="min", ascending=False).astype(int)

    work["_id"] = ""
    matched = 0
    if metadata_df is not None:
        for idx, row in work.iterrows():
            matched_id = match_text_to_metadata(str(row[text_col]), metadata_df, schema)
            if matched_id:
                work.at[idx, "_id"] = matched_id
                matched += 1

    result = work.rename(columns={text_col: "text"})[["_id", "text", "rank"]]
    return result.sort_values(["rank", "text"]).reset_index(drop=True), matched
