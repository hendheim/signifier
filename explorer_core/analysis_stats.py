#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.analysis_stats
============================

UI-freie Logik für die Dashboard-Seite **Statistik** (ersetzt den früheren
Pipeline-Schritt ``s01_3_statistics``). Alles ist metadatengetrieben und
auswählbar:

- **Token je Stufe** (content/min/lem/stop/gen): Gesamtzahl + pro Dokument.
- **Milestones**: Gesamttokens / N → N gleich große Token-Abschnitte
  (chronologisch), je mit Jahresspanne und Tokenzahl
  (z. B. ``1788–1890; 1.000.298 Tokens``).
- **Frequenz je Metadatenspalte**: Dokumente *und* Tokens je Wert – beides in
  einer Grafik.

Tokenzählung = whitespace-getrennt (die verarbeiteten Stufen sind bereits
tokenisiert/leerzeichensepariert). Jahre kommen über das Metadatenschema
(``get_year_series``), funktionieren also auch mit einer ``date``-Spalte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from .data_store import read_csv_auto
from .schema import MetadataSchema

# Stufe -> (Dateiname, bevorzugte Content-Spalte). 'content' = Rohkorpus.
STAGE_FILES: Dict[str, Tuple[str, str]] = {
    "content": ("korpus.csv", "content"),
    "min": ("korpus_min.csv", "content_min"),
    "lem": ("korpus_lem.csv", "content_lem"),
    "stop": ("korpus_stop.csv", "content_stop"),
    "gen": ("korpus_gen.csv", "content_gen"),
}
_CONTENT_FALLBACKS = ["content_stop", "content_lem", "content_min", "content_gen",
                      "content", "text"]


def _token_counts(series: pd.Series) -> pd.Series:
    """Tokens pro Zelle (whitespace-getrennt).

    Vektorisiert über ``str.count`` statt ``map(len(split()))`` – zählt
    identisch, materialisiert aber keine Token-Listen (bei ~45 Mio. Tokens
    Sekunden statt Minuten und ein Bruchteil des Speichers).
    """
    return (series.fillna("").astype(str)
            .str.count(r"\S+").fillna(0).astype(int))


def stage_file_stamp(project_root: Path, stages: List[str]) -> tuple:
    """(Stufe, mtime_ns) je vorhandener Stufen-Datei.

    Als Cache-Key für die Dashboard-Seite gedacht: Ändert sich eine
    Korpus-Datei (neuer Pipeline-Lauf), ändert sich der Stempel und der
    Streamlit-Cache wird automatisch ungültig.
    """
    out = []
    for s in stages:
        fname, _ = STAGE_FILES.get(s, STAGE_FILES["stop"])
        p = _stage_path(project_root, s, fname)
        try:
            out.append((s, p.stat().st_mtime_ns))
        except OSError:
            pass
    return tuple(out)


def _content_col(df: pd.DataFrame, preferred: str) -> Optional[str]:
    if preferred in df.columns:
        return preferred
    for c in _CONTENT_FALLBACKS:
        if c in df.columns:
            return c
    return None


def _stage_path(project_root: Path, stage: str, fname: str) -> Path:
    root = Path(project_root)
    return (root / "korpus" / fname) if stage == "content" \
        else (root / "output" / "processed_corpus" / fname)


def available_stages(project_root: Path) -> List[str]:
    """Welche Stufen liegen als Datei vor (für die Checkbox-Auswahl)."""
    return [s for s, (f, _) in STAGE_FILES.items()
            if _stage_path(project_root, s, f).exists()]


def stage_token_summary(project_root: Path, stages: List[str],
                        schema: MetadataSchema) -> pd.DataFrame:
    """Pro gewählter Stufe: Dokumente, Gesamttokens, Ø Tokens/Dokument."""
    rows = []
    for stage in stages:
        fname, col = STAGE_FILES[stage]
        path = _stage_path(project_root, stage, fname)
        if not path.exists():
            continue
        df = read_csv_auto(path, schema.delimiter)
        cc = _content_col(df, col)
        if cc is None:
            continue
        toks = _token_counts(df[cc])
        n = len(df)
        rows.append({"stufe": stage, "dokumente": n, "tokens": int(toks.sum()),
                     "tokens_pro_dok": round(toks.mean(), 1) if n else 0})
    return pd.DataFrame(rows, columns=["stufe", "dokumente", "tokens", "tokens_pro_dok"])


def base_doc_table(project_root: Path, schema: MetadataSchema,
                   stage: str = "min") -> pd.DataFrame:
    """Pro Dokument: alle Metadaten + ``_tokens`` (gewählte Stufe) + ``_year``.

    Grundlage für Milestones und die Spalten-Frequenz.
    """
    fname, col = STAGE_FILES.get(stage, STAGE_FILES["stop"])
    path = _stage_path(project_root, stage, fname)
    df = read_csv_auto(path, schema.delimiter).copy()
    cc = _content_col(df, col)
    df["_tokens"] = _token_counts(df[cc]) if cc else 0
    years = schema.get_year_series(df)
    df["_year"] = years if years is not None else pd.Series(index=df.index, dtype="float64")
    return df


def token_milestones(doc_table: pd.DataFrame, n_milestones: int
                     ) -> Tuple[int, pd.DataFrame]:
    """Gesamttokens / N → N gleich große Token-Abschnitte chronologisch.

    Dokumente werden nach Jahr sortiert und so in N Abschnitte geteilt, dass
    jeder ~``gesamt/N`` Tokens enthält. Je Abschnitt: Jahresspanne + Tokenzahl.

    Returns ``(gesamttokens, df)`` mit Spalten
    ``meilenstein, jahr_von, jahr_bis, tokens``.
    """
    d = doc_table.dropna(subset=["_year"]).copy()
    d["_year"] = d["_year"].astype(int)
    d = d.sort_values("_year")
    total = int(d["_tokens"].sum())
    n = max(1, int(n_milestones))
    target = total / n if total else 0

    rows: List[dict] = []
    cur_years: List[int] = []
    cur_tok = 0
    cum = 0
    m = 1
    for _, r in d.iterrows():
        cur_years.append(int(r["_year"]))
        cur_tok += int(r["_tokens"])
        cum += int(r["_tokens"])
        if m < n and target and cum >= m * target:
            rows.append({"meilenstein": m, "jahr_von": min(cur_years),
                         "jahr_bis": max(cur_years), "tokens": cur_tok})
            cur_years, cur_tok = [], 0
            m += 1
    if cur_years:
        rows.append({"meilenstein": m, "jahr_von": min(cur_years),
                     "jahr_bis": max(cur_years), "tokens": cur_tok})

    df = pd.DataFrame(rows, columns=["meilenstein", "jahr_von", "jahr_bis", "tokens"])
    return total, df


def year_token_counts(doc_table: pd.DataFrame) -> pd.DataFrame:
    """Dokumente + Tokens pro Jahr (auch für ``year_count_tokens.csv``)."""
    d = doc_table.dropna(subset=["_year"]).copy()
    d["_year"] = d["_year"].astype(int)
    out = (d.groupby("_year")
           .agg(documents=("_tokens", "size"), tokens=("_tokens", "sum"))
           .reset_index().rename(columns={"_year": "year"}))
    return out.sort_values("year")


def metadata_columns(doc_table: pd.DataFrame, schema: MetadataSchema) -> List[str]:
    """Metadatenspalten (ohne Content/ID/Hilfsspalten) für die Frequenz-Auswahl."""
    drop = {"_tokens", "_year"} | {c for c in doc_table.columns
                                   if str(c).startswith("content")}
    id_col = schema.find_id_column(doc_table)
    if id_col:
        drop.add(id_col)
    return [c for c in doc_table.columns if c not in drop]


def metadata_frequency(doc_table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pro Wert der Spalte: Dokumente + summierte Tokens (absteigend)."""
    g = (doc_table.dropna(subset=[column])
         .groupby(column)
         .agg(dokumente=("_tokens", "size"), tokens=("_tokens", "sum"))
         .reset_index())
    return g.sort_values("tokens", ascending=False).reset_index(drop=True)


def plot_metadata_frequency(freq: pd.DataFrame, column: str,
                            top: int = 30) -> plt.Figure:
    """Kombigrafik: Tokens (links) und Dokumente (rechts) je Wert in einer Figur.

    Den ``Dokumente``-Balken nur zeigen, wenn die Spalte mehrere Dokumente je
    Wert hat. Bei eindeutigen Spalten (z. B. ``title``: jeder Wert genau 1
    Dokument) ist diese Reihe sinnlos – dann nur Tokens, einachsig.
    """
    d = freq.head(top)
    labels = d[column].astype(str).tolist()
    x = list(range(len(d)))
    fig, ax = plt.subplots(figsize=(max(8, len(d) * 0.4), 6))
    show_docs = bool((freq["dokumente"] != 1).any())
    if show_docs:
        ax.bar([i - 0.2 for i in x], d["tokens"], width=0.4,
               color="#3b6", label="Tokens")
        ax2 = ax.twinx()
        ax2.bar([i + 0.2 for i in x], d["dokumente"], width=0.4,
                color="#e69", label="Dokumente")
        ax2.set_ylabel("Dokumente", color="#e69")
    else:
        ax.bar(x, d["tokens"], width=0.6, color="#3b6", label="Tokens")
    ax.set_ylabel("Tokens", color="#3b6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    suffix = f" (Top {top})" if len(freq) > top else ""
    reihen = "Dokumente & Tokens" if show_docs else "Tokens (je Wert 1 Dokument)"
    ax.set_title(f"Frequenz je '{column}': {reihen}{suffix}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
