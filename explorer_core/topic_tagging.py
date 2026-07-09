#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.topic_tagging
===========================

UI-freie Logik für die Dashboard-Seite "Topics taggen".

Datengrundlage ist eine **Topic-Word-Matrix** (Ausgabe von ``topic_model.py``
bzw. MALLET): erste Spalte = Topic-ID, weitere Spalten = die Top-Wörter des
Topics in Rangfolge (typischerweise 100). Die Seite zeigt die **vollständige**
Matrix (alle Wörter, horizontal scrollbar) und lässt in der **Topic-ID-Spalte**
einen frei formulierten, komplexen **Namen** je Topic eintragen. Gespeichert
wird versioniert.

Es wird kein spaCy benötigt; eine bereits benannte Tabelle kann erneut geladen
und weiterbearbeitet werden.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .data_store import read_csv_auto

# Spaltennamen, unter denen die Topic-ID erwartet wird (Reihenfolge = Priorität).
TOPIC_ID_CANDIDATES = ["Topic", "topic", "Topic_ID", "topic_id", "ID", "id"]


def topic_id_column(df: pd.DataFrame) -> str:
    """Bestimmt die Topic-ID-Spalte; Fallback ist die erste Spalte."""
    for cand in TOPIC_ID_CANDIDATES:
        if cand in df.columns:
            return cand
    return str(df.columns[0])


def word_columns(df: pd.DataFrame, topic_col: Optional[str] = None) -> List[str]:
    """Alle Wortspalten (= Matrix ohne die Topic-ID-Spalte)."""
    topic_col = topic_col or topic_id_column(df)
    return [c for c in df.columns if c != topic_col]


def load_topic_table(path: Path, delimiter: str = "auto") -> pd.DataFrame:
    """Lädt die **vollständige** Topic-Word-Matrix.

    Ergebnis: Topic-ID-Spalte (Träger des editierbaren Namens) + **alle**
    Wortspalten unverändert (keine gekürzte Vorschau). Eine bereits benannte
    Tabelle (gleiche Struktur) kann ebenso geladen und fortgesetzt werden.
    """
    df = read_csv_auto(Path(path), delimiter=delimiter)
    df.columns = [str(c).strip() for c in df.columns]
    topic_col = topic_id_column(df)
    out = df.copy()
    out[topic_col] = out[topic_col].astype(str)
    return out[[topic_col] + word_columns(out, topic_col)]


def validate(df: pd.DataFrame) -> Tuple[bool, str]:
    """Prüft, ob eine echte Topic-Word-Matrix vorliegt."""
    if df is None or df.empty:
        return False, "Die Tabelle ist leer – keine Topics gefunden."
    if df.shape[1] < 2:
        return False, ("Keine Wortspalten gefunden – ist das wirklich eine "
                       "Topic-Word-Matrix (Topic-Spalte + Wortspalten)?")
    return True, "ok"


def naming_progress(df: pd.DataFrame,
                    topic_col: Optional[str] = None) -> Tuple[int, int]:
    """(benannte, gesamt). „Benannt" = die Topic-Spalte enthält keinen reinen
    Zahlencode mehr (es wurde also ein Name eingetragen)."""
    if df.empty:
        return 0, 0
    topic_col = topic_col or topic_id_column(df)
    vals = df[topic_col].fillna("").astype(str).str.strip()
    named = vals.map(lambda v: bool(v) and not v.isdigit())
    return int(named.sum()), int(len(df))


def next_version_path(target_dir: Path, base: str = "topic_names") -> Path:
    """Nächsten freien versionierten Dateinamen finden (…_v1, _v2, …)."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in target_dir.glob(f"{base}_v*.csv")}
    n = 1
    while f"{base}_v{n}.csv" in existing:
        n += 1
    return target_dir / f"{base}_v{n}.csv"


def model_dir_and_name(src_path: Path) -> Tuple[Path, str]:
    """(Topic-Modell-Ordner, Modellname) aus dem Pfad der Quell-Matrix.

    Der Modellname ist der Name des Topic-Modell-Ordners (Elternordner der
    gewählten Topic-Word-Matrix), z. B. ``sklearn_lda_10``.
    """
    src_path = Path(src_path)
    folder = src_path.parent
    return folder, folder.name


def next_tag_version_path(folder: Path, model_name: str,
                          matrix_type: str) -> Path:
    """Nächster freier Name ``{model}_{matrix_type}_tag_v{n}.csv`` im Ordner."""
    folder = Path(folder)
    base = f"{model_name}_{matrix_type}_tag"
    existing = {p.name for p in folder.glob(f"{base}_v*.csv")}
    n = 1
    while f"{base}_v{n}.csv" in existing:
        n += 1
    return folder / f"{base}_v{n}.csv"


def find_document_topic_matrices(folder: Path) -> List[Path]:
    """Document-Topic-Verteilungen im Topic-Modell-Ordner.

    ``*_pro_text``-Varianten werden zuerst gelistet (die für die Auswertung
    genutzte Textebene); bereits getaggte Ausgaben (``_tag_v*``) werden
    ausgeschlossen.
    """
    folder = Path(folder)
    hits = [p for p in sorted(folder.glob("document-topics-distribution*.csv"))
            if "_tag_v" not in p.name]
    hits.sort(key=lambda p: (0 if "pro_text" in p.name else 1, p.name))
    return hits


def save_named_document_topics(dist_path: Path, names_in_order: List[str],
                               out_path: Path) -> Tuple[Path, int]:
    """Benennt die Topic-Spalten der Document-Topic-Matrix um und speichert.

    Zuordnung **positionsbasiert**: Erste Spalte = Dokument-ID (bleibt),
    die übrigen Spalten sind die Topics in Reihenfolge (0,1,…) und werden
    der Reihe nach mit ``names_in_order`` (den Namen aus der Topic-Word-
    Matrix, Zeilenreihenfolge) abgeglichen. Nur wo ein Name eingetragen
    wurde, der vom bisherigen Spaltenlabel abweicht, wird umbenannt – so
    funktioniert es auch beim Weiterbearbeiten einer bereits (teilweise)
    benannten Liste. Gibt ``(Ausgabepfad, Anzahl umbenannter Topics)`` zurück.
    """
    dist_path, out_path = Path(dist_path), Path(out_path)
    df = read_csv_auto(dist_path, delimiter="auto")
    topic_cols = list(df.columns[1:])
    rename: dict = {}
    for col, name in zip(topic_cols, names_in_order):
        name = str(name).strip()
        if name and name != str(col).strip():
            rename[col] = name
    df = df.rename(columns=rename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path, len(rename)


def save_named(df: pd.DataFrame, path: Path,
               topic_col: Optional[str] = None) -> Path:
    """Speichert die benannte Topic-Word-Matrix (UTF-8, kommagetrennt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    topic_col = topic_col or topic_id_column(out)
    out[topic_col] = out[topic_col].fillna("").astype(str).str.strip()
    out.to_csv(path, index=False, encoding="utf-8")
    return path
