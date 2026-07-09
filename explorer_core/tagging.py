#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.tagging
=====================

UI-freie Logik für die Dashboard-Seite "POS-Liste taggen".

Aufgabe: die von ``s01_4_pos_tag.py`` erzeugte POS-Frequenzliste der Stufe
"stop" (Spalten ``word, pos, count``) um drei semantische Tag-Spalten
``tag1, tag2, tag3`` ergänzen und versioniert speichern – im Format, das
``tt_s01_stop_pos_tag.py`` erwartet.

POS-Tagging erfolgt mit **spaCy** (``de_core_news_lg``), wie in
``s01_4_pos_tag.py`` – nicht mit dem HanoverTagger (HanTa); HanTa ist im
Projekt ausschließlich für die Lemmatisierung zuständig. spaCy wird nur dann
gebraucht, wenn für NEU ergänzte Wörter ein POS-Tag bestimmt werden soll;
das Bearbeiten vorhandener Zeilen funktioniert auch ohne spaCy.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .data_store import read_csv_auto

# Pflicht- und Tag-Spalten (Reihenfolge wie von tt_s01_stop_pos_tag erwartet)
REQUIRED_COLUMNS = ["word", "pos", "count"]
TAG_COLUMNS = ["tag1", "tag2", "tag3"]
DEFAULT_POS_MODEL = "de_core_news_lg"


def load_pos_list(path: Path, delimiter: str = "auto") -> pd.DataFrame:
    """Lädt die POS-Frequenzliste und stellt die drei Tag-Spalten sicher.

    Fehlende ``tag1/2/3`` werden als leere Spalten ergänzt (z. B. wenn die
    Liste frisch aus s01_4 kommt und noch nie getaggt wurde).
    """
    df = read_csv_auto(Path(path), delimiter=delimiter)
    df.columns = [str(c).strip() for c in df.columns]
    for col in TAG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Tag-Spalten als String, NaN → leer (sauber im Editor)
    for col in TAG_COLUMNS:
        df[col] = df[col].fillna("").astype(str)
    return df


def existing_tag_lists(resources_dir: Path) -> List[Path]:
    """Bereits gespeicherte (ggf. begonnene) Tag-Listen im Zielordner.

    Grundlage dafür, eine früher begonnene Liste erneut zu laden und
    weiterzubearbeiten (neueste zuerst über den Versionszähler im Namen).
    """
    resources_dir = Path(resources_dir)
    if not resources_dir.exists():
        return []
    return sorted(resources_dir.glob("*.csv"), reverse=True)


def validate(df: pd.DataFrame) -> Tuple[bool, str]:
    """Prüft, ob die Pflichtspalten vorhanden sind."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, ("Es fehlen Pflichtspalten: " + ", ".join(missing) +
                       f". Vorhanden: {', '.join(map(str, df.columns))}")
    return True, "ok"


def existing_tags(df: pd.DataFrame) -> List[str]:
    """Alle bereits vergebenen Tagwerte (für Wiederverwendung/Vorschläge)."""
    values = set()
    for col in TAG_COLUMNS:
        if col in df.columns:
            for v in df[col].dropna().astype(str):
                v = v.strip()
                if v:
                    values.add(v)
    return sorted(values, key=str)


def pos_for_words(words: List[str], model: str = DEFAULT_POS_MODEL) -> dict:
    """Bestimmt POS-Tags für (neu ergänzte) Wörter mit spaCy.

    Lazy import: spaCy wird nur geladen, wenn diese Funktion wirklich
    aufgerufen wird – das Dashboard läuft sonst auch ohne spaCy.
    """
    import spacy  # lazy
    nlp = spacy.load(model)
    result = {}
    for w in words:
        doc = nlp(str(w))
        tag = ""
        for token in doc:
            if not token.is_space:
                tag = token.pos_
                break
        result[w] = tag
    return result


def next_version_path(resources_dir: Path,
                      base: str = "vocab_top5000_stop_pos_tag") -> Path:
    """Nächsten freien versionierten Dateinamen finden (…_v1, _v2, …).

    Überschreibt nie eine vorhandene Datei.
    """
    resources_dir = Path(resources_dir)
    resources_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in resources_dir.glob(f"{base}_v*.csv")}
    n = 1
    while f"{base}_v{n}.csv" in existing:
        n += 1
    return resources_dir / f"{base}_v{n}.csv"


def normalize_for_save(df: pd.DataFrame) -> pd.DataFrame:
    """Bringt die Tabelle in die Zielreihenfolge word, pos, count, tag1–tag3.

    Zusätzliche Spalten bleiben hinten erhalten. Tag-Werte werden getrimmt.
    """
    out = df.copy()
    for col in TAG_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.strip()
    ordered = [c for c in REQUIRED_COLUMNS + TAG_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest]


def save_tagged(df: pd.DataFrame, path: Path) -> Path:
    """Speichert die getaggte Liste als UTF-8-CSV (kommagetrennt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_for_save(df).to_csv(path, index=False, encoding="utf-8")
    return path


def tagging_progress(df: pd.DataFrame) -> Tuple[int, int]:
    """(Anzahl getaggter Zeilen, Gesamtzeilen) – für eine Fortschrittsanzeige.

    "Getaggt" = mindestens ein Tag in tag1/tag2/tag3 gesetzt.
    """
    if df.empty:
        return 0, 0
    has_tag = pd.Series(False, index=df.index)
    for col in TAG_COLUMNS:
        if col in df.columns:
            has_tag = has_tag | df[col].fillna("").astype(str).str.strip().ne("")
    return int(has_tag.sum()), int(len(df))
