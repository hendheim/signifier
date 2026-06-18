#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.schema
====================

Zentrales, konfigurierbares Metadatenschema.

Diese eine Klasse ersetzt die DREI bisherigen, inkonsistenten Erkennungen:

- ``MetadataDetector`` in ``gui_corpus_explorer.py``
  (Mapping: Metadaten = Spalten, die auch in metadata.csv vorkommen)
- ``MetadataDetector`` in ``gui_tag_topic_explorer.py``
  (Heuristiken: Namenslisten, Regex-Patterns, Kardinalität)
- Kandidatenlisten in ``pipeline_utils.py``
  (ID-/Jahr-/Content-Spalten-Erkennung)

Prinzip (didaktisch wichtig):
1. Optional wird ein YAML-Schema (``config/metadata_schema.yaml``) geladen.
2. Was dort fehlt, wird automatisch erkannt – mit denselben Heuristiken
   wie bisher. Bestehende Korpora funktionieren daher OHNE Änderung.
3. Wird zusätzlich eine Metadatendatei registriert (``register_metadata``),
   gilt wie im alten Korpus-Explorer: Metadaten = Spalten der Metadatendatei,
   Terme = alle anderen numerischen Spalten. Das ist die zuverlässigste
   Methode, DTM-/TF-IDF-Spalten zu klassifizieren.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yaml  # PyYAML – einzige neue Kern-Abhängigkeit (winzig, Standard)
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - yaml gehört zu requirements.txt
    _YAML_AVAILABLE = False


# ----------------------------------------------------------------------------
# Fallback-Kandidaten – identisch zu den bisherigen Listen in pipeline_utils.py
# (Diese Werte gelten, wenn KEIN YAML vorhanden ist → volle Abwärtskompatibilität)
# ----------------------------------------------------------------------------
DEFAULT_ID_CANDIDATES = ["_id", "id", "doc_id", "document_id", "filename", "file_id", "index"]
DEFAULT_CONTENT_CANDIDATES = [
    "content_stop", "content_lem", "content_min", "content_gen",
    "content", "text", "clean_text", "body", "fulltext",
]
DEFAULT_YEAR_FIRST = ["year_first", "Jahr_first", "jahr_first"]
DEFAULT_YEAR_MAIN = ["year", "jahr", "Jahr", "year_final", "Jahr_final", "date"]
DEFAULT_DISPLAY = ["author_surname", "title", "year_final"]
DEFAULT_FACETS = ["textclass", "genre"]

# Spalten, die nie als Anzeige-Metadatum angeboten werden
_CONTENT_PREFIXES = ("content", "text_", "clean_text", "fulltext", "body")


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Findet die erste vorhandene Spalte (case-insensitive, whitespace-tolerant).

    Identisch zur gleichnamigen Funktion in den Legacy-GUIs – hier nur
    EINMAL definiert statt zweimal.
    """
    normalized = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        hit = normalized.get(str(cand).lower().strip())
        if hit is not None:
            return hit
    return None


class MetadataSchema:
    """Konfigurierbares Metadatenschema mit Auto-Detection-Fallback.

    Parameters
    ----------
    yaml_path:
        Pfad zur YAML-Datei. ``None`` oder nicht vorhandene Datei → reine
        Auto-Detection mit den bisherigen Standard-Heuristiken.
    """

    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = Path(yaml_path) if yaml_path else None
        cfg: dict = {}
        if self.yaml_path and self.yaml_path.exists() and _YAML_AVAILABLE:
            try:
                cfg = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}  # defekte YAML → Fallback auf Defaults
        self._cfg = cfg

        year_cfg = cfg.get("year") or {}
        self.id_candidates: List[str] = list(cfg.get("id_candidates") or DEFAULT_ID_CANDIDATES)
        self.content_candidates: List[str] = list(cfg.get("content_candidates") or DEFAULT_CONTENT_CANDIDATES)
        self.year_first_candidates: List[str] = list(year_cfg.get("first_candidates") or DEFAULT_YEAR_FIRST)
        self.year_main_candidates: List[str] = list(year_cfg.get("main_candidates") or DEFAULT_YEAR_MAIN)
        self.display_columns_cfg: List[str] = list(cfg.get("display_columns") or [])
        self.facets_cfg: List[str] = list(cfg.get("facets") or [])
        self.min_year: Optional[int] = cfg.get("min_year")
        self.delimiter: str = cfg.get("delimiter") or "auto"

        # Spaltennamen der registrierten Metadatendatei (lowercased)
        self._metadata_columns: set = set()

    # ------------------------------------------------------------------
    # Registrierung der Metadatendatei (Mapping-Methode des alten Explorers)
    # ------------------------------------------------------------------

    def register_metadata(self, metadata_df: pd.DataFrame) -> None:
        """Merkt sich die Spalten der Metadatendatei.

        Danach gilt für DTM/TF-IDF: Metadaten = Spalten, die auch hier
        vorkommen; Terme = alle anderen numerischen Spalten.
        """
        self._metadata_columns = {str(c).lower().strip() for c in metadata_df.columns}
        # year_final entsteht erst durch coalesce_years → mitregistrieren
        self._metadata_columns.add("year_final")

    @property
    def metadata_registered(self) -> bool:
        return bool(self._metadata_columns)

    def is_metadata_column(self, col: str) -> bool:
        return str(col).lower().strip() in self._metadata_columns

    # ------------------------------------------------------------------
    # Rollen-Erkennung
    # ------------------------------------------------------------------

    def find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        return find_column(df, self.id_candidates)

    def find_content_column(self, df: pd.DataFrame) -> Optional[str]:
        return find_column(df, self.content_candidates)

    def find_year_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """(year_first_col, year_main_col) – jeweils None, wenn nicht vorhanden."""
        return (
            find_column(df, self.year_first_candidates),
            find_column(df, self.year_main_candidates),
        )

    def get_year_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Kombinierte Jahr-Serie: 'first' (Erstausgabe) hat Vorrang vor 'main'."""
        yf_col, y_col = self.find_year_columns(df)
        if not yf_col and not y_col:
            return None
        to_num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
        yf = to_num(df[yf_col]) if yf_col else pd.Series(index=df.index, dtype="float64")
        y = to_num(df[y_col]) if y_col else pd.Series(index=df.index, dtype="float64")
        return yf.where(~yf.isna(), y)

    def coalesce_years(self, df: pd.DataFrame, output_col: str = "year_final") -> pd.DataFrame:
        """Fügt eine kombinierte Jahr-Spalte hinzu (wie früher ``coalesce_years``)."""
        years = self.get_year_series(df)
        if years is not None:
            df = df.copy()
            df[output_col] = years
        return df

    def ensure_doc_id(self, df: pd.DataFrame, output_col: str = "doc_id") -> pd.DataFrame:
        """Stellt eine String-ID-Spalte sicher (Fallback: laufende Nummer)."""
        df = df.copy()
        id_col = self.find_id_column(df)
        if id_col and df[id_col].notna().any():
            df[output_col] = df[id_col].astype(str)
        else:
            df[output_col] = [str(i) for i in range(1, len(df) + 1)]
        return df

    # ------------------------------------------------------------------
    # Term- vs. Metadaten-Spalten (DTM/TF-IDF)
    # ------------------------------------------------------------------

    def metadata_columns_in(self, df: pd.DataFrame) -> List[str]:
        """Spalten von df, die als Metadaten gelten."""
        if self.metadata_registered:
            return [c for c in df.columns if self.is_metadata_column(c)]
        # Fallback ohne registrierte Metadaten: nicht-numerische Spalten
        # plus bekannte Rollen-Spalten gelten als Metadaten.
        known = set(
            x.lower() for x in (
                self.id_candidates + self.year_first_candidates
                + self.year_main_candidates + self.content_candidates
                + self.display_columns_cfg + self.facets_cfg + ["year_final", "doc_id"]
            )
        )
        result = []
        for col in df.columns:
            if str(col).lower().strip() in known:
                result.append(col)
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() / max(len(series), 1) < 0.8:
                result.append(col)
        return result

    def term_columns_in(self, df: pd.DataFrame) -> List[str]:
        """Term-Spalten = numerische Spalten, die keine Metadaten sind.

        Exakt die Logik des alten ``MetadataDetector.get_term_columns``.
        """
        meta = set(self.metadata_columns_in(df))
        result = []
        for col in df.columns:
            if col in meta:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                result.append(col)
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() / max(len(series), 1) >= 0.8:
                result.append(col)
        return result

    # ------------------------------------------------------------------
    # Anzeige & Facetten
    # ------------------------------------------------------------------

    def selectable_metadata(self, metadata_df: pd.DataFrame) -> List[str]:
        """Alle Metadaten-Spalten ohne ID/Content – für Dropdowns im Dashboard."""
        id_like = {c.lower() for c in self.id_candidates} | {"doc_id"}
        result = []
        for col in metadata_df.columns:
            cl = str(col).lower()
            if cl in id_like:
                continue
            if cl in {"content", "text", "clean_text", "cleaned_text"}:
                continue
            if any(cl.startswith(p) for p in _CONTENT_PREFIXES):
                continue
            result.append(col)
        return sorted(result, key=str.lower)

    def display_columns(self, metadata_df: pd.DataFrame, n: int = 3) -> List[str]:
        """Default-Anzeigespalten: konfiguriert > automatisch ergänzt.

        Sind weniger als ``n`` konfigurierte Spalten vorhanden (z. B. bei
        einem neuen Korpus mit anderen Spaltennamen), werden die übrigen
        Plätze mit textuellen Metadaten-Spalten aufgefüllt.
        """
        available = self.selectable_metadata(metadata_df)
        result = [c for c in self.display_columns_cfg if c in metadata_df.columns]
        # Auffüllen: zuerst textuelle Spalten (Autor/Titel-artig), dann Rest
        text_first = sorted(
            available,
            key=lambda c: (pd.api.types.is_numeric_dtype(metadata_df[c]), available.index(c)),
        )
        for col in text_first:
            if len(result) >= n:
                break
            if col not in result:
                result.append(col)
        return result[:n]

    def facet_columns(self, metadata_df: pd.DataFrame, max_unique: int = 30) -> List[str]:
        """Kategoriale Felder (Färbung/Gruppierung): konfiguriert > Auto-Erkennung
        über geringe Wertevielfalt."""
        configured = [c for c in self.facets_cfg if c in metadata_df.columns]
        if configured:
            return configured
        result = []
        for col in self.selectable_metadata(metadata_df):
            series = metadata_df[col]
            # dtype-agnostisch: pandas >= 3 nutzt 'str' statt 'object' für Text
            if pd.api.types.is_numeric_dtype(series):
                continue
            if 1 < series.nunique(dropna=True) <= max_unique:
                result.append(col)
        return result

    # ------------------------------------------------------------------
    # Speichern (für den Schema-Editor im Dashboard)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id_candidates": self.id_candidates,
            "content_candidates": self.content_candidates,
            "year": {
                "first_candidates": self.year_first_candidates,
                "main_candidates": self.year_main_candidates,
            },
            "display_columns": self.display_columns_cfg,
            "facets": self.facets_cfg,
            "min_year": self.min_year,
            "delimiter": self.delimiter,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Schreibt das aktuelle Schema als YAML (für den Schema-Editor)."""
        if not _YAML_AVAILABLE:
            raise RuntimeError("PyYAML ist nicht installiert (pip install pyyaml).")
        target = Path(path) if path else self.yaml_path
        if target is None:
            raise ValueError("Kein Zielpfad für das Schema angegeben.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.yaml_path = target
        return target
