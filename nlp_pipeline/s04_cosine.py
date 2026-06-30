#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Berechnung einer Cosinus-Ähnlichkeitsmatrix auf Grundlage von TF-IDF/DTM-Matrizen.

ÄNDERUNG v3:
- Verwendet gemeinsame pipeline_utils für konsistente Erkennung
- Erweiterte ID-Erkennung (doc_id > _id > id > filename)
- Content-Spalten-Ausschluss
- Konsistent mit Pipeline v3

Input-Datei:
    output/dtm_tfidf_stop/tfidf-2000.csv (oder jede andere TF-IDF/DTM-Matrix)

Output-Datei:
    output/cosine/cosine_tfidf2000.csv

Beispielaufruf:

    python s04_cosine.py \\
        --input output/dtm_tfidf_stop/tfidf-2000.csv \\
        --output output/cosine/cosine_tfidf2000.csv
"""

import argparse
import re
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        identify_content_column,
        identify_id_column
    )
except ImportError:
    from pipeline_utils import (
        identify_content_column,
        identify_id_column
    )


# =============================================================================
# FLEXIBLE METADATEN-ERKENNUNG
# =============================================================================

KNOWN_METADATA_NAMES = {
    "_id", "id", "doc_id", "filename",
    "author", "author_prename", "author_surname", "author_surname_norm", "author_address", "author_address_geo",
    "editor_prename", "editor_surname",
    "title", "title_norm", "title_addition",
    "source", "journal", "magazine",
    "year", "year_first", "year_final", "Jahr_final",
    "date", "datum", "year_pub", "erscheinungsjahr",
    "author_forename", "editor_forename",
    "gnd", "wikidata", "idno", "ref", "url",
    "volume", "edition", "issue", "pages", "pages_exzerpt",
    "textclass", "genre", "address", "address_geo",
    "lang", "language", "note", "archive",
    "female_education",
    # Content-Spalten (werden NICHT als Features gezählt!)
    "content", "text", "clean_text", "content_min", "content_lem", "content_stop", "content_gen"
}


# Metadaten-Namensmuster: erkennt zusammengesetzte Metadatenfelder zuverlässig,
# ohne echte Term-Spalten (einzelne Lemmata ohne Unterstrich) zu treffen.
_METADATA_NAME = re.compile(
    r"^(author|editor|autor|verfasser|herausgeber|hrsg)_"          # author_*, editor_* ...
    r"|_(forename|surname|prename|vorname|nachname|name|norm|geo|"
    r"addition|first|final|exzerpt|address)$",                     # *_forename, *_norm ...
    re.I)


def is_metadata_column(col_name: str) -> bool:
    """Prüft, ob eine Spalte eine Metadaten-Spalte ist (Name oder Muster)."""
    col_lower = str(col_name).strip().lower()
    if col_name in KNOWN_METADATA_NAMES or col_lower in {n.lower() for n in KNOWN_METADATA_NAMES}:
        return True
    if _METADATA_NAME.search(str(col_name)):
        return True
    return False


_NONFEATURE_NAME = re.compile(
    r"(^|_)(id|gnd|wikidata|idno|jahr|year|datum|date|band|volume|"
    r"seite|page|pages|nr|nummer|index|ref|url)$", re.I)


def identify_feature_columns(df: pd.DataFrame, exclude_content: bool = True) -> List[str]:
    """
    Identifiziert Feature-Spalten (= numerische Term-Spalten, KEINE Metadaten).

    Zusätzlich werden numerische Spalten verworfen, deren Name wie eine ID,
    Jahres-/Datumsangabe o. Ä. aussieht (z. B. ``date``, ``gnd``) – solche
    grossen Zahlen würden den Kosinus dominieren und alle Werte gegen 1 treiben.
    """
    feature_cols = []
    content_col = identify_content_column(df) if exclude_content else None

    for col in df.columns:
        if content_col and col == content_col:
            continue
        if is_metadata_column(col):
            continue
        if _NONFEATURE_NAME.search(str(col)):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def identify_metadata_columns_cosine(df: pd.DataFrame) -> List[str]:
    """Identifiziert Metadaten-Spalten (alle nicht-Feature-Spalten)."""
    feature_cols = set(identify_feature_columns(df, exclude_content=True))
    return [col for col in df.columns if col not in feature_cols]


# =============================================================================
# Funktionen
# =============================================================================

def load_tfidf_matrix(path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Lädt TF-IDF/DTM-Matrix und trennt Metadaten von Feature-Spalten.
    
    Returns:
        (df, metadata_columns, feature_columns)
    """
    if not path.exists():
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {path}")

    print(f"📄 Lade Datei: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    
    # Automatische Trennung
    meta_cols = identify_metadata_columns_cosine(df)
    feature_cols = identify_feature_columns(df, exclude_content=True)
    
    if not feature_cols:
        raise ValueError("❌ Keine Feature-Spalten gefunden.")
    
    # Content-Spalte erkennen (falls vorhanden)
    content_col = identify_content_column(df)
    if content_col:
        print(f"   ⚠️ Warnung: Content-Spalte '{content_col}' in Matrix gefunden (wird ignoriert)")
    
    # ID-Spalte erkennen
    id_col = identify_id_column(df)
    if id_col:
        print(f"   🔑 ID-Spalte: {id_col}")
    
    print(f"   📋 Metadaten: {len(meta_cols)} Spalten")
    print(f"   📊 Features: {len(feature_cols)} Spalten")
    
    # Features bereinigen: numerisch erzwingen, NaN → 0
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[feature_cols] = features

    return df, meta_cols, feature_cols


def compute_cosine(df_features: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Cosinus-Ähnlichkeitsmatrix."""
    if df_features.isna().values.any():
        raise ValueError("❌ Nach Bereinigung sind noch NaN in den Features vorhanden.")

    matrix = df_features.to_numpy(dtype=float)

    if matrix.size == 0:
        raise ValueError("❌ Die Matrix enthält keine Daten.")

    print(f"   🔢 Matrix-Größe: {matrix.shape[0]} Dokumente × {matrix.shape[1]} Features")
    
    cos = cosine_similarity(matrix)
    return pd.DataFrame(cos)


# =============================================================================
# run-Funktion für Pipeline
# =============================================================================

def run(
    input_path: Path,
    output_path: Path,
) -> None:
    """Berechnet eine Cosinusmatrix aus TF-IDF/DTM-CSV."""

    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print()
    
    df, meta_cols, feature_cols = load_tfidf_matrix(input_path)

    # Dokument-IDs bestimmen
    id_col = identify_id_column(df)
    
    if id_col:
        doc_ids = df[id_col].fillna("").astype(str).tolist()
        print(f"   📍 Verwende '{id_col}' als Dokument-ID")
    else:
        doc_ids = [f"doc_{i}" for i in range(len(df))]
        print(f"   📍 Keine ID-Spalte gefunden, verwende generierte IDs")

    print("\n➡ Berechne Cosinus-Ähnlichkeit ...")
    df_cos = compute_cosine(df[feature_cols])

    # Spalten und Index beschriften
    df_cos.index = doc_ids
    df_cos.columns = doc_ids

    # Zielordner anlegen
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Speichere Datei: {output_path}")
    df_cos.to_csv(output_path, encoding="utf-8", index=True)

    print(f"✅ Cosinusmatrix erfolgreich erstellt ({len(doc_ids)}×{len(doc_ids)})")


# =============================================================================
# Argumentparser
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Berechnet eine Cosinusmatrix aus TF-IDF/DTM-CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Verwendet gemeinsame pipeline_utils
  - Erweiterte ID-Erkennung (doc_id > _id > id > filename)
  - Konsistent mit Pipeline v3

Beispiele:
  python s04_cosine.py \\
      --input output/dtm_tfidf_stop/tfidf-2000.csv \\
      --output output/cosine/cosine_tfidf2000.csv
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur TF-IDF/DTM-Eingabedatei.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Pfad zur Cosinus-Ausgabedatei.",
    )
    return parser.parse_args(argv)


# =============================================================================
# MAIN – CLI-Wrapper
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
