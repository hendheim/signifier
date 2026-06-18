#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erstellt Vokabulare aus den Preprocessing-Ausgaben.

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Automatische Metadaten-Erkennung
- Dynamische Intervall-Generierung aus den Jahresdaten
- Flexible ID-Spalten-Erkennung
- Konsistent mit Pipeline v3

Input-Dateien:
    korpus_min.csv (mit content_min)
    korpus_lem.csv (mit content_lem)
    korpus_stop.csv (mit content_stop)

Ausgabe:
    vocab_full_<variant>.json
    vocab_textclass_<variant>_<textclass>.json (falls textclass-Spalte existiert)
    vocab_interval_<variant>_<interval>.json (falls year/year_first existiert)
    vocab_genre_<variant>_<genre>.json (falls genre-Spalte existiert)

Beispielaufruf: 

    python s01_vocabulary.py \\
        --input-dir output/processed_corpus \\
        --output-dir output/vocabular \\
        --delimiter auto
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter
from typing import Iterable, Dict, List, Tuple, Optional

import pandas as pd

# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter, read_csv_auto,
        identify_content_column, identify_content_column_strict,
        identify_metadata_columns, identify_id_column,
        identify_year_columns, get_year_series,
        has_column, safe_filename, detect_year_range,
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter, read_csv_auto,
        identify_content_column, identify_content_column_strict,
        identify_metadata_columns, identify_id_column,
        identify_year_columns, get_year_series,
        has_column, safe_filename, detect_year_range,
    )


# ---------------------------------------------------------
# Hilfsfunktion: Tokenisierung & Frequenzen
# ---------------------------------------------------------

def analyze_vocabulary(texts: Iterable[str]) -> Counter:
    """Erstellt ein Frequenzvokabular aus einer Sequenz von Texten.

    Case-sensitiv: Die Groß-/Kleinschreibung der Tokens bleibt erhalten
    (z. B. „Hund" und „hund" werden getrennt gezählt), passend zur übrigen
    fallerhaltenden Verarbeitung.
    """
    all_text = " ".join(t for t in texts if isinstance(t, str))
    tokens = re.findall(r"\b\w+\b", all_text)
    return Counter(tokens)


# ---------------------------------------------------------
# Speichern
# ---------------------------------------------------------

def save_vocab(data: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✔ gespeichert: {out_path}")


# ---------------------------------------------------------
# Dynamische Intervall-Generierung
# ---------------------------------------------------------

def generate_intervals_from_data(
    df: pd.DataFrame,
    custom_intervals: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[str, int, int]]:
    """Liefert Intervalle AUSSCHLIESSLICH aus explizit übergebenen
    ``custom_intervals``.

    Es findet KEINE automatische Ableitung aus den Jahresdaten mehr statt:
    Intervall-Vokabulare entstehen nur, wenn Intervalle (über die
    Streamlit-Oberfläche bzw. die Config) angegeben werden. Ohne
    ``custom_intervals`` wird eine leere Liste zurückgegeben.

    Returns:
        Liste von (label, start_year, end_year) Tupeln (leer ohne Intervalle).
    """
    if custom_intervals:
        return [(f"{s}-{e}", s, e) for s, e in custom_intervals]
    return []


# ---------------------------------------------------------
# Dynamische Genre-Erkennung
# ---------------------------------------------------------

def detect_genres(df: pd.DataFrame) -> List[str]:
    """
    Erkennt alle einzigartigen Genres im DataFrame.
    
    Unterstützt sowohl einzelne Werte als auch kommagetrennte Listen.
    """
    if not has_column(df, "genre"):
        return []
    
    genres = set()
    for val in df["genre"].dropna().astype(str):
        # Kommagetrennte Werte aufteilen
        parts = [p.strip() for p in val.split(",")]
        genres.update(p for p in parts if p)
    
    return sorted(genres)


# ---------------------------------------------------------
# Vokabularerstellung für jede Variante
# ---------------------------------------------------------

def build_vocabularies(
    df: pd.DataFrame, 
    variant: str, 
    output_dir: Path,
    custom_intervals: Optional[List[Tuple[int, int]]] = None,
):
    """
    Erstellt Vokabulare für:
        - Gesamt
        - Textklassen (falls Spalte vorhanden)
        - Zeitintervalle (falls year/year_first vorhanden - dynamisch generiert)
        - Genres (falls genre-Spalte vorhanden - dynamisch erkannt)
    """
    
    # Content-Spalte identifizieren
    content_col = identify_content_column_strict(df)
    print(f"  📋 Content-Spalte: {content_col}")
    
    # Metadaten identifizieren
    metadata_cols = identify_metadata_columns(df, content_col)
    print(f"  📋 Metadaten-Spalten: {len(metadata_cols)}")

    # -----------------------------------------------------
    # 1) Gesamtvokabular
    # -----------------------------------------------------
    print(f"  📄 Erzeuge Gesamtvokabular ({variant}) ...")

    freq = analyze_vocabulary(df[content_col])
    vocab_data = {
        "variant": variant,
        "vocabulary_size": len(freq),
        "total_tokens": sum(freq.values()),
        "top_words": freq.most_common(5000),
        "full_vocab": dict(freq)
    }
    save_vocab(vocab_data, output_dir / f"vocab_full_{variant}.json")

    # -----------------------------------------------------
    # 2) Textklassen (falls vorhanden)
    # -----------------------------------------------------
    if has_column(df, "textclass"):
        print(f"  📄 Erzeuge Vokabulare für Textklassen ({variant}) ...")
        textclasses = sorted(df["textclass"].dropna().unique())
        print(f"      Gefunden: {len(textclasses)} Textklassen")
        
        for tc in textclasses:
            texts = df.loc[df["textclass"] == tc, content_col].astype(str).tolist()
            if not texts:
                continue

            freq = analyze_vocabulary(texts)
            vocab_data = {
                "variant": variant,
                "textclass": tc,
                "document_count": len(texts),
                "vocabulary_size": len(freq),
                "total_tokens": sum(freq.values()),
                "top_words": freq.most_common(5000),
                "full_vocab": dict(freq)
            }
            out = output_dir / "textclass" / f"vocab_textclass_{variant}_{safe_filename(tc)}.json"
            save_vocab(vocab_data, out)
    else:
        print(f"  ⚠️  Keine 'textclass'-Spalte gefunden → Textklassen-Vokabulare übersprungen.")

    # -----------------------------------------------------
    # 3) Zeitintervalle – NUR bei explizit angegebenen Intervallen
    #    (keine automatische Ableitung aus den Jahresdaten mehr)
    # -----------------------------------------------------
    year_first_col, year_col = identify_year_columns(df)

    if custom_intervals and (year_first_col or year_col):
        print(f"  📄 Erzeuge Vokabulare für Zeitintervalle ({variant}) ...")
        
        # Jahr-Serie erstellen
        year_series = get_year_series(df)
        
        # Intervalle generieren
        intervals = generate_intervals_from_data(df, custom_intervals)
        print(f"      Gefunden: {len(intervals)} Intervalle")
        
        for label, start_y, end_y in intervals:
            mask = (year_series >= start_y) & (year_series <= end_y)
            texts = df.loc[mask, content_col].astype(str).tolist()
            
            if not texts:
                continue

            freq = analyze_vocabulary(texts)
            vocab_data = {
                "variant": variant,
                "interval": label,
                "year_range": [start_y, end_y],
                "document_count": len(texts),
                "vocabulary_size": len(freq),
                "total_tokens": sum(freq.values()),
                "top_words": freq.most_common(5000),
                "full_vocab": dict(freq)
            }
            out = output_dir / "intervals" / f"vocab_interval_{variant}_{label}.json"
            save_vocab(vocab_data, out)
    else:
        print(f"  ⚠️  Keine 'year' oder 'year_first'-Spalte gefunden → Intervall-Vokabulare übersprungen.")

    # -----------------------------------------------------
    # 4) Genres (dynamisch erkannt)
    # -----------------------------------------------------
    if has_column(df, "genre"):
        print(f"  📄 Erzeuge Vokabulare für Genres ({variant}) ...")
        
        genres = detect_genres(df)
        print(f"      Gefunden: {len(genres)} Genres")
        
        for genre in genres:
            # Prüfe, ob der Eintrag den Genre-String enthält (kommagetrennte Listen)
            mask = df["genre"].astype(str).str.contains(
                rf"(^|,\s*){re.escape(genre)}($|,)",
                regex=True, na=False
            )
            texts = df.loc[mask, content_col].astype(str).tolist()
            
            if not texts:
                continue

            freq = analyze_vocabulary(texts)
            vocab_data = {
                "variant": variant,
                "genre": genre,
                "document_count": len(texts),
                "vocabulary_size": len(freq),
                "total_tokens": sum(freq.values()),
                "top_words": freq.most_common(5000),
                "full_vocab": dict(freq)
            }
            out = output_dir / "genres" / f"vocab_genre_{variant}_{safe_filename(genre)}.json"
            save_vocab(vocab_data, out)
    else:
        print(f"  ⚠️  Keine 'genre'-Spalte gefunden → Genre-Vokabulare übersprungen.")


# ---------------------------------------------------------
# Argumente
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Erstellt Vokabulare aus Preprocessing-Outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v3:
  - Automatische Delimiter-Erkennung (Fallback: ";")
  - Dynamische Intervall-Generierung aus Jahresdaten
  - Dynamische Genre-Erkennung
  - Flexible Metadaten-/ID-Erkennung

Beispiel:
  python s01_vocabulary.py \\
      --input-dir output/processed_corpus \\
      --output-dir output/vocabular \\
      --delimiter auto
        """
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Ordner mit korpus_min.csv, korpus_lem.csv, korpus_stop.csv",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Ordner zum Speichern der Vokabulardateien (JSON).",
    )
    p.add_argument(
        "--delimiter",
        default="auto",
        help="CSV/TSV-Feldtrenner ('auto' für automatische Erkennung, Standard: auto)",
    )
    p.add_argument(
        "--intervals",
        nargs="+",
        default=None,
        help='Explizite Intervalle, z. B. "1782-1852" "1853-1864". '
             'Nur damit werden Intervall-Vokabulare erzeugt (keine '
             'automatische Ableitung mehr).',
    )
    return p.parse_args(argv)


def parse_interval_strings(interval_strings):
    """Wandelt ["1782-1852", ...] in [(1782, 1852), ...] um.

    Ungültige Einträge werden mit Warnung übersprungen.
    """
    if not interval_strings:
        return None
    result = []
    for s in interval_strings:
        try:
            start, end = str(s).replace("–", "-").split("-")
            result.append((int(start.strip()), int(end.strip())))
        except (ValueError, AttributeError):
            print(f"⚠️  Überspringe ungültiges Intervall: {s!r}")
    return result or None


# ---------------------------------------------------------
# Run-Funktion (für Pipeline / direkten Funktionsaufruf)
# ---------------------------------------------------------

def run(
    input_dir: Path,
    output_dir: Path,
    delimiter: str = "auto",
    custom_intervals: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """
    Erstellt Vokabulare für alle Preprocessing-Varianten.
    
    Args:
        input_dir: Ordner mit korpus_*.csv Dateien
        output_dir: Ausgabeordner für JSON-Dateien
        delimiter: CSV-Delimiter ("auto" für automatische Erkennung)
        custom_intervals: Optional - Liste von (start, end) Tupeln; nur damit
            werden Intervall-Vokabulare erzeugt (keine automatische Ableitung).
    """
    variants = ["min", "lem", "stop"]
    detected_delimiter = None

    for variant in variants:
        infile = input_dir / f"korpus_{variant}.csv"
        if not infile.exists():
            print(f"⚠️  Datei fehlt: {infile} → überspringe.")
            continue

        print(f"\n{'='*70}")
        print(f"VARIANTE: {variant.upper()}")
        print(f"{'='*70}")
        print(f"📄 Lese {infile} ...")
        
        # Delimiter nur einmal erkennen
        if delimiter == "auto" and detected_delimiter is None:
            detected_delimiter = detect_delimiter(infile)
        
        use_delimiter = detected_delimiter if delimiter == "auto" else delimiter
        df = pd.read_csv(infile, sep=use_delimiter, encoding="utf-8")
        
        print(f"  📊 {len(df)} Dokumente geladen")

        build_vocabularies(df, variant, output_dir, custom_intervals)

    print("\n" + "="*70)
    print("✅ Fertig. Alle Vokabulare erstellt.")
    print("="*70)


# ---------------------------------------------------------
# Main (CLI-Wrapper)
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
        custom_intervals=parse_interval_strings(args.intervals),
    )


if __name__ == "__main__":
    main()
