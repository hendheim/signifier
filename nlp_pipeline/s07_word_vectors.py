#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainiert Word2Vec-Modelle auf Basis eines oder mehrerer vorverarbeiteter
Korpora 'korpus_gen*.csv' (Gensim-Preprocessing, stopwortbereinigt, lemmatisiert).

ÄNDERUNG v3:
- Automatische Delimiter-Erkennung (Fallback: ";")
- Verwendet gemeinsame pipeline_utils
- Portable Modelle: Speichert nur KeyedVectors
- Adaptive Parameter basierend auf Korpusgröße
- Konsistent mit Pipeline v3

Beispielaufruf:

    python s07_word_vectors.py \\
        --input-dir output/processed_corpus \\
        --pattern "korpus_gen*.csv" \\
        --output-dir output/word2vec_models \\
        --delimiter auto
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import pandas as pd
from gensim.models import Word2Vec


def _ensure_nltk_punkt():
    """Stellt sicher, dass die NLTK-Tokenizer verfügbar sind.

    Wichtig: NLTK >= 3.9 benötigt 'punkt_tab' statt des alten 'punkt'.
    Damit `sent_tokenize`/`word_tokenize` zuverlässig funktionieren (und nicht
    still auf einen schlechteren Fallback zurückfallen), werden beide
    Ressourcen geprüft und bei Bedarf geladen.
    """
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


# Import der gemeinsamen Utils
try:
    from .pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_id_column,
        identify_year_columns
    )
except ImportError:
    from pipeline_utils import (
        detect_delimiter,
        identify_content_column,
        identify_metadata_columns,
        identify_id_column,
        identify_year_columns
    )

try:
    import gensim
    import numpy as np
    GENSIM_VERSION = gensim.__version__
    NUMPY_VERSION = np.__version__
except ImportError:
    GENSIM_VERSION = "unknown"
    NUMPY_VERSION = "unknown"


# =============================================================================
# Adaptive Parameter-Berechnung
# =============================================================================

def calculate_word2vec_params(num_tokens: int) -> Dict[str, int]:
    """Voreingestellte Word2Vec-Parameter je nach Korpusgröße (Token-Anzahl).

    Quellen / Begründung (Stand 2024/25):
    - gensim-Defaults: vector_size=100, window=5, min_count=5, sample=1e-3,
      sg=0, hs=0, negative=5, epochs=5 (gensim-Doku).
    - Diachrone Wortvektoren (Hamilton et al. 2016, HistWords/COHA) verwenden
      durchgängig Skip-gram mit Negative Sampling (SGNS), Fenster ~5,
      Dimension 100–300 je nach Korpusgröße. Skip-gram ist für kleine Korpora
      und seltene Wörter überlegen, CBOW eher für große Korpora/häufige Wörter.
    - Stabilitätsstudien (z. B. arXiv:2007.16006) zeigen: bei kleinen Korpora
      verbessern MEHR Epochen und MEHR Negative-Samples Qualität und Stabilität.
    Da das Korpus historisch ist und in (teils kleine) Jahresintervalle
    zerlegt wird, ist Skip-gram (sg=1) der robuste Standard; Dimension und
    Frequenzschwelle skalieren mit der Korpusgröße, kleine Korpora bekommen
    mehr Epochen/Negatives.
    """
    base_params = {
        'workers': 4,
        'sg': 1,          # Skip-gram (SGNS) – Standard für diachrone/kleine Korpora
        'hs': 0,          # Negative Sampling statt hierarchical softmax
        'sample': 1e-3,   # Subsampling häufiger Wörter (gensim-Default)
        'seed': 42,
    }

    # Schwellen nach Altszyler et al. (2017): Word2Vec entfaltet seine Stärke
    # erst ab ~10 Mio Tokens; darunter gilt ein Korpus als klein (frühere
    # Schwellen 100k/1 Mio stuften kleine historische Korpora fälschlich als
    # "mittel/groß" ein).
    if num_tokens < 1_000_000:
        size_params = {'vector_size': 100, 'window': 5, 'min_count': 2, 'negative': 10, 'epochs': 40}
        category = "KLEIN"
    elif num_tokens < 10_000_000:
        size_params = {'vector_size': 150, 'window': 5, 'min_count': 5, 'negative': 5, 'epochs': 20}
        category = "MITTEL"
    else:
        size_params = {'vector_size': 300, 'window': 5, 'min_count': 10, 'negative': 5, 'epochs': 15}
        category = "GROSS"

    all_params = {**base_params, **size_params, '_category': category}
    return all_params


# Hyperparameter, die in der Streamlit-Pipeline überschreibbar sind.
OVERRIDABLE_W2V_KEYS = (
    "vector_size", "window", "min_count", "negative", "epochs", "sg", "sample", "hs",
)


def merge_word2vec_params(num_tokens: int,
                          overrides: Optional[Dict] = None) -> Dict:
    """Adaptive Defaults berechnen und mit Nutzer-Overrides kombinieren.

    ``overrides`` (z. B. aus dem Dashboard) überschreiben gezielt einzelne
    Hyperparameter; alles andere bleibt auf dem größenabhängigen Default.
    """
    params = calculate_word2vec_params(num_tokens)
    if overrides:
        for key, value in overrides.items():
            if key in OVERRIDABLE_W2V_KEYS and value is not None:
                params[key] = value
        params['_category'] = params.get('_category', 'UNKNOWN') + " (angepasst)"
    return params


# =============================================================================
# Tokenisierung
# =============================================================================

def tokenize_corpus(texts: pd.Series) -> List[List[str]]:
    """Zerlegt Texte in Sätze und Tokens."""
    _ensure_nltk_punkt()  # Stelle sicher, dass NLTK-Ressourcen verfügbar sind
    sentences = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            sents = nltk.sent_tokenize(text, language='german')
            for sent in sents:
                tokens = nltk.word_tokenize(sent, language='german')
                tokens_clean = [t for t in tokens if t.isalpha()]
                if tokens_clean:
                    sentences.append(tokens_clean)
        except Exception:
            tokens = text.split()
            tokens_clean = [t for t in tokens if t.isalpha()]
            if tokens_clean:
                sentences.append(tokens_clean)
    return sentences


def estimate_token_count(texts: pd.Series) -> int:
    """Schätzt die Token-Anzahl im Korpus (Stichprobe, schnell)."""
    sample_size = min(100, len(texts))
    sample = texts.sample(n=sample_size, random_state=42) if len(texts) > sample_size else texts
    avg_tokens = sample.astype(str).apply(lambda x: len(x.split())).mean()
    return int(avg_tokens * len(texts))


def count_tokens_in_file(file_path: Path, delimiter: str = "auto") -> Tuple[int, int]:
    """Zählt Tokens und Dokumente einer korpus_gen-Datei (für die Voreinstellung).

    Wird vom Dashboard genutzt, um anhand der tatsächlichen Korpusgröße
    sinnvolle Word2Vec-Hyperparameter vorzuschlagen. Gibt ``(tokens, docs)``
    zurück; ``(0, 0)`` falls die Datei fehlt/leer ist oder keine Content-Spalte
    gefunden wird.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0, 0
    if delimiter in (None, "auto"):
        delimiter = detect_delimiter(file_path)
    try:
        df = pd.read_csv(file_path, encoding="utf-8", sep=delimiter)
    except Exception:
        return 0, 0
    content_col = identify_content_column(df)
    if content_col is None:
        return 0, 0
    texts = df[content_col].astype(str)
    tokens = int(texts.apply(lambda x: len(x.split())).sum())
    return tokens, len(df)


def train_word2vec(sentences: List[List[str]], params: Dict) -> Word2Vec:
    """Trainiert ein Word2Vec-Modell."""
    clean_params = {k: v for k, v in params.items() if not k.startswith('_')}
    return Word2Vec(sentences, **clean_params)


# =============================================================================
# Modell speichern
# =============================================================================

def save_model_portable(model: Word2Vec, base_path: Path, params: Dict, corpus_metadata: Dict) -> None:
    """Speichert das Modell portabel."""
    wv_path = Path(str(base_path) + ".wordvectors")
    model.wv.save(str(wv_path))
    print(f"   ✔ Word-Vektoren: {wv_path.name}")
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'gensim_version': GENSIM_VERSION,
        'numpy_version': NUMPY_VERSION,
        'parameters': {k: v for k, v in params.items() if not k.startswith('_')},
        'category': params.get('_category', 'UNKNOWN'),
        'vocabulary_size': len(model.wv),
        **corpus_metadata
    }
    
    json_path = Path(str(base_path) + ".json")
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✔ Metadaten: {json_path.name}")
    
    model_path = Path(str(base_path) + ".model")
    model.save(str(model_path))
    print(f"   ✔ Vollmodell: {model_path.name}")


# =============================================================================
# Datei-Handling
# =============================================================================

def find_input_files(input_dir: Path, pattern: str) -> List[Path]:
    """Findet alle Dateien, die dem Pattern entsprechen."""
    return sorted(list(input_dir.glob(pattern)))


def categorize_files(files: List[Path]) -> Tuple[Optional[Path], List[Path]]:
    """Kategorisiert Dateien in Gesamtkorpus und Intervalle."""
    gesamtkorpus = None
    intervals = []
    interval_pattern = re.compile(r'_\d{4}-\d{4}')
    
    for f in files:
        if interval_pattern.search(f.stem):
            intervals.append(f)
        else:
            gesamtkorpus = f
    
    return gesamtkorpus, sorted(intervals)


def read_corpus_gen(input_file: Path,
                    delimiter: str = "auto") -> Tuple[Optional["pd.DataFrame"], Optional[str], Optional[str]]:
    """Liest eine korpus_gen-Datei robust ein und bestimmt ihr Trennzeichen.

    Hintergrund: Das an Schritt 10 durchgereichte Trennzeichen stammt aus dem
    *Rohkorpus* (Schritt 1) und kann von dem abweichen, mit dem Schritt 5 die
    ``korpus_gen``-Dateien tatsächlich geschrieben hat. Deshalb wird das
    Trennzeichen hier aus der Datei selbst bestimmt und die Wahl dadurch
    abgesichert, dass eine Content-Spalte gefunden werden MUSS.

    Reihenfolge der Versuche:
      1. automatisch erkanntes Trennzeichen (``detect_delimiter``),
      2. das explizit übergebene Trennzeichen (falls konkret),
      3. alle gängigen Kandidaten (";", ",", Tab, "|").
    Das erste Trennzeichen, mit dem mindestens zwei Spalten UND eine
    Content-Spalte entstehen, gewinnt. Dadurch ist das Einlesen unabhängig
    davon, welches Trennzeichen ein vorheriger Schritt durchgereicht hat – und
    auch unabhängig davon, welche ``pipeline_utils``-Version importiert wurde.

    Returns:
        (DataFrame, verwendetes_Trennzeichen, Content-Spalte) oder
        (None, None, None), wenn keine Variante eine Content-Spalte liefert.
    """
    candidates: List[str] = []
    try:
        detected = detect_delimiter(input_file)
        if detected:
            candidates.append(detected)
    except Exception:
        pass
    if delimiter not in (None, "auto") and delimiter not in candidates:
        candidates.append(delimiter)
    for fallback in (";", ",", "\t", "|"):
        if fallback not in candidates:
            candidates.append(fallback)

    for sep in candidates:
        try:
            df = pd.read_csv(input_file, encoding="utf-8", sep=sep)
        except Exception:
            continue
        if df.shape[1] < 2:
            continue  # falsches Trennzeichen -> alles in einer Spalte
        content_col = identify_content_column(df)
        if content_col is not None:
            return df, sep, content_col
    return None, None, None


def process_file(input_file: Path, output_dir: Path, delimiter: str = "auto",
                 file_type: str = "unknown",
                 hyperparams: Optional[Dict] = None) -> bool:
    """Verarbeitet eine CSV-Datei und trainiert ein Word2Vec-Modell.

    ``hyperparams`` (optional) überschreibt einzelne, größenabhängig
    voreingestellte Word2Vec-Parameter (aus dem Dashboard).

    Returns:
        True, wenn ein Modell gespeichert wurde, sonst False.
    """
    print(f"\n📄 Verarbeite Datei: {input_file}")
    print(f"   🏷️  Typ: {file_type}")

    if not input_file.exists():
        print(f"⚠️ Datei existiert nicht, übersprungen.")
        return False

    # korpus_gen ist ein Pipeline-internes, selbstbeschreibendes Artefakt: sein
    # Trennzeichen kann sich vom Rohkorpus unterscheiden (Schritt 5 schreibt es).
    # Deshalb wird hier NICHT einem von außen durchgereichten Trennzeichen
    # vertraut, sondern das passende aus der Datei selbst bestimmt und durch
    # Prüfung auf eine erkennbare Content-Spalte abgesichert.
    df, used_sep, content_col = read_corpus_gen(input_file, delimiter)
    if df is None:
        print(f"⚠️ {input_file.name} ließ sich mit keinem Trennzeichen sinnvoll "
              f"einlesen (keine Content-Spalte gefunden), übersprungen.")
        return False
    delimiter = used_sep
    print(f"   🔣 Trennzeichen: {used_sep!r}")
    print(f"   📝 Content-Spalte: {content_col}")

    texts = df[content_col].astype(str)
    if texts.str.strip().eq("").all():
        print(f"⚠️ Alle Einträge leer, übersprungen.")
        return False

    estimated_tokens = estimate_token_count(texts)
    print(f"   📊 Geschätzte Token-Anzahl: ~{estimated_tokens:,}")

    params = merge_word2vec_params(estimated_tokens, hyperparams)
    category = params.get('_category', 'UNKNOWN')
    print(f"   ⚙️  Kategorie: {category}")
    if hyperparams:
        gezeigt = {k: params[k] for k in OVERRIDABLE_W2V_KEYS if k in params}
        print(f"   🎛️  Hyperparameter: {gezeigt}")

    print("   ➜ Tokenisiere Korpus ...")
    sentences = tokenize_corpus(texts)

    if not sentences:
        print(f"⚠️ Keine Sätze nach Tokenisierung, übersprungen.")
        return False

    print(f"   ➜ Trainiere Word2Vec-Modell ({len(sentences):,} Sätze) ...")
    try:
        model = train_word2vec(sentences, params)
    except Exception as e:
        print(f"❌ Fehler beim Training: {e}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    model_base_path = output_dir / input_file.stem
    
    corpus_metadata = {
        'source_file': input_file.name,
        'file_type': file_type,
        'content_column': content_col,
        'num_documents': len(df),
        'num_sentences': len(sentences),
        'estimated_tokens': estimated_tokens,
    }

    save_model_portable(model, model_base_path, params, corpus_metadata)
    print(f"   📈 Vokabulargröße: {len(model.wv):,} Wörter")
    return True


# =============================================================================
# run-Funktion
# =============================================================================

def run(input_dir: Path, output_dir: Path, pattern: str = "korpus_gen*.csv",
        delimiter: str = "auto", hyperparams: Optional[Dict] = None) -> None:
    """Trainiert Word2Vec-Modelle.

    ``hyperparams`` (optional) überschreibt die größenabhängig
    voreingestellten Word2Vec-Hyperparameter für Gesamtkorpus und alle
    Intervalle gleichermaßen (aus dem Dashboard durchgereicht).
    """
    print(f"📚 Eingabeordner: {input_dir}")
    print(f"📚 Ausgabeordner: {output_dir}")
    print(f"🔽 Dateipattern:  {pattern}")
    if hyperparams:
        print(f"🎛️  Manuelle Hyperparameter: {hyperparams}")

    input_files = find_input_files(input_dir, pattern)
    if not input_files:
        print("⚠️ Keine passenden Eingabedateien gefunden.")
        return

    print(f"✔ {len(input_files)} Datei(en) gefunden.")

    gesamtkorpus, intervals = categorize_files(input_files)
    saved = 0

    if gesamtkorpus:
        print("\n" + "="*60)
        print("🌐 GESAMTKORPUS")
        print("="*60)
        if process_file(gesamtkorpus, output_dir, delimiter,
                        file_type="GESAMTKORPUS", hyperparams=hyperparams):
            saved += 1

    if intervals:
        print("\n" + "="*60)
        print(f"📅 INTERVALLE ({len(intervals)} Dateien)")
        print("="*60)
        for interval_file in intervals:
            if process_file(interval_file, output_dir, delimiter,
                           file_type="INTERVALL", hyperparams=hyperparams):
                saved += 1

    print("\n" + "="*60)
    if saved == 0:
        print("⚠️ Es wurde KEIN Word2Vec-Modell gespeichert.")
        print("   Mögliche Ursachen:")
        print("   - Eingabedatei(en) ohne erkennbare Content-Spalte oder leer")
        print("     (Schritt 5 muss 'content_gen' erzeugt haben).")
        print("   - korpus_gen*.csv fehlt (Schritt 5 vorher ausführen).")
        print("   - Tokenisierung lieferte keine Sätze.")
    else:
        print(f"✅ {saved} Modell(e) gespeichert in: {output_dir}")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trainiert Word2Vec-Modelle.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Ordner mit Eingabe-CSV-Dateien.")
    parser.add_argument("--pattern", default="korpus_gen*.csv", help="Dateinamen-Pattern.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Zielordner für Modelle.")
    parser.add_argument("--delimiter", default="auto", help="CSV-Delimiter ('auto' für automatische Erkennung).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(input_dir=args.input_dir, output_dir=args.output_dir, pattern=args.pattern, delimiter=args.delimiter)


if __name__ == "__main__":
    main()
