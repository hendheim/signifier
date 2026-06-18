#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.data_store
========================

UI-freie Daten- und Modellverwaltung.

Extrahiert aus den beiden Legacy-GUIs (``DataManager``/``ModelManager`` in
``gui_corpus_explorer.py`` und ``DataManager``/``FileDiscovery`` in
``gui_tag_topic_explorer.py``). Die Klassen hier kennen KEIN Tkinter und
KEIN Streamlit – sie laden, cachen und bereiten Daten gemäß dem
Metadatenschema (``explorer_core.schema``) auf. Dadurch können sie sowohl
vom Dashboard als auch in Notebooks/Skripten verwendet werden (didaktisch
praktisch für Seminare).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .schema import MetadataSchema, find_column


# ----------------------------------------------------------------------------
# CSV-Einlesen mit automatischer Trennzeichen-Erkennung
# (zusammengeführt aus pipeline_utils.detect_delimiter und den GUI-Varianten)
# ----------------------------------------------------------------------------

def detect_delimiter(file_path: Path, sample_lines: int = 25) -> str:
    """Erkennt das Trennzeichen einer CSV-Datei (";", ",", Tab oder "|").

    Konsistenzbasierte Heuristik – identisch zu ``pipeline_utils.detect_delimiter``,
    aber hier selbständig gehalten (keine paketübergreifenden Importe). Der
    Sample-Block wird als Ganzes mit ``csv.reader`` geparst, damit gequotete
    Felder mit eingebetteten Trennzeichen/Zeilenumbrüchen korrekt
    zusammengehalten werden. Bewertet wird zuerst die Konsistenz der
    Spaltenzahl (inkl. Kopfzeile), dann die Spaltenzahl. So gewinnt bei
    semikolongetrennten Korpora mit kommareichem Content nicht mehr ",".
    """
    import io
    from collections import Counter

    candidates = [";", ",", "\t", "|"]
    fallback = ";"
    file_path = Path(file_path)
    if not file_path.exists():
        return fallback
    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(131072)
    except Exception:
        return fallback
    if not sample.strip():
        return fallback

    best, best_key = None, None
    for d in candidates:
        try:
            rows = [r for r in csv.reader(io.StringIO(sample), delimiter=d) if r]
        except csv.Error:
            continue
        rows = rows[:sample_lines]
        if len(rows) < 2:
            continue
        counts = [len(r) for r in rows]
        modal = Counter(counts).most_common(1)[0][0]
        if modal < 2:
            continue
        consistency = sum(c == modal for c in counts) / len(counts)
        key = (round(consistency, 4), modal)
        if best_key is None or key > best_key:
            best_key, best = key, d
    if best is not None:
        return best
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters="".join(candidates)).delimiter
        if sniffed in candidates:
            return sniffed
    except csv.Error:
        pass
    return fallback


def read_csv_auto(path: Path, delimiter: str = "auto", **kwargs) -> pd.DataFrame:
    """Liest eine CSV-Datei; ``delimiter='auto'`` erkennt das Trennzeichen."""
    path = Path(path)
    sep = detect_delimiter(path) if delimiter in (None, "auto") else delimiter
    return pd.read_csv(path, sep=sep, encoding="utf-8", **kwargs)


# ----------------------------------------------------------------------------
# Standard-Pfade relativ zum Projektordner
# (übernommen aus den Defaults beider GUIs – über die Daten-Seite änderbar)
# ----------------------------------------------------------------------------

DEFAULT_PATHS: Dict[str, str] = {
    # Korpus-Explorer
    "corpus": "output/processed_corpus/korpus_stop.csv",
    "dtm": "output/dtm_tfidf_stop/dtm_minfreq6.csv",
    "tfidf": "output/dtm_tfidf_stop/tfidf-2000.csv",
    "metadata": "korpus/metadaten.csv",
    "cosine": "output/cosine/cosine_tfidf2000.csv",
    "topics_dist": "resources/topic-models/sklearn_lda_40/document-topics-distribution_sklearn_lda_40_pro_text.csv",
    "w2v_model": "output/word2vec_models/korpus_gen.model",
    "termset": "resources/termsets/Termset_Test.csv",
    # Tag-Topic-Explorer
    "topic_words": "resources/topic-models/sklearn_lda_40/sklearn_lda_40_topic_words.csv",
    "ranks": "output/processed_termset/Termset_Test/Termset_Test_tag_topic_rank.csv",
    "relevance": "output/processed_termset/Termset_Test/Termset_Test_tag_topic_relevance.csv",
    "counts_per_year": "output/processed_termset/Termset_Test/Termset_Test_dtti_topdocs_topic_counts_per_year.csv",
    "top10_year_value": "output/processed_topics/document-topics-distribution_sklearn_lda_40_pro_text_topdocs_year_value.csv",
    "top10_value_per_text": "output/processed_topics/document-topics-distribution_sklearn_lda_40_pro_text_topdocs_value_per_text_topic.csv",
    "tokens_year": "output/statistics/year_count_tokens.csv",
    "global_topdocs": "output/processed_topics/document-topics-distribution_sklearn_lda_40_pro_text_topdocs_year_value.csv",
}

# Beschriftungen für die Daten-Seite des Dashboards
PATH_LABELS: Dict[str, str] = {
    "corpus": "Korpus (verarbeitet, z. B. korpus_stop.csv)",
    "dtm": "Document-Term-Matrix (DTM)",
    "tfidf": "TF-IDF-Matrix",
    "metadata": "Metadaten der Texte",
    "cosine": "Kosinus-Matrix der Texte",
    "topics_dist": "Document-Topic-Verteilung",
    "w2v_model": "Word2Vec-Modell",
    "termset": "Termset (Pivot-Tabelle)",
    "topic_words": "Top-Topic-Words-Matrix",
    "ranks": "Rangliste der Topics (relativ zum Termset)",
    "relevance": "Relevanzscore der Topics (relativ zum Termset)",
    "counts_per_year": "Topic-Counts pro Jahr (Termset, TopDocs)",
    "top10_year_value": "Summierter Relevanzscore pro Jahr (Termset)",
    "top10_value_per_text": "Summierter Relevanzscore pro Text (Termset)",
    "tokens_year": "Tokenverteilung pro Jahr",
    "global_topdocs": "Text-Topic-Relevanzscore pro Jahr (global)",
}


# Glob-Muster zur relativen Auflösung der Datenpfade (greift, wenn die konkrete
# Datei aus DEFAULT_PATHS fehlt). Die Termset-Ergebnisse sind dadurch nicht an
# einen festen Termset-Ordner gebunden: 'Termset*' findet jeden Termset-
# Unterordner, und der Dateiname muss mindestens die charakteristische
# Zeichenfolge (z. B. '_tag_topic_rank') enthalten.
DISCOVERY_PATTERNS: Dict[str, List[str]] = {
    "corpus": ["output/processed_corpus/*.csv"],
    "dtm": ["output/dtm_tfidf*/dtm*.csv"],
    "tfidf": ["output/dtm_tfidf*/tfidf*.csv"],
    "metadata": ["korpus/metadaten.csv"],
    "cosine": ["output/cosine/*.csv"],
    "topics_dist": ["resources/topic-models/topics*/document-topics-distribution*.csv"],
    "w2v_model": ["output/word2vec_models/*.model", "output/word2vec_models/*.kv"],
    "termset": ["resources/termsets/*.csv"],
    "topic_words": ["resources/topic-models/topics*/*words*tag*.csv"],
    "ranks": ["output/processed_termset/Termset*/*_tag_topic_rank.csv"],
    "relevance": ["output/processed_termset/Termset*/*_tag_topic_relevance.csv"],
    "counts_per_year": ["output/processed_termset/Termset*/*_dtti_topdocs_topic_counts_per_year.csv"],
    "top10_year_value": ["output/processed_termset/Termset*/*_dtti_topdocs_top10_year_value.csv"],
    "top10_value_per_text": ["output/processed_termset/Termset*/*_dtti_topdocs_top10_value_per_text_topic.csv"],
    "tokens_year": ["output/statistics/*tokens*.csv"],
    "global_topdocs": ["output/processed_topics/*year*value*.csv"],
}


def detect_project_root(start: Optional[Path] = None) -> Path:
    """Sucht ein Verzeichnis mit ``output/`` und ``resources/`` (wie die GUIs)."""
    candidate = Path(start) if start else Path.cwd()
    for path in [candidate, *candidate.parents]:
        if (path / "output").exists() and (path / "resources").exists():
            return path
    return candidate


class DataStore:
    """Lädt und cached alle Datenquellen, aufbereitet gemäß Metadatenschema."""

    def __init__(self, project_root: Path, schema: Optional[MetadataSchema] = None):
        self.project_root = Path(project_root)
        self.schema = schema or MetadataSchema()
        self.paths: Dict[str, Path] = {
            key: self.project_root / rel for key, rel in DEFAULT_PATHS.items()
        }
        self._cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Verwaltung
    # ------------------------------------------------------------------

    def set_project_root(self, root: Path) -> None:
        """Setzt einen neuen Projektordner und leitet alle Pfade neu ab."""
        self.project_root = Path(root)
        self.paths = {key: self.project_root / rel for key, rel in DEFAULT_PATHS.items()}
        self._cache.clear()

    def set_path(self, key: str, path) -> None:
        """Setzt einen einzelnen Datenpfad und invalidiert dessen Cache."""
        self.paths[key] = Path(path)
        self._cache.pop(key, None)
        if key == "metadata":
            # Metadaten bestimmen das Term-Mapping → alles invalidieren
            self._cache.clear()

    def invalidate(self) -> None:
        self._cache.clear()

    def status(self) -> Dict[str, bool]:
        """Existenz-Check aller Pfade (für die Daten-Seite)."""
        return {key: Path(p).exists() for key, p in self.paths.items()}

    def auto_discover(self) -> Dict[str, Optional[Path]]:
        """Sucht fehlende Dateien per Muster im Projektordner.

        Übernimmt die Discovery-Heuristiken der ``FileDiscovery``-Klasse des
        Tag-Topic-Explorers, ohne festes Termset-Suffix: Es wird der erste
        Treffer pro Muster verwendet.
        """
        root = self.project_root
        patterns = DISCOVERY_PATTERNS
        found: Dict[str, Optional[Path]] = {}
        for key, globs in patterns.items():
            hit = None
            for pattern in globs:
                matches = sorted(root.glob(pattern))
                if matches:
                    hit = matches[0]
                    break
            found[key] = hit
            if hit is not None:
                self.set_path(key, hit)
        return found

    # ------------------------------------------------------------------
    # Loader (alle geben pandas-DataFrames zurück)
    # ------------------------------------------------------------------

    def _resolve_glob(self, key: str) -> Optional[Path]:
        """Findet die Datei relativ über DISCOVERY_PATTERNS (erster Treffer)."""
        for pattern in DISCOVERY_PATTERNS.get(key, []):
            matches = sorted(self.project_root.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _read(self, key: str, **kwargs) -> pd.DataFrame:
        path = self.paths[key]
        if not Path(path).exists():
            alt = self._resolve_glob(key)
            if alt is not None:
                path = alt
                self.paths[key] = alt  # aufgelösten Pfad merken
            else:
                raise FileNotFoundError(
                    f"{PATH_LABELS.get(key, key)} nicht gefunden: {path}")
        return read_csv_auto(path, delimiter=self.schema.delimiter, **kwargs)

    def load_metadata(self) -> pd.DataFrame:
        """Metadaten laden + ID/Jahr normalisieren + im Schema registrieren."""
        if "metadata" in self._cache:
            return self._cache["metadata"]  # type: ignore[return-value]
        df = self._read("metadata")
        df = self.schema.ensure_doc_id(df)
        df = self.schema.coalesce_years(df)
        self.schema.register_metadata(df)
        self._cache["metadata"] = df
        return df

    def _ensure_metadata_registered(self) -> None:
        """DTM/TF-IDF-Mapping braucht die Metadaten-Spalten – falls möglich laden."""
        if not self.schema.metadata_registered:
            try:
                self.load_metadata()
            except FileNotFoundError:
                pass  # Schema fällt dann auf Heuristiken zurück

    def load_corpus(self) -> pd.DataFrame:
        if "corpus" in self._cache:
            return self._cache["corpus"]  # type: ignore[return-value]
        df = self._read("corpus")
        content_col = self.schema.find_content_column(df)
        df["text"] = df[content_col].fillna("").astype(str) if content_col else ""
        df = self.schema.ensure_doc_id(df)
        df = self.schema.coalesce_years(df)
        self._cache["corpus"] = df
        return df

    def load_dtm(self) -> pd.DataFrame:
        if "dtm" in self._cache:
            return self._cache["dtm"]  # type: ignore[return-value]
        self._ensure_metadata_registered()
        df = self.schema.coalesce_years(self._read("dtm"))
        self._cache["dtm"] = df
        return df

    def load_tfidf(self) -> pd.DataFrame:
        if "tfidf" in self._cache:
            return self._cache["tfidf"]  # type: ignore[return-value]
        self._ensure_metadata_registered()
        df = self._read("tfidf")
        self._cache["tfidf"] = df
        return df

    def load_cosine(self) -> pd.DataFrame:
        if "cosine" in self._cache:
            return self._cache["cosine"]  # type: ignore[return-value]
        df = self._read("cosine", index_col=0)
        self._cache["cosine"] = df
        return df

    def load_topics_dist(self) -> pd.DataFrame:
        if "topics_dist" in self._cache:
            return self._cache["topics_dist"]  # type: ignore[return-value]
        df = self._read("topics_dist", index_col=0)
        df.index = df.index.astype(str).str.replace(".txt", "", regex=False)
        self._cache["topics_dist"] = df
        return df

    def load_termset(self) -> pd.DataFrame:
        """Termset (Groß-/Kleinschreibung erhalten, nur getrimmt).

        Case-sensitiv passend zum übrigen Korpus: Die Einträge müssen in der
        korrekten Schreibweise vorliegen (Nomen groß), damit sie gegen das
        nun case-sensitive Word2Vec-Vokabular und die TF-IDF-Terme treffen.
        """
        if "termset" in self._cache:
            return self._cache["termset"]  # type: ignore[return-value]
        df = self._read("termset")
        df = df.apply(lambda col: col.map(
            lambda x: str(x).strip() if pd.notna(x) else x))
        self._cache["termset"] = df
        return df

    def load_topic_words(self) -> pd.DataFrame:
        if "topic_words" in self._cache:
            return self._cache["topic_words"]  # type: ignore[return-value]
        df = self._read("topic_words", index_col=0)
        df = df.apply(lambda col: col.map(
            lambda x: str(x).strip().lower() if pd.notna(x) else x))
        self._cache["topic_words"] = df
        return df

    def load_ranks(self) -> pd.DataFrame:
        if "ranks" not in self._cache:
            self._cache["ranks"] = self._read("ranks")
        return self._cache["ranks"]  # type: ignore[return-value]

    def load_relevance(self) -> pd.DataFrame:
        if "relevance" not in self._cache:
            self._cache["relevance"] = self._read("relevance")
        return self._cache["relevance"]  # type: ignore[return-value]

    def load_counts_per_year(self) -> pd.DataFrame:
        if "counts_per_year" not in self._cache:
            self._cache["counts_per_year"] = self._read("counts_per_year", index_col=0)
        return self._cache["counts_per_year"]  # type: ignore[return-value]

    def load_top10_year_value(self) -> pd.DataFrame:
        if "top10_year_value" not in self._cache:
            self._cache["top10_year_value"] = self._read("top10_year_value")
        return self._cache["top10_year_value"]  # type: ignore[return-value]

    def load_top10_value_per_text(self) -> pd.DataFrame:
        if "top10_value_per_text" not in self._cache:
            self._cache["top10_value_per_text"] = self._read("top10_value_per_text")
        return self._cache["top10_value_per_text"]  # type: ignore[return-value]

    def load_tokens_year(self) -> pd.DataFrame:
        if "tokens_year" not in self._cache:
            self._cache["tokens_year"] = self._read("tokens_year")
        return self._cache["tokens_year"]  # type: ignore[return-value]

    def load_global_topdocs(self) -> pd.DataFrame:
        if "global_topdocs" not in self._cache:
            self._cache["global_topdocs"] = self._read("global_topdocs")
        return self._cache["global_topdocs"]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Abgeleitete Daten
    # ------------------------------------------------------------------

    def term_columns(self, df: pd.DataFrame):
        """Bequemer Zugriff auf die Term-Spalten gemäß Schema."""
        self._ensure_metadata_registered()
        return self.schema.term_columns_in(df)

    def tfidf_averages(self) -> pd.DataFrame:
        """Durchschnittliche TF-IDF-Werte pro Term (für den TF-IDF-Rang)."""
        df = self.load_tfidf()
        term_cols = self.term_columns(df)
        if not term_cols:
            raise ValueError("Keine Term-Spalten in der TF-IDF-Datei gefunden.")
        avg = df[term_cols].mean().sort_values(ascending=False)
        return pd.DataFrame({
            "rank": range(1, len(avg) + 1),
            "term": avg.index,
            "tfidf_avg": avg.values,
        })

    def tfidf_sums(self) -> pd.Series:
        """TF-IDF-Summen pro Term (für das Bubble-Chart, case-sensitiver Index)."""
        df = self.load_tfidf()
        term_cols = self.term_columns(df)
        if not term_cols:
            raise ValueError("Keine Term-Spalten in der TF-IDF-Datei gefunden.")
        numeric = df[term_cols].apply(pd.to_numeric, errors="coerce")
        sums = numeric.sum(skipna=True)
        sums.index = sums.index.astype(str).str.strip()
        return sums


class ModelStore:
    """Lädt das Word2Vec-Modell (gensim) – getrennt vom DataStore, weil
    Modelle groß sind und nur für die Vektor-Seiten gebraucht werden."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._kv = None

    def set_path(self, path) -> None:
        self.model_path = Path(path)
        self._kv = None

    def load(self):
        """Gibt die KeyedVectors zurück (unterstützt .model/.kv/.bin wie bisher)."""
        if self._kv is not None:
            return self._kv
        from gensim.models import KeyedVectors, Word2Vec  # lazy import

        path = self.model_path
        if not path.exists():
            raise FileNotFoundError(f"Word2Vec-Modell nicht gefunden: {path}")
        if path.suffix in {".wordvectors", ".kv"}:
            self._kv = KeyedVectors.load(str(path))
        elif path.suffix == ".model":
            self._kv = Word2Vec.load(str(path)).wv
        else:
            binary = path.suffix.lower() in {".bin", ".gz"}
            self._kv = KeyedVectors.load_word2vec_format(str(path), binary=binary)
        return self._kv


__all__ = [
    "DataStore", "ModelStore", "DEFAULT_PATHS", "PATH_LABELS",
    "read_csv_auto", "detect_delimiter", "detect_project_root", "find_column",
]
