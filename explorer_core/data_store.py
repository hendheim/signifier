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
    "ranks": "output/processed_termset/Termset_Test/sklearn_lda_40/Termset_Test_tag_topic_rank.csv",
    "relevance": "output/processed_termset/Termset_Test/sklearn_lda_40/Termset_Test_tag_topic_relevance.csv",
    "counts_per_year": "output/processed_termset/Termset_Test/sklearn_lda_40/Termset_Test_dtti_topdocs_topic_counts_per_year.csv",
    "top10_year_value": "output/processed_topics/sklearn_lda_40/document-topics-distribution_sklearn_lda_40_pro_text_topdocs_year_value.csv",
    "top10_value_per_text": "output/processed_topics/sklearn_lda_40/document-topics-distribution_sklearn_lda_40_pro_text_topdocs_value_per_text_topic.csv",
    "tokens_year": "output/statistics/year_count_tokens.csv",
}

# Beschriftungen für die Daten-Seite des Dashboards (Reihenfolge folgt den
# Kategorien in PATH_CATEGORIES).
PATH_LABELS: Dict[str, str] = {
    # Korpus
    "corpus": "Korpus",
    "metadata": "Metadaten",
    "tokens_year": "Tokenverteilung pro Jahr",
    "dtm": "DTM",
    "tfidf": "TF-IDF",
    "cosine": "Kosinus-Matrix",
    # Topic-Model
    "topics_dist": "Document-Topic-Matrix",
    "topic_words": "Topic-Word-Matrix",
    # Verarbeitete Topics (Postprocessing, output/processed_topics/<topic-model>/)
    "top10_year_value": "Topic-Ranking pro Jahr",
    "top10_value_per_text": "Topic-Ranking pro Text",
    # Wort-Vektor-Modell
    "w2v_model": "Wort-Vektor-Modell",
    # Termset
    "termset": "Termset",
    # Document-Termset-Topics-Verarbeitungen (output/processed_termset/<Termset>/)
    "ranks": "Term-Topic-Ranking",
    "relevance": "Term-Topic-Score",
    "counts_per_year": "Term-Topic-Year-Matrix",
}


# Gruppierung der Datenquellen für die Lade-Seite (Reihenfolge = Anzeige).
PATH_CATEGORIES: Dict[str, List[str]] = {
    "Korpus": ["corpus", "metadata", "tokens_year", "dtm", "tfidf", "cosine"],
    "Topic-Model": ["topics_dist", "topic_words"],
    "Verarbeitete Topics": ["top10_year_value", "top10_value_per_text"],
    "Wort-Vektor-Modell": ["w2v_model"],
    "Termset": ["termset"],
    "Document-Termset-Topics-Verarbeitungen": ["ranks", "relevance",
                                               "counts_per_year"],
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
    # Topic-Model-Quellen: resources/topic-models/<Modell>/ (Ordnername beliebig,
    # daher '*' statt 'topics*'; Dateinamen mit Fallbacks für beide Formate).
    "topics_dist": ["resources/topic-models/*/document-topics-distribution*.csv",
                    "resources/topic-models/*/*document*topic*.csv"],
    "w2v_model": ["output/word2vec_models/*.model", "output/word2vec_models/*.kv"],
    "termset": ["resources/termsets/*.csv"],
    "topic_words": ["resources/topic-models/*/*topic_words*.csv",
                    "resources/topic-models/*/*words*tag*.csv"],
    # DTTI-Ergebnisse: output/processed_termset/<Termset>/<Topic-Modell>/...
    "ranks": ["output/processed_termset/**/*_tag_topic_rank.csv"],
    "relevance": ["output/processed_termset/**/*_tag_topic_relevance.csv"],
    "counts_per_year": ["output/processed_termset/**/*_dtti_topdocs_topic_counts_per_year.csv"],
    # Verarbeitete Topics: output/processed_topics/<topic-model>/...
    "top10_year_value": ["output/processed_topics/**/*_topdocs_year_value.csv"],
    "top10_value_per_text": ["output/processed_topics/**/*_topdocs_value_per_text_topic.csv"],
    "tokens_year": ["output/statistics/*tokens*.csv"],
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
        # Optionaler injizierter CSV-Reader (path, delimiter, **kwargs) -> DataFrame.
        # Das Dashboard hängt hier einen st.cache_data-Wrapper ein, damit
        # Reads Browser-Reloads und Sessions überleben; ohne Reader wird
        # direkt read_csv_auto benutzt (Notebooks/Skripte unverändert).
        self.reader = None

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
    # Ergebnis-Ordner-Auswahl (processed_termset / processed_topics)
    # ------------------------------------------------------------------

    # Datenquellen je processed_termset-Ordner (ein Unterordner je Termset).
    TERMSET_TOPIC_KEYS = ["ranks", "relevance", "counts_per_year"]
    # Datenquellen je processed_topics-Ordner (ein Unterordner je Topic-Modell).
    PROCESSED_TOPICS_KEYS = ["top10_year_value", "top10_value_per_text"]
    # Topic-Model-Quellen je resources/topic-models/<Modell>/-Ordner.
    TOPIC_MODEL_KEYS = ["topics_dist", "topic_words"]

    def processed_termset_root(self) -> Path:
        """Basisordner der Termset-Ergebnisse (ein Unterordner je Termset)."""
        return self.project_root / "output" / "processed_termset"

    def processed_topics_root(self) -> Path:
        """Basisordner der Topic-Postprocessings (ein Unterordner je Topic-Modell)."""
        return self.project_root / "output" / "processed_topics"

    def topic_models_root(self) -> Path:
        """Basisordner der Topic-Modelle (ein Unterordner je Modell)."""
        return self.project_root / "resources" / "topic-models"

    @staticmethod
    def _list_dirs(base: Path) -> List[str]:
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def list_termset_dirs(self) -> List[str]:
        """Namen der Unterordner in output/processed_termset/ (je Termset)."""
        return self._list_dirs(self.processed_termset_root())

    def list_dtti_dirs(self) -> List[str]:
        """Zweistufige Ordner '<Termset>/<Topic-Modell>' unter processed_termset/."""
        base = self.processed_termset_root()
        if not base.exists():
            return []
        out: List[str] = []
        for termset in sorted(p for p in base.iterdir() if p.is_dir()):
            for model in sorted(c for c in termset.iterdir() if c.is_dir()):
                out.append(f"{termset.name}/{model.name}")
        return out

    def list_processed_topics_dirs(self) -> List[str]:
        """Namen der Unterordner in output/processed_topics/ (je Topic-Modell)."""
        return self._list_dirs(self.processed_topics_root())

    def list_topic_model_dirs(self) -> List[str]:
        """Namen der Topic-Modelle unter resources/topic-models/."""
        return self._list_dirs(self.topic_models_root())

    def _apply_dir(self, base: Path, folder: str,
                   keys: List[str]) -> Dict[str, Optional[Path]]:
        """Setzt die Pfade der Schlüssel auf die passenden Dateien in ``base/folder``.

        Der Dateiname-Glob wird aus DISCOVERY_PATTERNS abgeleitet (nur der Teil
        hinter dem letzten ``/``) und rekursiv gesucht, sodass auch eine tiefere
        Ordnerstruktur (z. B. <Termset>/<Topic-Modell>/) gefunden wird. Es werden
        nur tatsächlich vorhandene Dateien gesetzt (kein Fehler bei fehlenden).
        Gibt ``{key: gefundener Pfad | None}`` zurück.
        """
        target = base / folder
        applied: Dict[str, Optional[Path]] = {}
        for key in keys:
            hit = None
            for pattern in DISCOVERY_PATTERNS.get(key, []):
                fname = pattern.rsplit("/", 1)[-1]  # nur der Dateiname-Glob
                matches = sorted(target.rglob(fname)) if target.exists() else []
                if matches:
                    hit = matches[0]
                    break
            applied[key] = hit
            if hit is not None:
                self.set_path(key, hit)
        return applied

    def apply_termset_dir(self, folder: str,
                          keys: Optional[List[str]] = None) -> Dict[str, Optional[Path]]:
        """Übernimmt die Termset-Ergebnisse aus einem processed_termset-Unterordner."""
        return self._apply_dir(self.processed_termset_root(), folder,
                               keys or self.TERMSET_TOPIC_KEYS)

    def apply_processed_topics_dir(self, folder: str,
                                   keys: Optional[List[str]] = None) -> Dict[str, Optional[Path]]:
        """Übernimmt die verarbeiteten Topics aus einem processed_topics-Unterordner."""
        return self._apply_dir(self.processed_topics_root(), folder,
                               keys or self.PROCESSED_TOPICS_KEYS)

    def apply_topic_model_dir(self, model: str) -> Dict[str, Optional[Path]]:
        """Übernimmt zu einem Topic-Modell beide Quellgruppen:

        - Topic-Model-Dateien (Document-Topic-Matrix, Topic-Word-Matrix) aus
          ``resources/topic-models/<model>/``
        - Verarbeitete Topics (Topic-Ranking pro Jahr/Text) aus
          ``output/processed_topics/<model>/``

        Es werden nur tatsächlich vorhandene Dateien gesetzt.
        """
        applied = self._apply_dir(self.topic_models_root(), model,
                                  self.TOPIC_MODEL_KEYS)
        applied.update(self._apply_dir(self.processed_topics_root(), model,
                                       self.PROCESSED_TOPICS_KEYS))
        return applied

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

    def _resolve_path(self, key: str) -> Path:
        path = self.paths[key]
        if not Path(path).exists():
            alt = self._resolve_glob(key)
            if alt is not None:
                path = alt
                self.paths[key] = alt  # aufgelösten Pfad merken
            else:
                raise FileNotFoundError(
                    f"{PATH_LABELS.get(key, key)} nicht gefunden: {path}")
        return Path(path)

    def _read(self, key: str, **kwargs) -> pd.DataFrame:
        path = self._resolve_path(key)
        if self.reader is not None:
            return self.reader(path, self.schema.delimiter, **kwargs)
        return read_csv_auto(path, delimiter=self.schema.delimiter, **kwargs)

    def _read_corpus_raw(self) -> pd.DataFrame:
        """Korpus-CSV lesen, mit Parquet-Sidecar-Cache für große Dateien.

        Beim ersten Laden wird neben der CSV eine ``<name>.parquet``
        abgelegt; solange die CSV unverändert ist, laden Folge-Starts das
        deutlich schnellere Parquet (bei ~45 Mio. Tokens Sekunden statt
        Minuten). Schlägt Parquet fehl (z. B. pyarrow fehlt), bleibt der
        CSV-Weg vollständig funktionsfähig.
        """
        path = self._resolve_path("corpus")
        sidecar = path.with_name(path.name + ".parquet")
        try:
            if (sidecar.exists()
                    and sidecar.stat().st_mtime_ns >= path.stat().st_mtime_ns):
                return pd.read_parquet(sidecar)
        except Exception:
            pass
        df = read_csv_auto(path, delimiter=self.schema.delimiter)
        try:
            df.to_parquet(sidecar, index=False)
        except Exception:
            pass
        return df

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
        df = self._read_corpus_raw()
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
    "DataStore", "ModelStore", "DEFAULT_PATHS", "PATH_LABELS", "PATH_CATEGORIES",
    "read_csv_auto", "detect_delimiter", "detect_project_root", "find_column",
]
