#!/usr/bin/env python3
"""
Berechnung eines Topic-Rankings, Identifikation der Top-N-Dokumente pro Topic,
Mapping dieser Dokumente auf Metadaten, Jahr-Topic-Matrix sowie
Dokument-Topic-Count- und Topic-Counts-pro-Jahr-Matrizen und rangbasierten
Jahres- und Textwerten.

WICHTIG:
    Die beiden Ausgabedateien

        *_topdocs_year_value.csv
        *_topdocs_value_per_text_topic.csv

    werden NUR auf Basis

        - der Top-10-Topics (gemäß Rank in der Rang-Datei)
        - und der Texte, die in der gemappten Topdocs-Tabelle in mindestens
          einem dieser Topics unter den ersten 30 Rängen vorkommen,

    berechnet.

Beispielaufruf:

    python nlp_pipeline/tt_s02_topics.py `
        --input-file resources/topic-models/topics_v3/document-topics-distribution_tag.csv `
        --output-dir output/processed_topics `
        --header-row 0 `
        --index-col 0 `
        --top-n-docs 50 `
        --metadata-file korpus/korpus.csv `
        --meta-sep ";" `
        --strip-txt-suffix
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


def _win_long_path(path: Path) -> str:
    """Windows-Extended-Length-Pfad (\\\\?\\…) für absolute Pfade > 260 Zeichen."""
    ap = os.path.abspath(str(path))
    if os.name == "nt" and not ap.startswith("\\\\?\\"):
        ap = "\\\\?\\" + ap
    return ap


def _safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Schreibt eine CSV robust und selbst-diagnostizierend.

    1) legt den Zielordner an,
    2) schreibt normal,
    3) scheitert das an einem OS-Fehler (u. a. Windows-260-Zeichen-Grenze
       oder Sync-Clients wie OneDrive), wird der Extended-Length-Pfad
       versucht,
    4) bleibt es dabei, wird ein Fehler mit vollem Pfad, Pfadlänge,
       Ordner-Existenz und Arbeitsverzeichnis geworfen – damit die Ursache
       im Log sichtbar ist, statt nur '[Errno 2] No such file or directory'.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        df.to_csv(path, **kwargs)
        return
    except OSError as e:
        try:
            df.to_csv(_win_long_path(path), **kwargs)
            print(f"[INFO] '{path.name}' via Langpfad-Fallback geschrieben "
                  "(Pfad überschritt die Windows-260-Zeichen-Grenze).")
            return
        except OSError:
            pass
        full = os.path.abspath(str(path))
        raise OSError(
            f"Konnte '{path.name}' nicht schreiben "
            f"([Errno {e.errno}] {e.strerror}). Voller Pfad "
            f"({len(full)} Zeichen): {full} · Zielordner existiert: "
            f"{path.parent.exists()} · Arbeitsverzeichnis: {os.getcwd()}"
        ) from e


# ---------------------------------------------------------------------------
# CLI / Argumente
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet Topic-Ranking, Top-N-Dokumente pro Topic und "
            "optional Metadaten-Mapping + diverse Aggregationen (Jahr, Texte)."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV-Datei mit Document–Topic-Distribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ordner, in dem alle Ausgabedateien gespeichert werden.",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=0,
        help="0-basierter Index der Header-Zeile (Default: 0).",
    )
    parser.add_argument(
        "--index-col",
        type=int,
        default=0,
        help=(
            "0-basierter Index der Index-Spalte (Default: 0). "
            "Wenn -1, wird keine Index-Spalte verwendet."
        ),
    )
    parser.add_argument(
        "--top-n-docs",
        type=int,
        default=50,
        help="Anzahl Top-Dokumente pro Topic (Default: 50).",
    )
    parser.add_argument(
        "--strip-txt-suffix",
        action="store_true",
        help="Entfernt '.txt' am Ende der Dokument-IDs im Index (optional).",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Optional: CSV-Datei mit Dokument-Metadaten (muss Spalte 'id' enthalten).",
    )
    parser.add_argument(
        "--meta-sep",
        type=str,
        default="auto",
        help="Spaltentrenner der Metadaten-CSV. 'auto' (Default) erkennt das "
             "Trennzeichen automatisch.",
    )
    parser.add_argument(
        "--year-column",
        type=str,
        default=None,
        help="Name der Metadaten-Spalte, die das Jahr enthält (z. B. 'date'). "
             "Wird intern auf 'year' gemappt. Leer = Auto-Erkennung "
             "(year_first/year/jahr/date/datum).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O und Kernlogik
# ---------------------------------------------------------------------------

def read_document_topic_distribution(
    path: Path,
    header_row: int = 0,
    index_col: Optional[int] = 0,
    strip_txt_suffix: bool = False,
) -> pd.DataFrame:
    """
    Liest eine Document–Topic-Distribution ein und gibt einen DataFrame zurück.
    """
    if index_col is not None and index_col < 0:
        index_col = None

    df = pd.read_csv(
        path,
        header=header_row,
        index_col=index_col,
    )

    # Spaltennamen säubern
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=True)

    # Index bereinigen (z. B. Dokument-ID ohne '.txt')
    if df.index.name is not None or index_col is not None:
        df.index = df.index.astype(str)
        if strip_txt_suffix:
            df.index = df.index.str.replace(".txt", "", regex=False)

    return df


def compute_topic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Summen, Mittelwerte und Standardabweichungen über alle
    numerischen Spalten des DataFrames.

    Gibt einen DataFrame mit:
        - Topic
        - Summe
        - Mittelwert
        - Standardabweichung
        - Rang
    zurück.
    """
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError(
            "Es wurden keine numerischen Spalten gefunden. "
            "Stelle sicher, dass die Topic-Spalten numerische Werte enthalten."
        )

    sums = numeric.sum()
    means = numeric.mean()
    stds = numeric.std()

    result = pd.DataFrame(
        {
            "Summe": sums,
            "Mittelwert": means,
            "Standardabweichung": stds,
        }
    )

    result = result.reset_index().rename(columns={"index": "Topic"})
    result["Rang"] = (
        result["Summe"].rank(method="dense", ascending=False).astype(int)
    )
    result = result.sort_values(by="Rang", ascending=True).reset_index(drop=True)
    return result


def compute_top_docs_per_topic(
    df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    Ermittelt für jede numerische Topic-Spalte die Top-N Dokumente
    (basierend auf den Topic-Werten).

    Gibt einen DataFrame zurück:
        - Zeilen = Rang (1..N)
        - Spalten = Topics
        - Zellen = Dokument-IDs (Indexwerte)
    """
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError(
            "Es wurden keine numerischen Spalten gefunden. "
            "Stelle sicher, dass die Topic-Spalten numerische Werte enthalten."
        )

    output = pd.DataFrame(index=range(1, top_n + 1))

    for col in numeric.columns:
        top = numeric[col].nlargest(top_n)
        doc_ids = list(top.index.astype(str))

        if len(doc_ids) < top_n:
            doc_ids += [""] * (top_n - len(doc_ids))

        output[col] = doc_ids

    return output


# ---------------------------------------------------------------------------
# Metadaten-Mapping
# ---------------------------------------------------------------------------

# Flexible ID-Spalten-Erkennung – gespiegelt aus explorer_core.schema
# (DEFAULT_ID_CANDIDATES), damit das Skript ohne Projekt-Import als CLI/Test
# lauffähig bleibt. Reihenfolge = Priorität.
_ID_CANDIDATES = ["_id", "id", "doc_id", "document_id",
                  "filename", "file_id", "index"]


def _find_id_column(df: pd.DataFrame) -> Optional[str]:
    """Findet die ID-Spalte case-insensitiv/whitespace-tolerant (wie das Schema)."""
    normalized = {str(c).lower().strip(): c for c in df.columns}
    for cand in _ID_CANDIDATES:
        if cand in df.columns:
            return cand
        hit = normalized.get(cand.lower().strip())
        if hit is not None:
            return hit
    return None


def load_metadata(metadata_file: Path, sep: str = "auto") -> pd.DataFrame:
    """
    Lädt eine Metadaten-CSV und setzt die (flexibel erkannte) ID-Spalte als
    String-Index. Die ID-Spalte wird wie im restlichen Projekt erkannt
    (``_id``/``id``/``doc_id``/…), nicht mehr hart auf 'id' verlangt.
    """
    if sep in (None, "auto"):
        # Trenner automatisch erkennen (pandas/csv.Sniffer).
        df_meta = pd.read_csv(metadata_file, sep=None, engine="python")
    else:
        df_meta = pd.read_csv(metadata_file, sep=sep)
    id_col = _find_id_column(df_meta)
    if id_col is None:
        raise ValueError(
            f"Metadaten-Datei {metadata_file} braucht eine ID-Spalte "
            f"(eine von: {', '.join(_ID_CANDIDATES)})."
        )
    df_meta[id_col] = df_meta[id_col].astype(str)
    df_meta = df_meta.set_index(id_col)
    df_meta.index.name = "id"
    return df_meta


def resolve_meta_id(doc_id: str, df_meta: pd.DataFrame) -> Optional[str]:
    """Findet die passende Metadaten-ID zu einer Dokument-ID.

    Toleriert die häufigsten Abweichungen zwischen Topic-Verteilung und
    Metadaten: umgebende Leerzeichen sowie eine '.txt'-Endung auf einer
    der beiden Seiten. Gibt die Index-ID oder None zurück.
    """
    doc_id = str(doc_id).strip()
    if not doc_id:
        return None
    if doc_id in df_meta.index:
        return doc_id
    alt = (doc_id[:-4] if doc_id.lower().endswith(".txt")
           else doc_id + ".txt")
    if alt in df_meta.index:
        return alt
    return None


def format_metadata_entry(doc_id: str, df_meta: pd.DataFrame) -> str:
    """
    Erzeugt einen formatierten Metadaten-String für eine gegebene Dokument-ID.
    Fallback: Wenn ID nicht gefunden wird, wird die Original-ID zurückgegeben.
    """
    key = resolve_meta_id(doc_id, df_meta)
    if key is None:
        return str(doc_id)

    row = df_meta.loc[key]
    parts: List[str] = []

    if "author_surname" in row.index and pd.notna(row["author_surname"]):
        parts.append(f"{row['author_surname']}:")
    if "title" in row.index and pd.notna(row["title"]):
        parts.append(f"{row['title']}.")
    if "source" in row.index and pd.notna(row["source"]):
        parts.append(f"{row['source']}.")
    year_str = None
    if "year_first" in row.index and pd.notna(row["year_first"]):
        try:
            year_str = str(int(row["year_first"]))
        except (ValueError, TypeError):
            year_str = str(row["year_first"])
    elif "year" in row.index and pd.notna(row["year"]):
        try:
            year_str = str(int(row["year"]))
        except (ValueError, TypeError):
            year_str = str(row["year"])
    if year_str is not None:
        parts.append(year_str + ".")

    return " ".join(str(p) for p in parts).strip()


def map_topdocs_to_metadata(
    topdocs_df: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wendet das Metadaten-Format auf jede Zelle der Topdocs-Matrix an.
    """
    df_out = topdocs_df.copy()
    for topic in df_out.columns:
        df_out[topic] = df_out[topic].apply(
            lambda doc_id: format_metadata_entry(doc_id, df_meta)
        )
    return df_out


# ---------------------------------------------------------------------------
# Jahr-Topic-Matrix & Jahr-Extraktion
# ---------------------------------------------------------------------------

def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extrahiert ein Jahr (1600–2099) aus einem Textstring mittels Regex.
    Gibt das Jahr als int zurück oder None.
    """
    match = re.search(r"(1[6-9]|20)\d{2}", str(text))
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# Direkte ID → Jahr-Zuordnung (robust, unabhängig vom Metadaten-String)
# ---------------------------------------------------------------------------

def meta_year_series(df_meta: pd.DataFrame) -> Optional[pd.Series]:
    """Gibt die Jahresspalte der Metadaten zurück (flexibel, priorisiert).

    Reihenfolge wie in ``format_metadata_entry`` (year_first vor year),
    zusätzlich jahr/Jahr/date/datum – damit auch Korpora mit abweichender
    Feldbenennung eine Jahreszuordnung erhalten.
    """
    for col in ("year_first", "year", "jahr", "Jahr", "date", "Date",
                "datum", "Datum"):
        if col in df_meta.columns:
            return df_meta[col]
    return None


def build_id_year_map(df_meta: pd.DataFrame) -> Dict[str, int]:
    """Ordnet jeder Dokument-ID (Metadaten-Index) direkt ein Jahr zu.

    Das Jahr wird aus der flexibel bestimmten Jahresspalte gelesen (robust
    gegen '1850', '1850.0', '1850-03-14' …). Dadurch hängt die Jahres-
    zuordnung NICHT mehr davon ab, ob das Jahr in den formatierten
    Metadaten-String gelangt – genau das war die Ursache dafür, dass
    ``*_year_topic_matrix.csv`` (und die davon abhängigen Dateien) fehlten.
    """
    series = meta_year_series(df_meta)
    if series is None:
        return {}
    id_year: Dict[str, int] = {}
    for idx, val in series.items():
        if pd.isna(val):
            continue
        year = extract_year_from_text(str(val))
        if year is not None:
            id_year[str(idx)] = year
    return id_year


def build_year_topic_matrix(
    topdocs_ids: pd.DataFrame,
    topdocs_mapped: pd.DataFrame,
    id_to_year: Dict[str, int],
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Jahr × Topic-Matrix – Jahr per ID→Jahresspalte statt Regex am String.

    Für jede Topic-Spalte werden Dokument-ID (aus ``topdocs_ids``) und der
    zugehörige Metadaten-String (aus ``topdocs_mapped``) parallel
    durchlaufen: Das Jahr kommt aus ``id_to_year`` (mit '.txt'-toleranter
    ID-Auflösung), in der Zelle steht weiterhin der Metadaten-String
    (Fallback: die ID).
    """
    data: Dict[Tuple[int, str], List[str]] = {}
    topics = list(topdocs_ids.columns)
    for topic in topics:
        ids = topdocs_ids[topic].tolist()
        strings = (topdocs_mapped[topic].tolist()
                   if topic in topdocs_mapped.columns else ids)
        for did, s in zip(ids, strings):
            did = str(did).strip()
            if not did:
                continue
            key = resolve_meta_id(did, df_meta)
            year = id_to_year.get(str(key)) if key is not None else None
            if year is None:
                continue
            cell = str(s).strip() if s is not None and str(s).strip() else did
            data.setdefault((year, topic), []).append(cell)

    if not data:
        return pd.DataFrame()

    years = sorted({y for (y, _) in data.keys()})
    reshaped_df = pd.DataFrame(index=years, columns=topics)
    for (year, topic), docs in data.items():
        reshaped_df.at[year, topic] = ", ".join(docs)
    return reshaped_df.fillna("")


# ---------------------------------------------------------------------------
# Dokument-Topic-Count-Matrix aus gemappten Metadaten
# ---------------------------------------------------------------------------

def build_document_topic_count_matrix(
    mapped_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine Dokument-Topic-Count-Matrix aus der gemappten Topdocs-Matrix.

    Rückgabe:
        - Zeilen = Dokumente (Metadaten-Strings)
        - Spalten = Topics
        - Zellen = 0/1
        - zusätzliche Spalte 'Anzahl Topics'
    """
    doc_topic_map: Dict[str, set] = {}

    for topic in mapped_df.columns:
        for doc in mapped_df[topic].dropna():
            doc_str = str(doc).strip()
            if not doc_str:
                continue
            doc_topic_map.setdefault(doc_str, set()).add(topic)

    if not doc_topic_map:
        return pd.DataFrame()

    unique_topics = list(mapped_df.columns)
    binary_matrix = pd.DataFrame(
        0, index=sorted(doc_topic_map.keys()), columns=unique_topics, dtype=int
    )

    for doc_str, topics in doc_topic_map.items():
        binary_matrix.loc[doc_str, list(topics)] = 1

    binary_matrix.insert(0, "Anzahl Topics", binary_matrix.sum(axis=1))
    binary_matrix.reset_index(inplace=True)
    binary_matrix.rename(columns={"index": "Dokument"}, inplace=True)

    return binary_matrix


# ---------------------------------------------------------------------------
# Topic-Counts pro Jahr aus Dokument-Topic-Count-Matrix
# ---------------------------------------------------------------------------

def build_topic_counts_per_year(
    doc_topic_count_df: pd.DataFrame,
    string_to_year: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Erzeugt eine numerische Matrix: Jahr × Topic = Anzahl Dokumente.

    ``string_to_year`` (Metadaten-String → Jahr, aus der direkten
    ID→Jahr-Zuordnung) hat Vorrang; nur wenn ein Dokument dort fehlt, wird
    als Fallback das Jahr aus dem String geregext.
    """
    df = doc_topic_count_df.copy()

    if "Dokument" not in df.columns:
        raise ValueError("Erwarte Spalte 'Dokument' in der Dokument-Topic-Count-Matrix.")

    if string_to_year:
        keys = df["Dokument"].astype(str).str.strip()
        df["Jahr"] = keys.map(string_to_year)
        miss = df["Jahr"].isna()
        if miss.any():
            df.loc[miss, "Jahr"] = df.loc[miss, "Dokument"].apply(extract_year_from_text)
    else:
        df["Jahr"] = df["Dokument"].apply(extract_year_from_text)
    df = df.dropna(subset=["Jahr"])
    df["Jahr"] = df["Jahr"].astype(int)

    non_topic_cols = {"Dokument", "Jahr", "Anzahl Topics"}
    topic_cols = [col for col in df.columns if col not in non_topic_cols]

    grouped = df.groupby("Jahr")[topic_cols].sum()
    return grouped


# ---------------------------------------------------------------------------
# Jahr → Liste wichtiger Texte aus Dokument-Topic-Count-Matrix
# ---------------------------------------------------------------------------

def build_year_document_ranking(
    doc_topic_count_df: pd.DataFrame,
    string_to_year: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Erzeugt eine Tabelle:

        - Zeilen = Jahre
        - Spalten = 'Anzahl', 'Dokument 1', 'Dokument 2', ...

    ``string_to_year`` (Metadaten-String → Jahr) hat Vorrang; Fallback ist
    das Regex-Jahr aus dem String.
    """
    if "Dokument" not in doc_topic_count_df.columns:
        raise ValueError("Erwarte Spalte 'Dokument' in der Dokument-Topic-Count-Matrix.")

    docs_in_order = doc_topic_count_df["Dokument"].astype(str).tolist()

    from collections import defaultdict
    jahres_map: Dict[int, List[str]] = defaultdict(list)
    seen: set[str] = set()

    for text in docs_in_order:
        if text in seen:
            continue
        year = None
        if string_to_year:
            year = string_to_year.get(text.strip())
        if year is None:
            year = extract_year_from_text(text)
        if year is not None:
            jahres_map[year].append(text)
            seen.add(text)

    if not jahres_map:
        return pd.DataFrame()

    max_len = max(len(doks) for doks in jahres_map.values())
    columns = ["Anzahl"] + [f"Dokument {i+1}" for i in range(max_len)]

    final_df = pd.DataFrame(index=sorted(jahres_map.keys()), columns=columns)

    for year, texts in jahres_map.items():
        final_df.at[year, "Anzahl"] = len(texts)
        final_df.loc[year, columns[1:1 + len(texts)]] = texts

    final_df = final_df.fillna("")
    return final_df


# ---------------------------------------------------------------------------
# Rangbasierte Jahreswerte (auf Basis beliebiger Ranking-Matrix)
# ---------------------------------------------------------------------------

def build_year_values_from_rank(
    ranked_topdocs: pd.DataFrame,
    year_document_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet pro Jahr einen Wert, der sich aus den Rangwerten der zugeordneten
    Texte ergibt.

    - ranked_topdocs: Ranking-Matrix (Zeilen = Ränge, Spalten = Topics, Zellen = Text)
      -> nur die Zeilen/Spalten, die du berücksichtigen willst (z.B. Top-10-Topics, Top-30).
    - year_document_df: Jahr → Texte (Anzahl + Dokument 1..N)

    Gewicht pro Text = (Anzahl Zeilen - RangIndex)
    Jahreswert = Summe der Gewichte aller Texte dieses Jahres
    """
    if ranked_topdocs.empty or year_document_df.empty:
        return pd.DataFrame()

    tmp = ranked_topdocs.copy()
    tmp = tmp.reset_index(drop=True)  # Rang 0..N-1
    tmp["Rang"] = tmp.index
    n = len(tmp)

    long_df = tmp.melt(id_vars="Rang", var_name="Topic", value_name="Text")
    long_df["Text"] = long_df["Text"].astype(str).str.strip()
    long_df = long_df[long_df["Text"].notna() & (long_df["Text"] != "")]
    long_df["Wert"] = n - long_df["Rang"]

    wert_by_text = long_df.groupby("Text", as_index=True)["Wert"].sum()

    text_cols = [c for c in year_document_df.columns if c.startswith("Dokument")]
    rows: List[Dict[str, int]] = []

    for year, row in year_document_df.iterrows():
        for col in text_cols:
            text = row.get(col)
            if pd.isna(text):
                continue
            t = str(text).strip()
            if not t:
                continue
            if t in wert_by_text.index:
                wert = int(wert_by_text.loc[t])
                rows.append({"Jahr": int(year), "Wert": wert})

    if not rows:
        return pd.DataFrame()

    df_jtw = pd.DataFrame(rows)
    df_jahreswerte = df_jtw.groupby("Jahr", as_index=False)["Wert"].sum()
    return df_jahreswerte


# ---------------------------------------------------------------------------
# Rangbasierte Textwerte (auf Basis beliebiger Ranking-Matrix)
# ---------------------------------------------------------------------------

def build_value_per_text_and_topic(
    ranked_topdocs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine Tabelle mit:

        - Zeilen = Texte
        - Spalten:
            - 'SummeWert' (Gesamtwert über alle Topics)
            - pro Topic eine Spalte mit Wert

    ranked_topdocs:
        - Zeilen = Ränge
        - Spalten = Topics
        - Zellen = Text
        -> nur relevanter Ausschnitt (z.B. Top-10-Topics, Top-30-Ränge).
    """
    if ranked_topdocs.empty:
        return pd.DataFrame(columns=["Text", "SummeWert"])

    df = ranked_topdocs.copy()
    df = df.reset_index(drop=True)
    n = len(df)
    df["__RANG__"] = range(n)  # 0 = beste Zeile

    df_long = df.melt(id_vars="__RANG__", var_name="Tag", value_name="Text")

    def clean_text(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v != "" else None
        return v

    df_long["Text"] = df_long["Text"].map(clean_text)
    df_long = df_long[df_long["Text"].notna()]

    df_long["Wert"] = (n - df_long["__RANG__"]).astype(int)

    summe_pro_text = (
        df_long.groupby("Text", as_index=False)["Wert"].sum()
        .rename(columns={"Wert": "SummeWert"})
    )

    tag_matrix = (
        df_long.groupby(["Text", "Tag"], as_index=False)["Wert"].sum()
        .pivot(index="Text", columns="Tag", values="Wert")
        .fillna(0)
        .astype(int)
    )

    ergebnis = (
        summe_pro_text.set_index("Text")
        .join(tag_matrix, how="left")
        .fillna(0)
        .reset_index()
    )

    tag_spalten = sorted([c for c in ergebnis.columns if c not in ("Text", "SummeWert")])
    ergebnis = ergebnis[["Text", "SummeWert"] + tag_spalten]
    ergebnis = ergebnis.sort_values("SummeWert", ascending=False)
    return ergebnis


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Top-10-Topics + Top-30-Ränge
# ---------------------------------------------------------------------------

def get_top_k_topics(topic_stats: pd.DataFrame, k: int = 10) -> List[str]:
    """
    Liefert die Namen der Top-k-Topics anhand der Spalte 'Rang'.
    """
    if "Topic" not in topic_stats.columns or "Rang" not in topic_stats.columns:
        raise ValueError("topic_stats braucht Spalten 'Topic' und 'Rang'.")
    return (
        topic_stats.sort_values("Rang", ascending=True)
        .head(k)["Topic"]
        .astype(str)
        .tolist()
    )


def build_mapped_topdocs_subset_topk(
    mapped_topdocs: pd.DataFrame,
    topic_stats: pd.DataFrame,
    top_k: int = 10,    
    max_rank: int = 0,
) -> pd.DataFrame:
    """
    Erzeugt einen Teil-DataFrame:

        - nur Top-k-Topics (gemäß topic_stats['Rang'])
        - nur die ersten max_rank Zeilen (Top-ranks)
    """
    top_topics = get_top_k_topics(topic_stats, k=top_k)
    available = [t for t in top_topics if t in mapped_topdocs.columns]
    if not available:
        return pd.DataFrame()

    sub = mapped_topdocs[available].copy()
    sub = sub.reset_index(drop=True)
    max_rank = min(max_rank, len(sub))
    sub = sub.iloc[:max_rank]
    return sub


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _ensure_output_file(path: Path, columns: List[str],
                        index_label: Optional[str] = None) -> None:
    """Schreibt eine leere, aber gültige CSV mit Kopfzeile, falls die Datei
    nicht existiert.

    Damit entstehen ALLE angekündigten Ausgabedateien auch dann, wenn keine
    Jahre/Metadaten zugeordnet werden konnten – nachgelagerte Seiten laufen
    dann auf leere Tabellen statt auf 'No such file or directory'.
    """
    if path.exists():
        return
    empty = pd.DataFrame(columns=columns)
    if index_label:
        empty.index.name = index_label
        _safe_to_csv(empty, path, encoding="utf-8", index_label=index_label)
    else:
        _safe_to_csv(empty, path, index=False, encoding="utf-8")
    print(f"[WARN] {path.name}: ohne Inhalt erzeugt (keine zuordenbaren "
          "Jahre/Metadaten – siehe Warnungen oben).")


def main() -> None:
    args = parse_args()

    # 1) Document–Topic-Distribution einlesen
    df = read_document_topic_distribution(
        path=args.input_file,
        header_row=args.header_row,
        index_col=args.index_col,
        strip_txt_suffix=args.strip_txt_suffix,
    )

    # Output-Ordner
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = args.input_file.stem

    rank_file = output_dir / f"{basename}_rank.csv"
    topdocs_file = output_dir / f"{basename}_topdocs.csv"
    mapped_file = output_dir / f"{basename}_topdocs_mapped.csv"
    year_topic_file = output_dir / f"{basename}_topdocs_year_topic_matrix.csv"
    doc_topic_count_file = output_dir / f"{basename}_topdocs_topic_count_matrix.csv"
    topic_counts_per_year_file = output_dir / f"{basename}_topdocs_topic_counts_per_year.csv"
    year_document_map_file = output_dir / f"{basename}_topdocs_year_document_map.csv"
    year_value_file = output_dir / f"{basename}_topdocs_year_value.csv"
    text_topic_value_file = output_dir / f"{basename}_topdocs_value_per_text_topic.csv"

    # 2) Topic-Ranking
    topic_stats = compute_topic_stats(df)
    _safe_to_csv(topic_stats, rank_file, index=False, encoding="utf-8")
    print(f"[OK] Topic-Ranking gespeichert in: {rank_file}")

    # 3) Top-N-Dokumente pro Topic
    topdocs_df = compute_top_docs_per_topic(df, top_n=args.top_n_docs)
    _safe_to_csv(topdocs_df, topdocs_file, index=False, encoding="utf-8")
    print(f"[OK] Top-{args.top_n_docs}-Dokumente pro Topic gespeichert in: {topdocs_file}")

    # 4) Metadaten-abhängige Schritte
    if args.metadata_file is not None:
        print(f"[INFO] Metadaten werden aus {args.metadata_file} geladen …")
        df_meta = load_metadata(args.metadata_file, sep=args.meta_sep)

        # Jahresspalte flexibel auf 'year' mappen, damit date/jahr/datum
        # genauso als Jahr erkannt werden wie year/year_first.
        year_col = (args.year_column or "").strip()
        if not year_col and "year_first" not in df_meta.columns \
                and "year" not in df_meta.columns:
            for cand in ("jahr", "Jahr", "date", "Date", "datum", "Datum"):
                if cand in df_meta.columns:
                    year_col = cand
                    break
        if year_col and year_col in df_meta.columns and "year" not in df_meta.columns:
            df_meta["year"] = df_meta[year_col]
            print(f"[INFO] Jahresspalte '{year_col}' wird als Jahr verwendet.")

        # Direkte ID→Jahr-Zuordnung: löst die Jahresangabe unabhängig vom
        # formatierten Metadaten-String (der das Jahr nicht immer enthält).
        id_to_year = build_id_year_map(df_meta)
        _year_series = meta_year_series(df_meta)
        if _year_series is None:
            print("[WARN] Keine Jahresspalte in den Metadaten gefunden "
                  "(gesucht: year_first, year, jahr, Jahr, date, datum). "
                  "Ohne Jahr entstehen die jahr-basierten Dateien nur leer.")

        # Diagnose: Wie viele Dokument-IDs der Topic-Verteilung finden sich
        # in den Metadaten – und wie viele haben ein zuordenbares Jahr?
        _ids = [str(i) for i in pd.unique(topdocs_df.values.ravel())
                if pd.notna(i) and str(i).strip()]
        _keys = [resolve_meta_id(i, df_meta) for i in _ids]
        _hits = sum(1 for k in _keys if k is not None)
        _year_hits = sum(1 for k in _keys
                         if k is not None and str(k) in id_to_year)
        print(f"[INFO] Metadaten-Mapping: {_hits} von {len(_ids)} "
              f"Dokument-IDs in den Metadaten gefunden; davon {_year_hits} "
              "mit zuordenbarem Jahr.")
        if _ids and _hits == 0:
            print("[WARN] KEINE Dokument-ID der Topic-Verteilung passt zu den "
                  "Metadaten! Bitte ID-Spalte der Metadaten und die "
                  "Dokument-IDs der Distribution vergleichen (z. B. "
                  "'.txt'-Endung, führende/abschließende Leerzeichen). "
                  "Alle jahr-/metadatenbasierten Ausgaben bleiben leer.")
        elif _ids and _year_hits == 0:
            print("[WARN] Zwar wurden IDs zugeordnet, aber KEINE besitzt ein "
                  "zuordenbares Jahr – die jahr-basierten Dateien bleiben "
                  "leer. Bitte die Werte der Jahresspalte prüfen.")

        # 4a) IDs → Metadaten-Strings
        topdocs_mapped = map_topdocs_to_metadata(topdocs_df, df_meta)
        _safe_to_csv(topdocs_mapped, mapped_file, index=False, encoding="utf-8")
        print(f"[OK] Gemappte Top-Dokumente gespeichert in: {mapped_file}")

        # Metadaten-String → Jahr (aus der direkten ID→Jahr-Zuordnung), damit
        # auch die count-basierten Ausgaben ohne Regex-am-String auskommen.
        string_to_year: Dict[str, int] = {}
        for _topic in topdocs_df.columns:
            _tids = topdocs_df[_topic].astype(str).tolist()
            _tstr = (topdocs_mapped[_topic].astype(str).tolist()
                     if _topic in topdocs_mapped.columns else _tids)
            for _did, _s in zip(_tids, _tstr):
                _s = _s.strip()
                if not _s:
                    continue
                _k = resolve_meta_id(_did.strip(), df_meta)
                _y = id_to_year.get(str(_k)) if _k is not None else None
                if _y is not None:
                    string_to_year.setdefault(_s, _y)

        # 4b) Jahr × Topic-Matrix (Texte in Zellen) – Jahr per ID→Jahresspalte
        year_topic_df = build_year_topic_matrix(
            topdocs_df, topdocs_mapped, id_to_year, df_meta)
        if not year_topic_df.empty:
            _safe_to_csv(year_topic_df, year_topic_file, encoding="utf-8", index_label="Jahr")
            print(f"[OK] Jahr-Topic-Matrix gespeichert in: {year_topic_file}")
        else:
            print("[WARN] Keine zuordenbaren Jahre – folgende Dateien werden "
                  "leer (nur Kopfzeile) erzeugt: "
                  f"{year_topic_file.name}, {topic_counts_per_year_file.name}, "
                  f"{year_document_map_file.name}, {year_value_file.name}, "
                  f"{text_topic_value_file.name}. Ursache meist: Jahresspalte "
                  "fehlt/unerkannt oder Dokument-IDs passen nicht zu den "
                  "Metadaten (siehe [INFO] Metadaten-Mapping oben).")

        # 4c) Dokument-Topic-Count-Matrix
        doc_topic_count_df = build_document_topic_count_matrix(topdocs_mapped)
        if not doc_topic_count_df.empty:
            _safe_to_csv(doc_topic_count_df, doc_topic_count_file, index=False, encoding="utf-8")
            print(f"[OK] Dokument-Topic-Count-Matrix gespeichert in: {doc_topic_count_file}")

            # 4d) Topic-Counts pro Jahr
            topic_counts_per_year_df = build_topic_counts_per_year(
                doc_topic_count_df, string_to_year=string_to_year)
            if not topic_counts_per_year_df.empty:
                _safe_to_csv(
                    topic_counts_per_year_df, topic_counts_per_year_file,
                    encoding="utf-8", index_label="Jahr")
                print(f"[OK] Topic-Counts pro Jahr gespeichert in: {topic_counts_per_year_file}")
            else:
                print("[WARN] Keine gültigen Jahresangaben für Topic-Counts-pro-Jahr gefunden.")

            # 4e) Jahr → Liste der wichtigsten Texte
            year_document_df = build_year_document_ranking(
                doc_topic_count_df, string_to_year=string_to_year)
            if not year_document_df.empty:
                _safe_to_csv(
                    year_document_df, year_document_map_file,
                    encoding="utf-8", index_label="Jahr")
                print(f"[OK] Jahr-Dokument-Matrix gespeichert in: {year_document_map_file}")
            else:
                print("[WARN] Keine gültigen Jahresangaben für Jahr-Dokument-Matrix gefunden.")

            # 4f) BESCHRÄNKTE RANG-AUSGABEN:
            #     Nur Top-10-Topics + Texte, die dort in den ersten 30 Rängen vorkommen.

            top10_sub = build_mapped_topdocs_subset_topk(
                mapped_topdocs=topdocs_mapped,
                topic_stats=topic_stats,
                top_k=10,
                max_rank=30,
            )

            if not top10_sub.empty and not year_document_df.empty:
                # 4f.1) value_per_text_topic (beschränkt)
                value_per_text_topic_df = build_value_per_text_and_topic(top10_sub)
                if not value_per_text_topic_df.empty:
                    _safe_to_csv(
                        value_per_text_topic_df, text_topic_value_file,
                        index=False, encoding="utf-8")
                    print(f"[OK] (Top-10 / Top-30) Text-Topic-Werte gespeichert in: {text_topic_value_file}")
                else:
                    print("[WARN] Keine rangbasierten Text-Topic-Werte (Top-10/Top-30) erzeugt.")

                # 4f.2) year_value (beschränkt)
                year_value_df = build_year_values_from_rank(top10_sub, year_document_df)
                if not year_value_df.empty:
                    _safe_to_csv(year_value_df, year_value_file, index=False, encoding="utf-8")
                    print(f"[OK] (Top-10 / Top-30) Jahreswerte gespeichert in: {year_value_file}")
                else:
                    print("[WARN] Keine rangbasierten Jahreswerte (Top-10/Top-30) erzeugt.")
            else:
                print("[WARN] Kein gültiges Subset für Top-10-Topics/Top-30-Ränge oder keine Jahr-Dokument-Matrix verfügbar.")

        # Garantie: Alle angekündigten Ausgabedateien existieren – notfalls
        # leer mit Kopfzeile. So können Folgeseiten nicht mehr an fehlenden
        # Dateien scheitern; leere Tabellen zeigen das Problem sichtbar an.
        topic_cols = [str(c) for c in topdocs_df.columns]
        _ensure_output_file(year_topic_file, topic_cols, index_label="Jahr")
        _ensure_output_file(doc_topic_count_file,
                            ["Dokument", "Anzahl Topics"] + topic_cols)
        _ensure_output_file(topic_counts_per_year_file, topic_cols,
                            index_label="Jahr")
        _ensure_output_file(year_document_map_file, ["Anzahl"],
                            index_label="Jahr")
        _ensure_output_file(year_value_file, ["Jahr", "Wert"])
        _ensure_output_file(text_topic_value_file,
                            ["Text", "SummeWert"] + topic_cols)


if __name__ == "__main__":
    main()
