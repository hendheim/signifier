#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s08_name_removal
============

Erzeugt die **eigennamen-bereinigte Korpus-Variante "-n"**: Eigennamen werden mit
spaCy getilgt (``POS == PROPN`` bzw. NER-Entität). Da spaCy auf OCR-/Dialekttext
sehr viel gewöhnliches Vokabular fälschlich als Eigenname taggt, wird zweifach
gefiltert: ein Token wird nur getilgt, wenn es (1) **konsistent** als Eigenname
getaggt ist (in ~allen seinen Vorkommen) und (2) **dokumentspezifisch** ist (in
wenigen Texten). So bleiben gewöhnliche/seltene Allerweltswörter (``Kind``,
``Haus``, Dialektwörter) erhalten und nur echte Figuren-/Personennamen gehen
raus. Zusätzlich werden die Autorennamen aus den Metadaten getilgt.

Es wird NICHT neu lemmatisiert. Ausgangspunkt sind die vorhandenen Dateien
``korpus_min/lem/stop/gen.csv`` (Schritt s01/s02). Geschrieben werden
``output/processed_corpus/korpus_stop_ner.csv`` (der benannte Artefakt, Basis
der "-n"-Verarbeitung) sowie die bereinigten Stufen unter
``output/processed_corpus-n/`` (gleiche Dateinamen, Ordner mit Suffix ``-n``).
``derive_stop_n_cfg`` leitet daraus eine Pipeline-Konfiguration ab, die die
Folgeschritte (Vokabular, Statistik, POS, DTM/TF-IDF, Kosinus, Intervalle,
TF-IDF-Rang, Word2Vec) auf die "-n"-Daten anwendet und die Ausgaben in
"-n"-Ordner schreibt.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    from .pipeline_utils import detect_delimiter, identify_content_column
except ImportError:  # Standalone-Ausführung
    from pipeline_utils import detect_delimiter, identify_content_column

STAGES = ["min", "lem", "stop", "gen"]
DEFAULT_SPACY_MODEL = "de_core_news_lg"
# spaCy taggt auf OCR-/Dialekttext sehr viel gewöhnliches Vokabular fälschlich
# als Eigenname. Zwei Filter trennen echte Namen heraus:
#  1) KONSISTENZ: Ein echtes Eigenwort wird in (fast) ALLEN seinen Vorkommen als
#     Eigenname getaggt; ein gelegentlich fehlgetaggtes Allerweltswort nur
#     selten. Nur Tokens mit Eigennamen-Anteil >= dieser Schwelle gelten als Name.
#  2) DOKUMENTFREQUENZ: Echte Figuren-/Personennamen sind dokumentspezifisch;
#     ubiquitäre Wörter (in fast allen Texten) werden ausgenommen.
CONSISTENCY_THRESHOLD = 0.95
MIN_OCCURRENCES = 2          # Einzelvorkommen sind statistisch nicht belastbar
DEFAULT_MAX_DOC_RATIO = 0.30
# Wörter, die NIE als Eigenname getilgt werden dürfen (klein geschrieben):
#  - Anredeformen/Allerweltswörter (spaCy taggt sie in "Frau Müller" als Namensteil)
#  - Adels-/Verbindungspartikel aus Autorennamen ("von Droste", "… zu …"); würden
#    sonst als Präpositionen massenhaft mitgetilgt.
# Bei Bedarf hier ergänzen.
PROTECTED_TERMS = {
    "frau", "herr", "fräulein", "mann",
    "von", "zu", "vom", "zur", "zum", "van", "de", "der", "den",
    "del", "di", "da", "la", "le", "du", "am",
}
_SPACY_FALLBACKS = ("de_core_news_lg", "de_core_news_md", "de_core_news_sm")
AUTHOR_NAME_COLUMNS = ("author_surname", "author_prename", "author",
                       "editor_surname", "editor_prename")


# ---------------------------------------------------------------------------
# spaCy laden (Tagger + NER + Lemmatizer; nur Parser aus)
# ---------------------------------------------------------------------------
def load_tagger(model: str = DEFAULT_SPACY_MODEL):
    """Lädt eine spaCy-Pipeline mit Tagger, NER und Lemmatizer (Parser aus).

    Eigennamen werden über die Wortart ``POS == PROPN`` UND die NER-Entitäten
    bestimmt – beides braucht den Tagger bzw. NER. Der Parser (Dependenzen)
    wird nicht benötigt und deaktiviert (schneller). Fällt auf ein vorhandenes
    deutsches Modell zurück, falls das gewünschte fehlt.
    """
    import spacy
    tried: List[str] = []
    order = [model] + [m for m in _SPACY_FALLBACKS if m != model]
    for name in order:
        try:
            return spacy.load(name, disable=["parser"])
        except Exception:
            tried.append(name)
    raise RuntimeError("Kein deutsches spaCy-Modell gefunden (versucht: "
                       + ", ".join(tried) + ").")


def _chunks(text: str, max_chars: int = 200_000) -> Iterable[str]:
    """Lange Texte in Stücke ≤ max_chars zerlegen (an Leerzeichen schneiden)."""
    n = len(text)
    if n <= max_chars:
        if text.strip():
            yield text
        return
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            end = cut if cut > start else end
        seg = text[start:end]
        if seg.strip():
            yield seg
        start = end


NER_LABELS = ("PER", "LOC", "GPE", "ORG", "MISC")


def name_candidate_counts(texts: Iterable[str], nlp, batch_size: int = 1):
    """Zählt je Token (klein), wie oft spaCy es als Eigenname taggt und wie oft
    es insgesamt vorkommt – plus eine Oberflächenform→Lemma-Zuordnung.

    Ein Vorkommen gilt als Eigenname, wenn ``POS == PROPN`` ist ODER das Token zu
    einer NER-Entität gehört. Über das Verhältnis ``proper/total`` lässt sich die
    *Konsistenz* bestimmen (echte Namen ≈ 1.0, gelegentliche Fehl-Tags ≪ 1.0).
    ``batch_size=1`` hält den Speicher klein (spaCy-Tok2vec bündelt sonst viele
    Chunks zu einer Riesenmatrix).

    Returns ``(proper_counts, total_counts, lemma_of)``.
    """
    from collections import Counter
    proper: Counter = Counter()
    total: Counter = Counter()
    lemma_of: Dict[str, str] = {}

    def chunked():
        for t in texts:
            for c in _chunks(str(t)):
                yield c

    for doc in nlp.pipe(chunked(), batch_size=batch_size):
        for tok in doc:
            if not tok.is_alpha:
                continue
            w = tok.text
            total[w] += 1
            if tok.pos_ == "PROPN" or tok.ent_type_ in NER_LABELS:
                proper[w] += 1
                lemma_of[w] = tok.lemma_
    return proper, total, lemma_of


def author_name_set(df: pd.DataFrame) -> Set[str]:
    """Autor-/Herausgebernamen aus Metadaten-Spalten (klein, tokenisiert)."""
    names: Set[str] = set()
    for col in AUTHOR_NAME_COLUMNS:
        if col in df.columns:
            for value in df[col].dropna().astype(str):
                for part in value.replace(",", " ").split():
                    token = part.strip()
                    if len(token) > 1:
                        names.add(token)
    return names


def strip_names(series: pd.Series, names: Set[str]) -> pd.Series:
    """Entfernt alle Tokens, deren Kleinschreibung in ``names`` liegt."""
    if not names:
        return series.astype(str)

    def clean(text: str) -> str:
        return " ".join(w for w in str(text).split() if w not in names)

    return series.astype(str).map(clean)


def document_frequencies(texts: List[str], vocab: Set[str]) -> Dict[str, int]:
    """Für jedes Token in ``vocab``: in wie vielen Dokumenten es vorkommt (klein)."""
    from collections import Counter
    df: Counter = Counter()
    for t in texts:
        present = {w for w in str(t).split()} & vocab
        for w in present:
            df[w] += 1
    return dict(df)


def name_set_with_freq(processed_dir: Path, nlp,
                       metadata_path: Optional[Path] = None,
                       max_doc_ratio: float = DEFAULT_MAX_DOC_RATIO,
                       consistency: float = CONSISTENCY_THRESHOLD,
                       min_occurrences: int = MIN_OCCURRENCES
                       ) -> Tuple[Set[str], Dict[str, int]]:
    """Tilgbare Eigennamen **mit Häufigkeit** + Autorennamen.

    spaCy taggt auf OCR-/Dialekttext sehr viel gewöhnliches Vokabular fälschlich
    als Eigenname. Daher wird ein Token nur dann als Name übernommen, wenn es
      1) in mindestens ``consistency`` seiner Vorkommen als Eigenname getaggt ist
         (echte Namen ≈ 1.0, Fehl-Tags ≪ 1.0),
      2) mindestens ``min_occurrences``-mal vorkommt, und
      3) dokumentspezifisch ist (in höchstens ``max_doc_ratio`` der Dokumente).
    ``PROTECTED_TERMS`` werden NIE aufgenommen. Es werden Oberflächen- **und**
    Lemmaform erfasst (passt zu den lemmatisierten Stufen). Autorennamen aus den
    Metadaten werden immer aufgenommen.

    Returns ``(names, freq)`` – ``freq`` = Gesamthäufigkeit je Term im Korpus.
    """
    processed_dir = Path(processed_dir)
    author_names: Set[str] = set()
    proper = total = None
    lemma_of: Dict[str, str] = {}
    texts: List[str] = []

    min_file = processed_dir / "korpus_min.csv"
    src = min_file if min_file.exists() else next(
        iter(sorted(processed_dir.glob("korpus_*.csv"))), None)
    if src and src.exists():
        df = pd.read_csv(src, sep=detect_delimiter(src), encoding="utf-8")
        content_col = identify_content_column(df)
        if content_col:
            texts = df[content_col].fillna("").tolist()
            proper, total, lemma_of = name_candidate_counts(texts, nlp)
        author_names |= author_name_set(df)

    if metadata_path and Path(metadata_path).exists():
        meta = pd.read_csv(Path(metadata_path),
                           sep=detect_delimiter(Path(metadata_path)),
                           encoding="utf-8")
        author_names |= author_name_set(meta)

    names: Set[str] = set()
    if proper and texts:
        # 1) Konsistenz + Mindestvorkommen
        qualifying = {w for w, c in proper.items()
                      if total[w] >= min_occurrences and c / total[w] >= consistency}
        # 2) Dokumentfrequenz-Filter (ubiquitäre Wörter ausnehmen)
        threshold = max(1, int(max_doc_ratio * len(texts)))
        doc_freq = document_frequencies(texts, qualifying)
        qualifying = {w for w in qualifying if doc_freq.get(w, 0) <= threshold}
        # Oberflächen- UND Lemmaform übernehmen
        for w in qualifying:
            names.add(w)
            lemma = lemma_of.get(w, "")
            if len(lemma) > 1:
                names.add(lemma)

    # Schutzliste case-insensitiv anwenden: Treffer sind in Originalschreibung,
    # PROTECTED_TERMS sind klein – sonst bliebe z. B. "Frau" ungeschützt.
    names = {n for n in (names | author_names) if n.lower() not in PROTECTED_TERMS}

    total = total or {}
    freq = {t: int(total.get(t, 0)) for t in names}
    return names, freq


def build_name_set(processed_dir: Path, nlp,
                   metadata_path: Optional[Path] = None) -> Set[str]:
    """Nur die Namensmenge (siehe ``name_set_with_freq``)."""
    return name_set_with_freq(processed_dir, nlp, metadata_path)[0]


def _write_stripped(processed_dir: Path, out_dir: Path,
                    names: Set[str]) -> Dict[str, object]:
    """Tilgt ``names`` aus allen Stufen und schreibt die "-n"-Dateien sowie den
    benannten Artefakt ``<processed_dir>/korpus_stop_ner.csv``."""
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for stage in STAGES:
        src = processed_dir / f"korpus_{stage}.csv"
        if not src.exists():
            continue
        sep = detect_delimiter(src)
        df = pd.read_csv(src, sep=sep, encoding="utf-8")
        content_col = identify_content_column(df)
        if content_col:
            df[content_col] = strip_names(df[content_col], names)
        target = out_dir / f"korpus_{stage}.csv"
        df.to_csv(target, sep=sep, index=False, encoding="utf-8")
        written.append(target.name)
        print(f"[remove-names] geschrieben: {target}", flush=True)
        # Benannter Artefakt in processed_corpus: korpus_stop_ner.csv
        if stage == "stop":
            ner_file = processed_dir / "korpus_stop_ner.csv"
            df.to_csv(ner_file, sep=sep, index=False, encoding="utf-8")
            written.append(ner_file.name)
            print(f"[remove-names] geschrieben: {ner_file}", flush=True)
    return {"n_names": len(names), "written": written}


def build_stop_n_corpus(processed_dir: Path, out_dir: Path,
                        model: str = DEFAULT_SPACY_MODEL,
                        metadata_path: Optional[Path] = None) -> Dict[str, object]:
    """Auto-Pfad: Eigennamen automatisch erkennen und tilgen (ohne Review).

    Schreibt ``<processed_dir>/korpus_stop_ner.csv`` (benannter Artefakt) und die
    bereinigten Stufen unter ``out_dir``.
    """
    print("[remove-names] Lade spaCy (POS+NER) und erkenne Eigennamen ...", flush=True)
    nlp = load_tagger(model)
    names = build_name_set(Path(processed_dir), nlp, metadata_path)
    print(f"[remove-names] {len(names)} Eigennamen-Tokens (konsistent getaggt, "
          f"dokumentspezifisch) + Autoren.", flush=True)
    return _write_stripped(processed_dir, out_dir, names)


def strip_terms_corpus(processed_dir: Path, out_dir: Path,
                       terms: Iterable[str]) -> Dict[str, object]:
    """Kuratierter Pfad: tilgt nur die übergebenen ``terms`` (ohne erneute
    spaCy-Erkennung). ``PROTECTED_TERMS`` werden zur Sicherheit ausgenommen
    (case-insensitiv: schützt auch „Frau“ trotz Originalschreibung)."""
    names = {t for t in (str(x).strip() for x in terms)
             if t and t.lower() not in PROTECTED_TERMS}
    print(f"[remove-names] tilge {len(names)} ausgewählte Ausdrücke ...", flush=True)
    return _write_stripped(processed_dir, out_dir, names)


def detect_name_candidates(processed_dir: Path, model: str = DEFAULT_SPACY_MODEL,
                           metadata_path: Optional[Path] = None) -> pd.DataFrame:
    """Erkennt die tilgbaren Eigennamen und liefert sie als Tabelle
    ``[term, frequency]`` (nach Häufigkeit absteigend)."""
    print("[detect-names] Lade spaCy (POS+NER) und erkenne Eigennamen ...", flush=True)
    nlp = load_tagger(model)
    names, freq = name_set_with_freq(Path(processed_dir), nlp, metadata_path)
    rows = [{"term": t, "frequency": int(freq.get(t, 0))} for t in names]
    rows.sort(key=lambda r: (-r["frequency"], r["term"]))
    print(f"[detect-names] {len(rows)} Eigennamen-Kandidaten erkannt.", flush=True)
    return pd.DataFrame(rows, columns=["term", "frequency"])


# ---------------------------------------------------------------------------
# Kandidaten aus einer fertigen POS-Liste (kein erneutes Tagging)
# ---------------------------------------------------------------------------
def propn_candidates_from_pos(pos_csv: Path) -> pd.DataFrame:
    """Eigennamen-Kandidaten aus einer POS-Liste (Spalten ``word,pos,count``).

    Nimmt **alle als ``PROPN`` getaggten** Ausdrücke – in Originalschreibung
    (Treffer sind case-sensitiv), nach Häufigkeit (``count``) absteigend.
    Kein erneutes spaCy-Tagging: nutzt die bereits in Pipeline-Schritt 4
    (``s01_pos_tag`` → ``vocab_top*_stop_pos.csv``) erzeugte Wortartenliste.
    """
    pos_csv = Path(pos_csv)
    df = pd.read_csv(pos_csv, sep=detect_delimiter(pos_csv), encoding="utf-8")
    cols = {str(c): c for c in df.columns}
    wcol, pcol, ccol = cols.get("word"), cols.get("pos"), cols.get("count")
    if not (wcol and pcol):
        raise ValueError(f"POS-Datei braucht Spalten 'word' und 'pos': {pos_csv}")
    propn = df[df[pcol].astype(str).str.upper() == "PROPN"].copy()
    propn["term"] = propn[wcol].astype(str)
    propn["frequency"] = (pd.to_numeric(propn[ccol], errors="coerce").fillna(0).astype(int)
                          if ccol else 0)
    out = (propn.groupby("term", as_index=False)["frequency"].sum()
           .sort_values("frequency", ascending=False).reset_index(drop=True))
    return out[["term", "frequency"]]


def load_terms(path: Path) -> Set[str]:
    """Lädt eine Term-Liste (CSV mit Spalte 'term' oder eine Zeile pro Term)."""
    path = Path(path)
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, sep=detect_delimiter(path), encoding="utf-8")
        col = "term" if "term" in df.columns else df.columns[0]
        return {str(t).strip() for t in df[col].dropna() if str(t).strip()}
    except Exception:
        return {ln.strip()
                for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


# ---------------------------------------------------------------------------
# Pipeline-Konfiguration für die "-n"-Folgeschritte ableiten
# ---------------------------------------------------------------------------
def _n_path(p: str) -> str:
    """Hängt ``-n`` an das erste Pfadsegment unter ``output/`` an.

    ``output/dtm_tfidf_stop`` → ``output/dtm_tfidf_stop-n``;
    ``output/processed_corpus/korpus_stop.csv`` →
    ``output/processed_corpus-n/korpus_stop.csv``.
    """
    parts = Path(str(p)).as_posix().split("/")
    if "output" in parts:
        i = parts.index("output")
        if i + 1 < len(parts):
            parts[i + 1] = parts[i + 1] + "-n"
    return "/".join(parts)


# Pfadfelder je Schritt, die auf die "-n"-Variante umgeschrieben werden.
_PATH_FIELDS: Dict[str, List[str]] = {
    "step2_s01_2_vocabular": ["input_dir", "output_dir"],
    "step4_s01_4_pos_tag": ["input_json", "output_csv"],
    "step6_s03_dtm_tfidf": ["input_path", "output_dir"],
    "step7_s04_cosine": ["input_path", "output_path"],
    "step8_s05_dtm_tfidf_cos_intervals": ["input_path", "dtm_output", "cos_output"],
    "step9_s06_tfidf_rank": ["output_dir"],   # input_dir wird gezielt gesetzt
    "step10_s07_word_vector_model": ["input_dir", "output_dir"],
}
# Schritte 1 (Vorverarbeitung) und 5 (Gensim) entfallen: corpus_stop-n und
# corpus_gen-n entstehen direkt durch die Namenstilgung.
STOP_N_STEPS = ["2", "4", "6", "7", "8", "9", "10"]


def derive_stop_n_cfg(cfg: dict) -> dict:
    """Leitet aus der Basis-Config die "-n"-Folge-Config ab (Pfade + Schritte)."""
    c = copy.deepcopy(cfg)
    for key, fields in _PATH_FIELDS.items():
        if key in c:
            for field in fields:
                if field in c[key]:
                    c[key][field] = _n_path(c[key][field])
    # TF-IDF-Rang nur über die "-n"-Matrizen laufen lassen (sonst würde der
    # breite Default 'output' auch die nicht-bereinigten Dateien mitnehmen).
    if "step9_s06_tfidf_rank" in c:
        c["step9_s06_tfidf_rank"]["input_dir"] = c.get(
            "step6_s03_dtm_tfidf", {}).get("output_dir", "output/dtm_tfidf_stop-n")
    c.setdefault("run", {})["steps"] = list(STOP_N_STEPS)
    return c
