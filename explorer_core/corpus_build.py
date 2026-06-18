#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.corpus_build
==========================

UI-freie Logik für die Seite **Korpus Datenbank erstellen**. Richtet sich nach
den Projekt-Notebooks ``00_01_txt_to_corpus`` und ``00_02_xml_to_corpus``:

- **txt** (Notebook Teil B): einen Ordner/Satz von ``.txt``-Dateien über eine
  ID-Spalte mit einer Metadaten-CSV zusammenführen (Dateiname = ID). Ohne
  Metadaten-CSV entsteht ein minimales Korpus (``id`` + ``content``).
- **xml** (TEI-Notebook): Metadaten per XPath aus dem TEI-Header lesen,
  Fließtext aus ``.//tei:text//tei:body`` (Fallback ``.//tei:text``), Whitespace
  normalisieren (Soft Hyphens entfernen). XPath via **lxml** (mit
  TEI-Namespace); Fallback auf die Standardbibliothek, falls lxml fehlt.

Geschrieben werden – wie vom Dashboard gefordert – zwei Dateien nach ``/korpus``:
``korpus.csv`` (Metadaten + ``content``) und ``metadaten.csv`` (nur Metadaten),
beide UTF-8 und komma-separiert.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from lxml import etree as _LET
    _HAS_LXML = True
except Exception:  # pragma: no cover
    _HAS_LXML = False
import xml.etree.ElementTree as _ET

CONTENT_COLUMN = "content"
ID_COLUMN = "id"

TEI_NS_URI = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS_URI}

# Default-XPaths (aus dem TEI-Notebook 00_02)
DEFAULT_TEI_META: Dict[str, str] = {
    "title": ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblFull"
             "//tei:title[@level='a' and @type='main']",
    "author_prename": ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc"
                      "//tei:biblFull//tei:author//tei:forename",
    "author_surname": ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc"
                      "//tei:biblFull//tei:author//tei:surname",
    "year": ".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblFull"
            "//tei:date[@type='firstPublication']",
}
DEFAULT_TEI_CONTENT: List[str] = [".//tei:text//tei:body", ".//tei:text"]
DEFAULT_ID_XPATH = "./@xml:id"


# ---------------------------------------------------------------------------
# Gemeinsame Helfer
# ---------------------------------------------------------------------------
def normalize_ws(s: str) -> str:
    """Whitespace normalisieren (Soft Hyphens entfernen, Mehrfach-Whitespace
    zu einem Leerzeichen) – wie im Notebook ``normalize_ws``."""
    s = (s or "").replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\n[ \t]*", "\n", s)        # Leerzeichen am Zeilenanfang entfernen
    s = re.sub(r"\n{3,}", "\n\n", s)        # höchstens eine Leerzeile zwischen Absätzen
    return s.strip()


def _dedupe_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Doppelte IDs eindeutig machen (id, id_2, id_3, …)."""
    if df.empty or ID_COLUMN not in df.columns:
        return df
    seen: Dict[str, int] = {}
    new_ids = []
    for raw in df[ID_COLUMN].astype(str):
        if raw in seen:
            seen[raw] += 1
            new_ids.append(f"{raw}_{seen[raw]}")
        else:
            seen[raw] = 1
            new_ids.append(raw)
    df = df.copy()
    df[ID_COLUMN] = new_ids
    return df


# ---------------------------------------------------------------------------
# XML-Backend (lxml bevorzugt, sonst ElementTree)
# ---------------------------------------------------------------------------
def _strip_namespaces_et(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _parse_root(xml_text: str, use_namespace: bool):
    """Parst XML; bei use_namespace=False werden Namespaces entfernt."""
    if _HAS_LXML:
        parser = _LET.XMLParser(recover=True, huge_tree=True)
        root = _LET.fromstring(xml_text.encode("utf-8"), parser)
        if not use_namespace:
            for el in root.iter():
                if isinstance(el.tag, str) and "}" in el.tag:
                    el.tag = _LET.QName(el).localname
            _LET.cleanup_namespaces(root)
        return root, True
    root = _ET.fromstring(xml_text)
    if not use_namespace:
        _strip_namespaces_et(root)
    return root, False


def _xpath_texts(root, is_lxml: bool, xpath: str, use_namespace: bool) -> List[str]:
    ns = NS if use_namespace else None
    # Wurzel selbst
    if xpath in (".", "", "/"):
        txt = normalize_ws(" ".join(root.itertext()) if is_lxml
                           else "".join(root.itertext()))
        return [txt] if txt else []
    out: List[str] = []
    if is_lxml:
        try:
            res = root.xpath(xpath, namespaces=ns)
        except Exception:
            return []
        for r in res:
            if hasattr(r, "itertext"):
                out.append(normalize_ws(" ".join(r.itertext())))
            else:
                out.append(normalize_ws(str(r)))
    else:
        try:
            res = root.findall(xpath, ns) if ns else root.findall(xpath)
        except Exception:
            return []
        for el in res:
            out.append(normalize_ws("".join(el.itertext())))
    return [x for x in out if x]


def _first_text(root, is_lxml, xpath, use_namespace) -> str:
    vals = _xpath_texts(root, is_lxml, xpath, use_namespace)
    return vals[0] if vals else ""


def _all_texts(root, is_lxml, xpath, use_namespace) -> str:
    # Identische Treffer entdoppeln (z. B. dieselbe <date> in mehreren <bibl>),
    # Reihenfolge bewahren; echte Mehrfachwerte bleiben mit '; ' verbunden.
    seen, uniq = set(), []
    for v in _xpath_texts(root, is_lxml, xpath, use_namespace):
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return "; ".join(uniq)


def _localname(tag) -> str:
    return tag.split("}", 1)[1] if isinstance(tag, str) and "}" in tag else tag


def path_leaf(path: str) -> str:
    """Letztes Pfadsegment ohne 'tei:'-Präfix (als Default-Spaltenname)."""
    seg = (path or "").rstrip("/").split("/")[-1]
    return seg.split(":", 1)[1] if ":" in seg else seg


def discover_xml_paths(xml_text: str, use_namespace: bool = True,
                       max_paths: int = 400) -> List[Tuple[str, str]]:
    """Ermittelt die in einem XML-Dokument vorhandenen Element-Pfade.

    Liefert eine Liste ``(xpath, textprobe)`` distinkter Tag-Ketten (absolute
    Pfade ab Wurzel, ``tei:``-Präfix bei ``use_namespace``). Nur Elemente mit
    nicht-leerem Text werden aufgenommen – als auswählbare Metadaten-Ziele.
    """
    root, is_lxml = _parse_root(xml_text, use_namespace)
    pre = (lambda n: f"tei:{n}") if use_namespace else (lambda n: n)
    results: Dict[str, str] = {}

    if is_lxml:
        def parent(node):
            return node.getparent()
    else:
        pmap = {c: p for p in root.iter() for c in p}
        def parent(node):
            return pmap.get(node)

    for el in root.iter():
        if not isinstance(getattr(el, "tag", None), str):
            continue
        chain, node = [], el
        while node is not None:
            chain.append(_localname(node.tag))
            node = parent(node)
        # RELATIV zur Wurzel (Wurzel-Tag weglassen) – so können sowohl lxml
        # als auch ElementTree die Pfade abfragen; absolute '/...'-Pfade kann
        # ElementTree NICHT, wodurch Metadaten sonst leer blieben.
        rel = [pre(c) for c in reversed(chain)][1:]
        path = "/".join(rel) if rel else "."
        text = normalize_ws(" ".join(el.itertext()) if is_lxml
                            else "".join(el.itertext()))
        if text and path not in results:
            results[path] = text[:120]
        if len(results) >= max_paths:
            break
    return list(results.items())


def parse_xml_document(xml_text: str, fname: str,
                       meta_xpaths: Optional[Dict[str, str]] = None,
                       content_xpaths: Optional[List[str]] = None,
                       use_namespace: bool = True,
                       id_xpath: str = DEFAULT_ID_XPATH
                       ) -> Tuple[str, str, Dict[str, str]]:
    """Parst ein TEI/XML-Dokument → (id, content, meta).

    Metadaten je Kategorie via ``all_texts`` (mehrere Treffer mit '; ' verbunden),
    Inhalt aus dem ersten nicht-leeren ``content_xpaths``-Treffer.
    """
    meta_xpaths = meta_xpaths if meta_xpaths is not None else DEFAULT_TEI_META
    content_xpaths = content_xpaths if content_xpaths is not None else DEFAULT_TEI_CONTENT
    root, is_lxml = _parse_root(xml_text, use_namespace)

    doc_id = _first_text(root, is_lxml, id_xpath, use_namespace) if id_xpath else ""
    if not doc_id:
        doc_id = Path(fname).stem

    content = ""
    for cx in content_xpaths:
        content = _first_text(root, is_lxml, cx, use_namespace)
        if content:
            break

    meta = {cat: _all_texts(root, is_lxml, xp, use_namespace)
            for cat, xp in meta_xpaths.items()}
    return doc_id, content, meta


def build_from_xml(items: List[Tuple[str, str]],
                   meta_xpaths: Optional[Dict[str, str]] = None,
                   content_xpaths: Optional[List[str]] = None,
                   use_namespace: bool = True,
                   id_xpath: str = DEFAULT_ID_XPATH) -> pd.DataFrame:
    """Baut ein Korpus aus ``(dateiname, xml_text)``-Paaren (TEI-orientiert)."""
    meta_xpaths = meta_xpaths if meta_xpaths is not None else DEFAULT_TEI_META
    rows = []
    for fname, xml_text in items:
        doc_id, content, meta = parse_xml_document(
            xml_text, fname, meta_xpaths, content_xpaths, use_namespace, id_xpath)
        row = {ID_COLUMN: doc_id}
        row.update({k: meta.get(k, "") for k in meta_xpaths})
        row[CONTENT_COLUMN] = content
        rows.append(row)
    cols = [ID_COLUMN] + list(meta_xpaths.keys()) + [CONTENT_COLUMN]
    return _dedupe_ids(pd.DataFrame(rows, columns=cols))


# ---------------------------------------------------------------------------
# TXT (Notebook Teil B: Metadaten-CSV + TXT-Ordner zusammenführen)
# ---------------------------------------------------------------------------
def build_from_txt(items: List[Tuple[str, str]],
                   metadata_df: Optional[pd.DataFrame] = None,
                   id_column: str = ID_COLUMN
                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Baut ein Korpus aus ``(dateiname, text)``-Paaren.

    Mit ``metadata_df`` wird der Text per ``id_column`` (Dateiname ohne Endung)
    in die Metadaten-Tabelle eingefügt (wie Notebook 00_01 Teil B). Ohne
    Metadaten entsteht ein minimales Korpus (``id`` + ``content``).

    Returns
    -------
    (corpus_df, unmatched) – ``unmatched`` listet Dateinamen/IDs ohne
    Entsprechung in der Metadaten-CSV.
    """
    texts = {Path(fname).stem: (text or "") for fname, text in items}

    if metadata_df is None:
        df = pd.DataFrame(
            [{ID_COLUMN: k, CONTENT_COLUMN: v.strip()} for k, v in texts.items()],
            columns=[ID_COLUMN, CONTENT_COLUMN])
        return _dedupe_ids(df), []

    df = metadata_df.copy()
    if id_column not in df.columns:
        raise ValueError(f"ID-Spalte '{id_column}' nicht in den Metadaten "
                         f"gefunden. Vorhandene Spalten: {list(df.columns)}")
    if CONTENT_COLUMN not in df.columns:
        df[CONTENT_COLUMN] = ""
    ids = df[id_column].astype(str)
    matched_ids = set()
    for fid, content in texts.items():
        mask = ids == str(fid)
        if mask.any():
            df.loc[mask, CONTENT_COLUMN] = content
            matched_ids.add(fid)
    unmatched = sorted(set(texts) - matched_ids)
    # id-Spalte vereinheitlichen
    if id_column != ID_COLUMN and ID_COLUMN not in df.columns:
        df = df.rename(columns={id_column: ID_COLUMN})
    return df, unmatched


# ---------------------------------------------------------------------------
# Schreiben nach /korpus
# ---------------------------------------------------------------------------
def split_corpus_metadata(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Reine Metadaten-Sicht (ohne ``content``)."""
    return corpus_df.drop(columns=[CONTENT_COLUMN], errors="ignore").copy()


def write_corpus(corpus_df: pd.DataFrame, korpus_dir: Path,
                 metadata_df: Optional[pd.DataFrame] = None,
                 id_column: str = ID_COLUMN) -> Tuple[Path, Path]:
    """Schreibt ``korpus.csv`` (Metadaten + content) und ``metadaten.csv``
    (nur Metadaten) nach ``korpus_dir`` – UTF-8, komma-separiert."""
    korpus_dir = Path(korpus_dir)
    korpus_dir.mkdir(parents=True, exist_ok=True)

    if metadata_df is not None:
        meta = metadata_df.drop(columns=[CONTENT_COLUMN], errors="ignore").copy()
        key = id_column if id_column in meta.columns and id_column in corpus_df.columns else ID_COLUMN
        content = corpus_df[[key, CONTENT_COLUMN]]
        full = meta.merge(content, on=key, how="left")
    else:
        full = corpus_df.copy()
        meta = split_corpus_metadata(corpus_df)

    cols = [c for c in full.columns if c != CONTENT_COLUMN] + [CONTENT_COLUMN]
    full = full[cols]

    korpus_path = korpus_dir / "korpus.csv"
    metadaten_path = korpus_dir / "metadaten.csv"
    full.to_csv(korpus_path, index=False, encoding="utf-8")
    meta.to_csv(metadaten_path, index=False, encoding="utf-8")
    return korpus_path, metadaten_path
