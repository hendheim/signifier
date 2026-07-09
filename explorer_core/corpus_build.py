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

# Default-XPaths (robuste TEI-P5-Ziele; greifen unabhängig davon, ob der Autor
# unter titleStmt oder biblFull steht). Datumswerte stehen in TEI oft im
# Attribut @when – das liest die Extraktion automatisch (siehe _attr_value).
DEFAULT_TEI_META: Dict[str, str] = {
    "title": ".//tei:teiHeader//tei:titleStmt//tei:title",
    "author_prename": ".//tei:teiHeader//tei:author//tei:forename",
    "author_surname": ".//tei:teiHeader//tei:author//tei:surname",
    "year": ".//tei:teiHeader//tei:publicationStmt//tei:date",
}
DEFAULT_TEI_CONTENT: List[str] = [".//tei:text//tei:body", ".//tei:text"]
DEFAULT_ID_XPATH = "./@xml:id"

# Elementnamen (lokal), deren Textinhalt bei der Content-Übernahme NICHT
# mitgenommen wird. TEI-``<head>`` sind Kapitel-/Abschnittsüberschriften im
# Body; sie sollen nicht in den Fließtext des Korpus wandern.
CONTENT_EXCLUDE_TAGS = {"head"}

# Wertbehaftete Attribute: TEI legt Datumsangaben u. ä. im Attribut ab, nicht im
# Elementtext (z. B. <date when="1850"/>). Diese werden als Wert gelesen, wenn
# ein Element keinen Text hat – sonst bliebe das Metadatum leer.
VALUE_ATTRS = ("when", "notBefore", "notAfter", "from", "to", "value", "key", "ref", "n")


def _attr_value(el) -> str:
    """Erster nicht-leerer Wert aus den wertbehafteten Attributen (z. B. @when)."""
    for a in VALUE_ATTRS:
        v = el.get(a)
        if v and str(v).strip():
            return str(v).strip()
    return ""


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


def _xpath_texts(root, is_lxml: bool, xpath: str, use_namespace: bool,
                 exclude_tags=None) -> List[str]:
    ns = NS if use_namespace else None

    def _text_of(el) -> str:
        """Elementtext (ganzer Teilbaum), optional ohne ``exclude_tags``."""
        if exclude_tags:
            pieces = _itertext_excluding(el, exclude_tags)
        else:
            pieces = el.itertext()
        return normalize_ws((" " if is_lxml else "").join(pieces))

    # Wurzel selbst
    if xpath in (".", "", "/"):
        txt = _text_of(root)
        return [txt] if txt else []
    out: List[str] = []
    if is_lxml:
        try:
            res = root.xpath(xpath, namespaces=ns)
        except Exception:
            return []
        for r in res:
            if hasattr(r, "itertext"):
                # Leerer Elementtext → Attributwert (z. B. <date when="1850"/>).
                out.append(_text_of(r) or _attr_value(r))
            else:
                out.append(normalize_ws(str(r)))
    else:
        try:
            res = root.findall(xpath, ns) if ns else root.findall(xpath)
        except Exception:
            return []
        for el in res:
            out.append(_text_of(el) or _attr_value(el))
    return [x for x in out if x]


def _first_text(root, is_lxml, xpath, use_namespace, exclude_tags=None) -> str:
    vals = _xpath_texts(root, is_lxml, xpath, use_namespace, exclude_tags)
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


def _itertext_excluding(el, exclude):
    """Wie ``el.itertext()``, überspringt aber komplette Teilbäume, deren
    lokaler Elementname in ``exclude`` steht (z. B. ``head``).

    Der **Tail** hinter einem ausgeschlossenen Element – also der Fließtext,
    der direkt auf eine ``<head>``-Überschrift folgt – bleibt erhalten.
    Kommentare/PIs (nicht-string Tags) werden wie bei ``itertext`` im Text
    übersprungen, ihr Tail aber übernommen.
    """
    if el.text:
        yield el.text
    for child in el:
        tag = child.tag
        if isinstance(tag, str) and _localname(tag) not in exclude:
            yield from _itertext_excluding(child, exclude)
        tail = getattr(child, "tail", None)
        if tail:
            yield tail


def path_leaf(path: str) -> str:
    """Letztes Pfadsegment ohne 'tei:'-Präfix (als Default-Spaltenname)."""
    seg = (path or "").rstrip("/").split("/")[-1]
    return seg.split(":", 1)[1] if ":" in seg else seg


def path_label(path: str) -> str:
    """Lesbarer **voller** Pfad ohne 'tei:'-Präfixe – zum Unterscheiden/Vergleichen
    gleichnamiger Ziele (z. B. mehrere 'surname' an verschiedenen Stellen).

    ``tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:author/tei:surname``
    → ``teiHeader/fileDesc/titleStmt/author/surname``
    """
    segs = (path or "").strip("/").split("/")
    return "/".join(s.split(":", 1)[1] if ":" in s else s for s in segs)


def _find_header(root, is_lxml: bool, use_namespace: bool):
    """Das ``teiHeader``-Element (dort liegen die Metadaten) oder None."""
    name = "tei:teiHeader" if use_namespace else "teiHeader"
    ns = NS if use_namespace else None
    try:
        hits = (root.xpath(f".//{name}", namespaces=ns) if is_lxml
                else root.findall(f".//{name}", ns) if ns
                else root.findall(f".//{name}"))
    except Exception:
        return None
    return hits[0] if hits else None


def discover_xml_paths(xml_text: str, use_namespace: bool = True,
                       max_paths: int = 400) -> List[Tuple[str, str]]:
    """Ermittelt die im ``teiHeader`` vorhandenen Metadaten-Pfade.

    Liefert ``(xpath, probe)`` distinkter Tag-Ketten (relativ zur Wurzel,
    ``tei:``-Präfix bei ``use_namespace``). Aufgenommen wird jedes Element mit
    nicht-leerem Text **oder** einem wertbehafteten Attribut – so erscheinen
    auch Datumsangaben wie ``<date when="1850"/>`` als auswählbares Ziel.
    Nur der ``teiHeader`` wird durchsucht (dort stehen die Metadaten); das ist
    schneller und überschwemmt die Auswahl nicht mit Body-Pfaden.
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

    scope = _find_header(root, is_lxml, use_namespace)
    if scope is None:
        scope = root

    for el in scope.iter():
        if not isinstance(getattr(el, "tag", None), str):
            continue
        text = normalize_ws(" ".join(el.itertext()) if is_lxml
                            else "".join(el.itertext()))
        probe = text or _attr_value(el)   # Attributwert, falls kein Text
        if not probe:
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
        if path not in results:
            results[path] = probe[:120]
        if len(results) >= max_paths:
            break
    return list(results.items())


def discover_xml_paths_multi(xml_texts: List[str], use_namespace: bool = True,
                             max_paths: int = 400,
                             max_files: int = 25) -> List[Tuple[str, str]]:
    """Wie ``discover_xml_paths``, aber über mehrere Dateien vereinigt.

    Pfade, die in der ersten Datei fehlen (z. B. der Autor bei einem anonymen
    Erstdokument), tauchen so trotzdem auf. Aus Performancegründen werden
    höchstens ``max_files`` Dateien analysiert.
    """
    merged: Dict[str, str] = {}
    for xml_text in xml_texts[:max_files]:
        try:
            for path, probe in discover_xml_paths(xml_text, use_namespace, max_paths):
                merged.setdefault(path, probe)
        except Exception:
            continue
    return list(merged.items())


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
        # <head>-Überschriften im Body nicht in den Fließtext übernehmen.
        content = _first_text(root, is_lxml, cx, use_namespace,
                              exclude_tags=CONTENT_EXCLUDE_TAGS)
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
# Robuste TEI-Metadaten-Entnahme
# ---------------------------------------------------------------------------
# Je Metadatum eine Prioritätsliste von XPaths, die die verschiedenen
# TEI-Auszeichnungen DESSELBEN Metadatums abdeckt. Der Autorname wird so
# erkannt, egal ob er unter titleStmt oder biblFull, als getrennte
# surname/forename-Elemente oder als zusammengesetzter persName-Text steht;
# das Jahr unabhängig davon, in welcher date-Kategorie es ausgezeichnet ist.

# --- Titel des Textes -------------------------------------------------------
TEI_TITLE_XPATHS = [
    ".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title[@type='main']",
    ".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title",
    ".//tei:teiHeader//tei:sourceDesc//tei:biblFull//tei:titleStmt//tei:title[@type='main']",
    ".//tei:teiHeader//tei:sourceDesc//tei:biblFull//tei:titleStmt//tei:title",
    ".//tei:teiHeader//tei:titleStmt//tei:title",
]
# --- Autorname: NUR der Werk-Autor (fileDesc/titleStmt/author) ---------------
# Bewusst NICHT die Quellen-Bibliografie (sourceDesc/biblFull) und NICHT
# Herausgeber/Bearbeiter (editor/respStmt) – sonst werden je nach Datei
# verschiedene Personen erfasst ("zerstreute" Metadaten). Getrennte
# surname/forename bevorzugt, sonst der persName-/author-Text.
_WORK_AUTHOR = ".//tei:teiHeader//tei:fileDesc//tei:titleStmt/tei:author"
TEI_SURNAME_XPATHS = [_WORK_AUTHOR + "//tei:surname"]
TEI_FORENAME_XPATHS = [_WORK_AUTHOR + "//tei:forename"]
TEI_PERSON_XPATHS = [_WORK_AUTHOR + "//tei:persName", _WORK_AUTHOR]

# Adelspartikel: gehören per GND-Konvention zum nameLink, nicht in den Vornamen.
# Sie werden aus dem Vornamen entfernt; der Nachname bleibt der Familienname.
NAME_PARTICLES = {"von", "vom", "van", "zu", "zur", "zum", "de", "del", "della",
                  "da", "di", "du", "le", "la", "den", "der", "ten", "ter", "af"}
# Platzhalter für „kein echter Autor" (anonyme/unbekannte Werke).
NON_AUTHOR_NAMES = {"anonymous", "anonym", "anonyme", "nn", "n.n.",
                    "unbekannt", "unknown", "o.a."}
# --- (Erst-)Veröffentlichung des WERKS (nicht Quellen-/Digitalausgabe) ------
# Diese Kategorien beziehen sich auf das Werk des Autors. Liefert eine davon
# ein Jahr, gilt die Erstveröffentlichung als belegt.
TEI_FIRST_PUB_XPATHS = [
    ".//tei:teiHeader//tei:date[@type='firstPublication']",
    ".//tei:teiHeader//tei:creation//tei:date[@type='OriginalSourcePublication']",
    ".//tei:teiHeader//tei:profileDesc//tei:creation//tei:date[@when]",
]
# Autor-gebundener Schöpfungs-Zeitraum (creation/date mit @notBefore/@notAfter,
# meist die Lebensspanne des Autors): kein präzises Jahr, aber eine Angabe „in
# Hinblick auf den Autor". @notBefore liefert das frühestmögliche Jahr und gilt
# als belegte Erstveröffentlichung, wenn keine präzise Angabe vorliegt.
TEI_CREATION_SPAN_XPATHS = [
    ".//tei:teiHeader//tei:profileDesc//tei:creation//tei:date[@notBefore or @notAfter]",
]
# Quellenausgabe (Edition, die der Digitalisierung zugrunde liegt) – nur als
# Fallback fürs Jahr. Das fileDesc/publicationStmt-Datum (= Digitalisat) zählt
# bewusst NICHT als Veröffentlichungsjahr des Werks.
TEI_SOURCE_PUB_XPATHS = [
    ".//tei:teiHeader//tei:sourceDesc//tei:biblFull//tei:publicationStmt//tei:date",
    ".//tei:teiHeader//tei:sourceDesc//tei:bibl//tei:date",
]

_YEAR_RE = re.compile(r"(1[4-9]\d\d|20\d\d)")


def extract_year(value: str) -> str:
    """Erste vierstellige Jahreszahl (1400–2099) aus einem Datumswert.

    Greift für ``@when`` (``'1924'``), ISO-Daten (``'2016-06'`` → ``'2016'``)
    und Textdaten (``'ca. 1856'``).
    """
    m = _YEAR_RE.search(value or "")
    return m.group(1) if m else ""


def _strip_particles(forename: str) -> str:
    """Adelspartikel (von/zu/van …) aus dem Vornamen entfernen."""
    return " ".join(w for w in forename.split()
                    if w.lower().strip(".") not in NAME_PARTICLES)


def split_person_name(text: str) -> Tuple[str, str]:
    """``'Nachname, Vorname'`` **oder** ``'Vorname Nachname'`` → ``(Nachname, Vorname)``.

    Erkennt beide im Korpus vorkommenden Schreibweisen:
      - komma-getrennt ``'Ringelnatz, Joachim'`` → ``('Ringelnatz', 'Joachim')``
      - natürliche Reihenfolge ``'Joseph von Eichendorff'`` → ``('Eichendorff', 'Joseph')``
    Adelspartikel werden aus dem Vornamen entfernt (GND-Konvention: nameLink).
    """
    text = normalize_ws(text)
    if not text:
        return "", ""
    if "," in text:
        surname, _, forename = text.partition(",")
        surname, forename = surname.strip(), forename.strip()
    else:
        parts = text.split()
        if len(parts) == 1:
            return parts[0], ""
        surname, forename = parts[-1], " ".join(parts[:-1])
    return surname, _strip_particles(forename)


def is_real_author(name: str) -> bool:
    """True, wenn ``name`` eine identifizierte Person ist (kein „Anonymous"/leer)."""
    t = normalize_ws(name).lower().strip().strip(".")
    return bool(t) and t not in NON_AUTHOR_NAMES and not t.startswith("anonym")


def first_nonempty(root, is_lxml, xpaths: List[str], use_namespace: bool) -> str:
    """Erster nicht-leerer Treffer aus mehreren XPaths – federt ab, dass
    dieselbe Angabe je nach Datei woanders (anders kategorisiert) steht."""
    for xp in xpaths:
        val = _first_text(root, is_lxml, xp, use_namespace)
        if val:
            return val
    return ""


def extract_tei_metadata(xml_text: str, use_namespace: bool = True) -> Dict[str, str]:
    """Entnimmt die vier Kern-Metadaten robust aus einem TEI-Header.

    Erkennt die verschiedenen TEI-Auszeichnungen je Metadatum und liefert ein
    Dict mit:

    - ``title``            – Titel des Textes
    - ``author_surname``   – Nachname des Autors
    - ``author_forename``  – Vorname des Autors
    - ``year``             – Veröffentlichungsjahr; bevorzugt die
      Erstveröffentlichung des Werks (präzises Jahr oder, ersatzweise, das
      früheste Jahr des autor-gebundenen creation-Bereichs), sonst die
      Quellenausgabe
    - ``first_publication_year`` – das Erstveröffentlichungsjahr, sofern das
      Werk eine entsprechende Angabe trägt – präzise oder als frühestes Jahr des
      autor-gebundenen creation-Bereichs (sonst leer). Damit lässt sich ein
      „bereinigtes" Korpus bilden, das nur Werke mit belegter Erstveröffentlichung
      enthält.
    """
    root, is_lxml = _parse_root(xml_text, use_namespace)

    title = first_nonempty(root, is_lxml, TEI_TITLE_XPATHS, use_namespace)

    surname = first_nonempty(root, is_lxml, TEI_SURNAME_XPATHS, use_namespace)
    forename = first_nonempty(root, is_lxml, TEI_FORENAME_XPATHS, use_namespace)
    if surname and forename:
        forename = _strip_particles(forename)
    else:
        name = first_nonempty(root, is_lxml, TEI_PERSON_XPATHS, use_namespace)
        if name:
            surname, forename = split_person_name(name)

    # Nur echte Autoren: Platzhalter (Anonymous/unbekannt) → leere Namensfelder.
    real_author = is_real_author((surname + " " + forename).strip())
    if not real_author:
        surname = forename = ""

    # Erstveröffentlichung: präzises Jahr bevorzugt, sonst der autor-gebundene
    # creation-Bereich (@notBefore = frühestmöglich). Beides gilt als belegt.
    precise_year = extract_year(
        first_nonempty(root, is_lxml, TEI_FIRST_PUB_XPATHS, use_namespace))
    span_year = extract_year(
        first_nonempty(root, is_lxml, TEI_CREATION_SPAN_XPATHS, use_namespace))
    source_year = extract_year(
        first_nonempty(root, is_lxml, TEI_SOURCE_PUB_XPATHS, use_namespace))

    first_publication_year = precise_year or span_year
    return {
        "title": title,
        "author_surname": surname,
        "author_forename": forename,
        "year": first_publication_year or source_year,
        "first_publication_year": first_publication_year,
        "real_author": real_author,
    }


# ---------------------------------------------------------------------------
# Provenienz: Ersteller der XML-Datei und zugehöriges Projekt
# ---------------------------------------------------------------------------
# Ersteller = die Person, die die Metadaten erfasst hat ("Erfassung der
# Metadaten", im KOLIMO-Workflow der kolimo-staff-respStmt). Das Projekt steht
# im publisher der digitalen Ausgabe (z. B. „Kolimo – Korpus der literarischen
# Moderne").
TEI_PROJECT_XPATHS = [
    ".//tei:teiHeader//tei:fileDesc//tei:publicationStmt//tei:publisher",
]
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _project_name(publisher: str) -> str:
    """Projektname aus dem publisher (E-Mail-Vorspann und Anschrift entfernen)."""
    pub = normalize_ws(publisher)
    low = pub.lower()
    if "kolimo" in low:
        name = pub[low.index("kolimo"):]
        for cut in (" Seminar", " Käthe", ","):   # Anschrift abschneiden
            i = name.find(cut)
            if i > 0:
                name = name[:i]
        return name.strip()
    return " ".join(w for w in pub.split() if "@" not in w).strip()


def _xml_creator(root, is_lxml: bool) -> str:
    """Ersteller (Metadaten-Erfasser): Name-Text → Name-``@xml:id`` →
    ``revisionDesc/@who``. Oft ist nur ein Kürzel (z. B. ``markus``) hinterlegt;
    fehlt jede Personenangabe, bleibt das Feld leer."""
    ns = {"tei": TEI_NS_URI}
    if is_lxml:
        for rs in root.xpath(".//tei:respStmt[tei:resp[contains(., 'Erfassung der "
                             "Metadaten')]]", namespaces=ns):
            nm = rs.find("tei:name", ns)
            if nm is not None:
                t = normalize_ws(" ".join(nm.itertext()))
                if t:
                    return t
                if nm.get(_XML_ID):
                    return nm.get(_XML_ID)
        for ch in root.xpath(".//tei:revisionDesc//tei:change[@who]", namespaces=ns):
            who = (ch.get("who") or "").lstrip("#").strip()
            if who:
                return who
        return ""
    # ElementTree-Fallback: nur der Name-Text
    for rs in root.findall(".//tei:respStmt", ns):
        resp = rs.find("tei:resp", ns)
        if resp is not None and "Erfassung der Metadaten" in "".join(resp.itertext()):
            nm = rs.find("tei:name", ns)
            if nm is not None and normalize_ws("".join(nm.itertext())):
                return normalize_ws("".join(nm.itertext()))
    return ""


def extract_tei_provenance(xml_text: str, use_namespace: bool = True) -> Dict[str, str]:
    """Ersteller der XML-Datei (Metadaten-Erfassung) und zugehöriges Projekt.

    Liefert ``{"xml_creator": …, "project": …}``.
    """
    root, is_lxml = _parse_root(xml_text, use_namespace)
    project = _project_name(
        first_nonempty(root, is_lxml, TEI_PROJECT_XPATHS, use_namespace))
    return {"xml_creator": _xml_creator(root, is_lxml), "project": project}


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
