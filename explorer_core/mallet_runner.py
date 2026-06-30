#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.mallet_runner
===========================

Dünner Wrapper um die **MALLET-Kommandozeile** (Java). Nutzt den in
``MALLET einrichten`` gemerkten Starter-Pfad, segmentiert das Korpus (über
``corpus_segment``), ruft ``import-file`` + ``train-topics`` auf und übersetzt
MALLETs Ausgaben in **dasselbe Format** wie der sklearn-Weg
(``document-topics-distribution`` + Topic-Wörter für s02/s03).

Hinweis: MALLET kleinschreibt Tokens beim Import standardmäßig – die
Topic-Wörter erscheinen daher klein.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:  # Paketkontext
    from .corpus_segment import segment_corpus, aggregate_by_source, SOURCE_COLUMN
    from .mallet_setup import windows_safe_launcher
except Exception:  # Skript-/Testkontext
    from corpus_segment import segment_corpus, aggregate_by_source, SOURCE_COLUMN
    from mallet_setup import windows_safe_launcher


class MalletNotConfigured(RuntimeError):
    pass


def read_mallet_path(project_root: Path) -> Optional[str]:
    """Findet den MALLET-Starter – konsistent mit der Status-Anzeige der Seite
    'MALLET einrichten'.

    1. Liest den gemerkten Pfad aus ``resources/mallet/mallet_path.txt`` (sofern
       die Datei existiert UND der dort gespeicherte Pfad noch existiert).
    2. Fallback: sucht den Starter **live** in ``resources/mallet`` (über
       ``locate_mallet``). Das deckt die Fälle ab, in denen MALLET manuell in den
       Ordner gelegt wurde (ohne Klick auf 'MALLET einrichten') oder der
       gemerkte **absolute** Pfad nach einem Ordner-Umzug/-Umbenennen veraltet
       ist. Der gefundene Pfad wird zurückgemerkt.
    """
    mallet_dir = Path(project_root) / "resources" / "mallet"
    cfg = mallet_dir / "mallet_path.txt"
    if cfg.exists():
        p = cfg.read_text(encoding="utf-8").strip()
        if p and Path(p).exists():
            # Auf Windows den .bat-Starter erzwingen (der gemerkte Pfad zeigt oft
            # auf das endungslose Unix-Skript → WinError beim Start).
            safe = str(windows_safe_launcher(Path(p)))
            if safe != p:
                try:
                    cfg.write_text(safe, encoding="utf-8")
                except Exception:
                    pass
            return safe

    try:  # Paketkontext
        from .mallet_setup import locate_mallet
    except Exception:  # Skript-/Testkontext
        from mallet_setup import locate_mallet
    launcher = locate_mallet(mallet_dir)
    if launcher is not None:
        try:  # Pfad (ggf. korrigiert) zurückmerken
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(str(launcher), encoding="utf-8")
        except Exception:
            pass
        return str(launcher)
    return None


def _mallet_env(mallet_path: Path) -> dict:
    """Setzt MALLET_HOME (= …/<install>/), damit der Starter sich findet."""
    env = dict(os.environ)
    install = Path(mallet_path).parent.parent  # …/bin/mallet -> install dir
    env["MALLET_HOME"] = str(install)
    return env


def _write_import_file(texts: List[str], ids: List[str], path: Path) -> None:
    """MALLET-Importformat: eine Zeile je Dokument: id<TAB>label<TAB>text."""
    with open(path, "w", encoding="utf-8") as fh:
        for doc_id, text in zip(ids, texts):
            clean = " ".join(str(text).split())  # Tabs/Zeilenumbrüche raus
            fh.write(f"{doc_id}\tX\t{clean}\n")


# ---------------------------------------------------------------------------
# Parser für MALLET-Ausgaben (robust + isoliert testbar)
# ---------------------------------------------------------------------------
def _parse_doc_topics(path: Path, n_topics: int) -> pd.DataFrame:
    """Liest MALLETs --output-doc-topics (dichtes ODER paarweises Format).

    Zeilen: ``idx <name> <werte…>``. ``<werte>`` sind entweder ``k`` dichte
    Anteile (in Topic-Reihenfolge) oder abwechselnd ``topic anteil topic …``.
    Index der Ergebnis-Tabelle ist der Dokumentname (= unsere ID).
    """
    rows, names = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            name = parts[1]
            vals = parts[2:]
            vec = [0.0] * n_topics
            if len(vals) == n_topics:                      # dicht
                for t in range(n_topics):
                    vec[t] = float(vals[t])
            else:                                          # paarweise topic,anteil
                it = iter(vals)
                for tok, prop in zip(it, it):
                    t = int(float(tok))
                    if 0 <= t < n_topics:
                        vec[t] = float(prop)
            names.append(name)
            rows.append(vec)
    df = pd.DataFrame(rows, index=pd.Index(names, name="_id"),
                      columns=[str(i) for i in range(n_topics)])
    # auf Summe 1 normalisieren (Sicherheit)
    s = df.sum(axis=1).replace(0, 1.0)
    return df.div(s, axis=0)


def _parse_topic_keys(path: Path) -> pd.DataFrame:
    """Liest MALLETs --output-topic-keys: ``topic <alpha> wort1 wort2 …``."""
    rows, topics = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                topic = int(float(parts[0]))
                words = parts[2].split()
            else:  # Fallback: whitespace-getrennt (topic alpha w1 w2 …)
                toks = line.split()
                topic = int(float(toks[0]))
                words = toks[2:]
            topics.append(topic)
            rows.append(words)
    width = max((len(r) for r in rows), default=0)
    rows = [r + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, index=pd.Index(topics, name="Topic"),
                        columns=[str(i) for i in range(1, width + 1)])


# ---------------------------------------------------------------------------
# Lauf
# ---------------------------------------------------------------------------
def run_mallet(mallet_path: str, texts: List[str], ids: List[str],
               n_topics: int = 20, top_words: int = 100,
               num_iterations: int = 1000, random_seed: int = 42,
               optimize_interval: int = 10, alpha: float = 5.0,
               beta: float = 0.01, preserve_case: bool = False,
               workdir: Optional[Path] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Führt import-file + train-topics aus und parst die Ausgaben.

    ``preserve_case=True`` erhält die Groß-/Kleinschreibung (case-sensitive).
    Standardmäßig kleinschreibt MALLET beim Import alle Tokens.
    """
    # Auf Windows den .bat-Starter verwenden (endungsloses Unix-Skript ist per
    # subprocess nicht ausführbar → WinError 2/193).
    mallet_path = str(windows_safe_launcher(Path(mallet_path)))
    if not Path(mallet_path).exists():
        raise MalletNotConfigured(f"MALLET-Starter nicht gefunden: {mallet_path}")

    workdir = Path(workdir or tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    txt = workdir / "input.txt"
    corpus = workdir / "corpus.mallet"
    doctopics = workdir / "doc-topics.txt"
    topickeys = workdir / "topic-keys.txt"
    env = _mallet_env(Path(mallet_path))

    # Vollständigkeit der Installation prüfen: ohne mallet-deps.jar (und die
    # kompilierten Klassen) scheitert MALLET sonst mit dem kryptischen
    # "Hauptklasse … konnte nicht gefunden oder geladen werden".
    install = Path(mallet_path).parent.parent
    deps = install / "lib" / "mallet-deps.jar"
    if not deps.exists():
        raise MalletNotConfigured(
            f"MALLET-Installation unvollständig: '{deps}' fehlt. Der Download "
            "war vermutlich unvollständig – bitte auf der Seite 'MALLET "
            "einrichten' neu einrichten.")

    _write_import_file(texts, ids, txt)

    def _step(args: List[str], what: str) -> None:
        """Ruft MALLET auf und reicht bei Fehler die echte Meldung weiter."""
        res = subprocess.run([mallet_path, *args],
                             capture_output=True, text=True, env=env)
        if res.returncode != 0:
            msg = (res.stderr or res.stdout or "").strip()
            raise MalletNotConfigured(
                f"MALLET '{what}' fehlgeschlagen (Code {res.returncode}). "
                f"Meldung:\n{msg[:2000] or '(keine Ausgabe)'}")

    import_args = ["import-file", "--input", str(txt), "--output", str(corpus),
                   "--keep-sequence"]
    if preserve_case:
        # MALLET kleinschreibt Tokens beim Import standardmäßig;
        # --preserve-case erhält die Groß-/Kleinschreibung (case-sensitive).
        import_args.append("--preserve-case")
    _step(import_args, "import-file")

    _step(["train-topics", "--input", str(corpus),
           "--num-topics", str(int(n_topics)),
           "--num-iterations", str(int(num_iterations)),
           "--random-seed", str(int(random_seed)),
           "--optimize-interval", str(int(optimize_interval)),
           "--alpha", str(float(alpha)), "--beta", str(float(beta)),
           "--num-top-words", str(int(top_words)),
           "--output-doc-topics", str(doctopics),
           "--output-topic-keys", str(topickeys)], "train-topics")

    doc_topic = _parse_doc_topics(doctopics, int(n_topics))
    topic_word = _parse_topic_keys(topickeys)
    info = {"engine": "mallet", "n_topics": int(n_topics),
            "num_iterations": int(num_iterations), "random_seed": int(random_seed),
            "optimize_interval": int(optimize_interval), "alpha": float(alpha),
            "beta": float(beta), "preserve_case": bool(preserve_case),
            "workdir": str(workdir)}
    return doc_topic, topic_word, info


def fit_mallet_from_corpus(corpus_df: pd.DataFrame, mallet_path: str,
                           n_topics: int = 20, *, content_col: str = "content",
                           id_col: str = "_id", chunk_words: int = 1000,
                           min_words: int = 50, top_words: int = 100,
                           num_iterations: int = 1000, random_seed: int = 42,
                           optimize_interval: int = 10, alpha: float = 5.0,
                           beta: float = 0.01, preserve_case: bool = False,
                           aggregate: bool = True):
    """Kompletter MALLET-Weg ab Korpus-Text mit Chunking (analog sklearn).

    ``preserve_case=True`` erhält die Groß-/Kleinschreibung (case-sensitive).
    """
    seg = segment_corpus(corpus_df, chunk_words=chunk_words, id_col=id_col,
                         content_col=content_col, min_words=min_words)
    if seg.empty:
        raise ValueError("Keine Segmente erzeugt – ist die Inhaltsspalte leer?")
    texts = seg[content_col].astype(str).tolist()
    ids = seg[id_col].astype(str).tolist()

    doc_topic, topic_word, info = run_mallet(
        mallet_path, texts, ids, n_topics=n_topics, top_words=top_words,
        num_iterations=num_iterations, random_seed=random_seed,
        optimize_interval=optimize_interval, alpha=alpha, beta=beta,
        preserve_case=preserve_case)

    agg = None
    if aggregate:
        source_ids = pd.Series(seg[SOURCE_COLUMN].values, index=seg[id_col].values)
        agg = aggregate_by_source(doc_topic, source_ids)

    info.update({"n_segments": len(seg),
                 "n_sources": int(seg[SOURCE_COLUMN].nunique()),
                 "chunk_words": int(chunk_words), "min_words": int(min_words)})
    return doc_topic, topic_word, info, agg
