#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite 8: Korpus verarbeiten
===========================

Startet die bestehende NLP-Pipeline (Schritte s01–s07) aus dem Dashboard.

Die Pipeline wird NICHT umgeschrieben: Diese Seite ruft den Runner
``run_pipeline.py`` als Subprozess auf (siehe ``explorer_core.pipeline_runner``),
der die vorhandenen Funktionen ``load_config`` / ``run_pipeline_with_cfg``
nutzt. Derselbe Lauf ist auch per Doppelklick über ``start_pipeline.bat`` /
``start_pipeline.sh`` möglich.

Voraussetzung: Auf der Startseite ist der Projektordner gesetzt (er enthält
``config/`` mit der TOML sowie ``konfig/``, ``output/``, ``resources/``).
"""

from pathlib import Path

import streamlit as st

from ui_helpers import get_store, show_error, APP_DIR
from explorer_core.pipeline_runner import (
    STEP_LABELS, available_configs, stream_pipeline, list_output_tree,
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die Word2Vec-Hyperparameter-Voreinstellung
# ---------------------------------------------------------------------------

def _suggest_w2v_params(num_tokens: int) -> dict:
    """Größenabhängige Word2Vec-Defaults.

    Nutzt – wenn verfügbar – dieselbe Funktion wie die Pipeline
    (``nlp_pipeline.s07_word_vectors.calculate_word2vec_params``), damit Vorschau und
    tatsächlicher Lauf identisch sind. Fällt auf eine lokale, wertgleiche
    Kopie zurück, falls der Import (z. B. gensim) hier nicht möglich ist.
    """
    try:  # bevorzugt die echte Pipeline-Funktion
        from nlp_pipeline.s07_word_vectors import calculate_word2vec_params
        return calculate_word2vec_params(num_tokens)
    except Exception:
        base = {"workers": 4, "sg": 1, "hs": 0, "sample": 1e-3, "seed": 42}
        if num_tokens < 100_000:
            base.update(vector_size=100, window=5, min_count=2, negative=10, epochs=40, _category="KLEIN")
        elif num_tokens < 1_000_000:
            base.update(vector_size=150, window=5, min_count=5, negative=5, epochs=20, _category="MITTEL")
        else:
            base.update(vector_size=300, window=5, min_count=10, negative=5, epochs=15, _category="GROSS")
        return base


def _detect_delimiter_safe(path: Path) -> str:
    """Trennzeichen robust erkennen – ohne Abhängigkeit vom Importpfad.

    Versucht zuerst die projektweite (gehärtete) Erkennung; schlägt der Import
    fehl (im Dashboard liegt ``pipeline_utils`` oft nicht unter dem bloßen
    Namen auf dem sys.path), nutzt sie eine selbständige, konsistenzbasierte
    Erkennung statt blind auf ";" zu setzen.
    """
    for modpath in ("explorer_core.data_store", "pipeline_utils"):
        try:
            mod = __import__(modpath, fromlist=["detect_delimiter"])
            return mod.detect_delimiter(path)
        except Exception:
            continue
    # Selbständiger Fallback (gleiche Logik wie pipeline_utils.detect_delimiter)
    import csv, io
    from collections import Counter
    candidates = [";", ",", "\t", "|"]
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(131072)
    except Exception:
        return ";"
    if not sample.strip():
        return ";"
    best, best_key = None, None
    for d in candidates:
        try:
            rows = [r for r in csv.reader(io.StringIO(sample), delimiter=d) if r][:25]
        except csv.Error:
            continue
        if len(rows) < 2:
            continue
        counts = [len(r) for r in rows]
        modal = Counter(counts).most_common(1)[0][0]
        if modal < 2:
            continue
        key = (round(sum(c == modal for c in counts) / len(counts), 4), modal)
        if best_key is None or key > best_key:
            best_key, best = key, d
    return best or ";"


def _count_corpus_tokens(input_dir: Path, pattern: str,
                         extra_dirs: tuple = ()) -> dict:
    """Zählt Tokens/Dokumente der korpus_gen-Grundlage für die Voreinstellung.

    Bevorzugt den Gesamtkorpus (Datei ohne ``_JJJJ-JJJJ``), sonst die größte
    Intervall-Datei. Sucht rekursiv unter ``input_dir`` und optionalen
    ``extra_dirs`` (z. B. dem output_dir). Das Trennzeichen wird
    importunabhängig erkannt (siehe ``_detect_delimiter_safe``).

    Gibt ein Diagnose-Dict zurück: tokens, docs, file, sep, content_col,
    searched (durchsuchte Verzeichnisse), n_files.
    """
    import re
    import pandas as pd

    diag = {"tokens": 0, "docs": 0, "file": "", "sep": "", "content_col": None,
            "searched": [], "n_files": 0}

    search_dirs = [d for d in (input_dir, *extra_dirs) if d is not None]
    files: list = []
    for d in search_dirs:
        diag["searched"].append(str(d))
        if d.exists():
            files.extend(sorted(d.rglob(pattern)))
    # Duplikate (gleicher Pfad) entfernen, Reihenfolge wahren
    seen = set()
    files = [f for f in files if not (f.resolve() in seen or seen.add(f.resolve()))]
    diag["n_files"] = len(files)
    if not files:
        return diag

    interval_re = re.compile(r"_\d{4}-\d{4}")
    gesamt = [f for f in files if not interval_re.search(f.stem)]
    chosen = gesamt[0] if gesamt else max(files, key=lambda p: p.stat().st_size)
    diag["file"] = chosen.name

    sep = _detect_delimiter_safe(chosen)
    diag["sep"] = sep
    try:
        df = pd.read_csv(chosen, encoding="utf-8", sep=sep)
    except Exception:
        return diag

    content_col = None
    try:  # bevorzugt die projektweite Content-Spalten-Erkennung
        from nlp_pipeline.pipeline_utils import identify_content_column
        content_col = identify_content_column(df)
    except Exception:
        pass
    if content_col is None:
        for cand in ("content_gen", "content_stop", "content_lem",
                     "content_min", "content", "text", "clean_text"):
            if cand in df.columns:
                content_col = cand
                break
    diag["content_col"] = content_col
    diag["docs"] = len(df)
    if content_col is None:
        return diag

    col = df[content_col]
    if isinstance(col, pd.DataFrame):          # doppelte content-Spaltennamen → erste
        col = col.iloc[:, 0]
    texts = col.fillna("").astype(str)         # NaN/Zahlen sicher zu Strings
    diag["tokens"] = int(texts.map(lambda x: len(str(x).split())).sum())
    return diag

st.set_page_config(page_title="Korpus verarbeiten", layout="wide")
st.title("⚙️ Korpus verarbeiten")
st.caption("Vorverarbeitungs-Pipeline ausführen: Vorverarbeitung, Vokabular, "
           "Statistik, POS-Tagging, Matrizen, TF-IDF-Ranglisten. (Die "
           "Wortvektoren entstehen auf der Seite 'Wortvektormodell erstellen'.)")

store = get_store()
project_root = Path(store.project_root)
runner_path = APP_DIR / "run_pipeline.py"

st.info(
    "Diese Seite startet die vorhandene Pipeline – sie schreibt Ergebnisse "
    "nach `output/`. Lange Schritte (Lemmatisierung, POS-Tagging) können einige "
    "Minuten dauern. Das Log erscheint live."
)

# ---------------------------------------------------------------------------
# 1) Konfiguration wählen
# ---------------------------------------------------------------------------
st.subheader("1 · Konfiguration")

configs = available_configs(project_root)
if not configs:
    st.warning(
        f"Keine TOML-Konfiguration im Projektordner gefunden "
        f"(`{project_root}`). Erwartet z. B. `config/fadelive_v3.toml`. "
        "Pfad ggf. auf der Startseite korrigieren."
    )
    st.stop()

# fadelive_v3.toml als Default vorauswählen, falls vorhanden
names = [str(p.relative_to(project_root)) for p in configs]
default_idx = next((i for i, n in enumerate(names) if n.endswith("fadelive_v3.toml")), 0)
chosen_name = st.selectbox("TOML-Konfiguration", names, index=default_idx)
config_path = project_root / chosen_name

# ---------------------------------------------------------------------------
# 2) Schritte wählen
# ---------------------------------------------------------------------------
st.subheader("2 · Schritte")

# Schritt 10 (Word2Vec/s07) ist auf eine eigene Seite ausgelagert.
all_steps = [s for s in STEP_LABELS.keys() if s != "10"]
run_all = st.checkbox("Alle Vorverarbeitungsschritte ausführen", value=True)

if run_all:
    steps = all_steps
    st.caption("Es laufen alle Schritte in Reihenfolge.")
else:
    chosen = st.multiselect(
        "Einzelne Schritte auswählen",
        options=all_steps,
        default=all_steps,
        format_func=lambda s: f"{s} · {STEP_LABELS[s]}",
    )
    # Reihenfolge 1→10 erzwingen, egal wie ausgewählt
    steps = [s for s in all_steps if s in chosen]

with st.expander("Was wird in den Schritten erzeugt?"):
    for s in all_steps:
        st.markdown(f"- **{s}** — {STEP_LABELS[s]}")

# ---------------------------------------------------------------------------
# 3) Starten
# ---------------------------------------------------------------------------
st.subheader("3 · Starten")

# Optional: Intervalle für Vokabular (Schritt 2) und Word2Vec (Schritte 5/10).
# Leer = keine Intervalle erzwingen (dynamische Erzeugung bzw. Werte aus der TOML).
intervals_raw = st.text_input(
    "Intervalle (optional, kommagetrennt)",
    value="",
    help="z. B. 1782-1852, 1853-1864, 1865-1876, 1877-1891. "
         "Wirkt hier auf das Vokabular (Schritt 2). Die Word2Vec-Modelle je "
         "Intervall werden auf der Seite 'Wortvektormodell erstellen' trainiert. "
         "Leer lassen = ohne Intervalle.",
)
intervals = [s.strip() for s in intervals_raw.replace(";", ",").split(",") if s.strip()] or None
if intervals:
    st.caption("Aktive Intervalle: " + ", ".join(intervals))

# Word2Vec (Schritt 10/s07) ist auf eine eigene Seite ausgelagert -
# hier werden keine Word2Vec-Parameter mehr gesetzt.
w2v_params = None

if not runner_path.exists():
    st.error(f"Runner nicht gefunden: {runner_path}. "
             "Liegt `run_pipeline.py` im Dashboard-Ordner?")
    st.stop()

if not steps:
    st.warning("Bitte mindestens einen Schritt auswählen.")

start = st.button("▶️ Pipeline starten", type="primary", disabled=not steps)

if start:
    log_lines: list[str] = []
    log_box = st.empty()
    returncode = None
    try:
        with st.status("Pipeline läuft …", expanded=True) as status:
            for kind, payload in stream_pipeline(
                runner_path=runner_path,
                project_root=project_root,
                config_path=config_path,
                steps=steps,
                intervals=intervals,
                w2v_params=w2v_params,
            ):
                if kind == "log":
                    log_lines.append(str(payload))
                    # Nur die letzten ~400 Zeilen anzeigen (Performance)
                    log_box.code("\n".join(log_lines[-400:]), language="text")
                elif kind == "done":
                    returncode = payload
            if returncode == 0:
                status.update(label="Pipeline abgeschlossen ✅", state="complete")
            else:
                status.update(label=f"Pipeline mit Fehler beendet (Code {returncode})",
                              state="error")
    except Exception as exc:  # einheitliche, laienfreundliche Fehleranzeige
        show_error(exc)
    else:
        if returncode == 0:
            st.success("Fertig. Ergebnisse liegen in `output/`.")
        else:
            st.error("Die Pipeline ist nicht sauber durchgelaufen. Bitte das "
                     "Log oben prüfen (häufig: fehlende Eingabedateien, "
                     "falsche Pfade in der TOML, fehlende Modelle).")

        # Vollständiges Log zum Download
        st.download_button(
            "⬇️ Vollständiges Log herunterladen",
            "\n".join(log_lines).encode("utf-8"),
            file_name="pipeline_log.txt",
            mime="text/plain",
        )

        # Ergebnisübersicht
        outputs = list_output_tree(project_root)
        if outputs:
            with st.expander(f"Erzeugte Dateien in output/ ({len(outputs)})",
                            expanded=True):
                st.code("\n".join(outputs), language="text")

# ---------------------------------------------------------------------------
# 4) Eigennamen tilgen (Variante "-n") – Erkennen → Prüfen → Tilgen
# ---------------------------------------------------------------------------
st.divider()
st.subheader("4 · Eigennamen tilgen (Variante `-n`)")
st.caption(
    "Kandidaten sind die als Eigenname (`PROPN`) getaggten Ausdrücke – entweder "
    "aus der vorhandenen POS-Liste (Top-5000 aus Schritt 4) geladen oder mit "
    "einer neu erstellten POS-Liste (wählbare Anzahl Top-Ausdrücke). Häkchen für "
    "die Ausdrücke setzen, die in der `-n`-Verarbeitung getilgt werden sollen. Es "
    "wird ab `corpus_stop` gearbeitet (keine Neu-Lemmatisierung). Ergebnis sind "
    "`output/processed_corpus/korpus_stop_ner.csv` und die `…-n`-Ordner. "
    "Einige gängige Ausdrücke wie „Frau“ oder „Mann“ sind durch eine Schutzliste "
    "von der Tilgung ausgenommen."
)

processed_dir = project_root / "output" / "processed_corpus"
stop_csv = processed_dir / "korpus_stop.csv"
cand_csv = processed_dir / "name_candidates.csv"
terms_csv = processed_dir / "name_exclude.csv"

if not stop_csv.exists():
    st.info("Voraussetzung: `output/processed_corpus/korpus_stop.csv` "
            "(und `korpus_gen.csv` für die Wortvektoren). Bitte zuerst die "
            "Pipeline oben laufen lassen.")


def _run_n(label: str, **kwargs) -> int | None:
    """Startet einen Runner-Lauf mit Live-Log; gibt den Returncode zurück."""
    log_lines: list[str] = []
    log_box = st.empty()
    rc = None
    try:
        with st.status(label, expanded=True) as status:
            for kind, payload in stream_pipeline(
                    runner_path=runner_path, project_root=project_root,
                    config_path=config_path, **kwargs):
                if kind == "log":
                    log_lines.append(str(payload))
                    log_box.code("\n".join(log_lines[-400:]), language="text")
                elif kind == "done":
                    rc = payload
            status.update(label=(f"{label} – fertig ✅" if rc == 0
                                 else f"{label} – Fehler (Code {rc})"),
                          state="complete" if rc == 0 else "error")
    except Exception as exc:
        show_error(exc)
    return rc

# Schritt 1: Kandidaten erzeugen – aus den als PROPN getaggten Ausdrücken.
# Zwei Wege: vorhandene POS-Liste laden (sofort) oder neue POS-Liste erstellen.
OPT_LOAD = "Kandidaten für Namen aus den Top-5000 Ausdrücken laden"
OPT_MAKE = "Neue POS-Liste erstellen und Kandidaten auswählen"
mode = st.radio("Kandidaten erzeugen", [OPT_LOAD, OPT_MAKE],
                index=0, key="cand_mode")

if mode == OPT_LOAD:
    # Vorhandene POS-Liste (Schritt 4) als Quelle; Standarddatei voreinstellen.
    default_pos = project_root / "output" / "vocabular" / "vocab_top5000_stop_pos.csv"
    pos_files = sorted((project_root / "output").glob("vocabular*/*_pos.csv"))
    opts = [str(p.relative_to(project_root)) for p in pos_files]
    default_rel = str(default_pos.relative_to(project_root))
    if default_rel not in opts and default_pos.exists():
        opts.insert(0, default_rel)
    if not opts:
        st.info("Keine POS-Liste gefunden. Bitte zuerst die Pipeline bis "
                "Schritt 4 (POS-Tagging) laufen lassen oder „Neue POS-Liste "
                "erstellen“ wählen.")
    else:
        sel_pos = st.selectbox("POS-Liste", opts,
                               index=opts.index(default_rel) if default_rel in opts else 0,
                               help="Alle als PROPN getaggten Ausdrücke werden geladen.")
        if st.button("🔎 Kandidaten laden", disabled=not (project_root / sel_pos).exists()):
            from nlp_pipeline.s08_name_removal import propn_candidates_from_pos
            try:
                cand = propn_candidates_from_pos(project_root / sel_pos)
                cand.to_csv(cand_csv, index=False, encoding="utf-8")
                st.success(f"{len(cand)} PROPN-Kandidaten aus `{sel_pos}` geladen.")
            except Exception as exc:
                show_error(exc)
else:
    # Neue POS-Liste mit wählbarer Anzahl Top-Ausdrücke (gleiche Grundlage wie
    # das reguläre POS-Tagging: vocab_full_stop.json) – taggt nur diese Wörter.
    n_pos = st.number_input("POS-Tagging für wie viele Top-Ausdrücke?",
                            min_value=100, max_value=50000, value=5000, step=500,
                            help="Grundlage ist dieselbe wie für das POS-Tagging "
                                 "(vocab_full_stop.json).")
    if st.button("🧠 Neue POS-Liste erstellen & Kandidaten erzeugen",
                 disabled=not stop_csv.exists()):
        _run_n(f"POS-Liste (Top-{int(n_pos)}) wird erstellt …",
               detect_names=True, make_pos=True, top_terms=int(n_pos))

# Schritt 2: Kandidaten prüfen (Häufigkeit + Checkbox, alle anfangs abgewählt)
if cand_csv.exists():
    import pandas as pd
    cand = pd.read_csv(cand_csv)
    st.markdown(f"**{len(cand)} Eigennamen-Kandidaten (PROPN)** – anhaken, was "
                "getilgt werden soll:")
    alle = st.checkbox("Alle auswählen / abwählen", value=False,
                       key="name_cand_all")
    cand.insert(0, "tilgen", alle)
    edited = st.data_editor(
        cand, use_container_width=True, height=420, hide_index=True,
        column_config={
            "tilgen": st.column_config.CheckboxColumn("tilgen", default=False),
            "term": st.column_config.TextColumn("Ausdruck", disabled=True),
            "frequency": st.column_config.NumberColumn("Frequenz", disabled=True),
        },
        key=f"name_cand_editor_{alle}",
    )
    selected = [str(t) for t, keep in zip(edited["term"], edited["tilgen"]) if keep]
    st.caption(f"{len(selected)} von {len(edited)} Ausdrücken ausgewählt.")

    # Schritt 3: Ausgewählte tilgen
    if st.button("🧹 Ausgewählte Ausdrücke tilgen", type="primary",
                 disabled=not selected):
        pd.DataFrame({"term": selected}).to_csv(terms_csv, index=False,
                                                encoding="utf-8")
        rc = _run_n("Ausgewählte Ausdrücke werden getilgt …",
                    remove_names=True, terms_file=terms_csv)
        if rc == 0:
            st.success("Fertig. Bereinigte Ergebnisse in `output/…-n/` und "
                       "`output/processed_corpus/korpus_stop_ner.csv`.")
        elif rc is not None:
            st.error("Die Tilgung ist nicht sauber durchgelaufen – Log prüfen.")
else:
    st.caption("Noch keine Kandidatenliste – zuerst oben Kandidaten laden bzw. "
               "eine POS-Liste erstellen.")