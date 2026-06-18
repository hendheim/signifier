#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seite: Wortvektormodell erstellen
=================================

Eigenständige Seite für die Word2Vec-Modellbildung (Pipeline-Schritt s07).
Sie wurde aus der allgemeinen Korpus-Verarbeitung herausgelöst: dort laufen
jetzt nur noch die Vorverarbeitungsschritte (s01-s06), die Wortvektoren
entstehen hier.

Die Seite ruft denselben Runner (``run_pipeline.py``) als Subprozess auf,
aber nur mit Schritt 10 (s07). Sie setzt voraus, dass die Vorverarbeitung
(insbesondere Schritt 5 = ``korpus_gen``) bereits gelaufen ist. Nach dem Lauf
kann das Modell direkt mit einer Qualitätsmetrik geprüft werden
(``explorer_core.wvm_metrics``).
"""

from pathlib import Path

import streamlit as st

from ui_helpers import get_store, get_models, show_error, APP_DIR
from explorer_core.pipeline_runner import (
    available_configs, stream_pipeline, list_output_tree,
)
from explorer_core import wvm_metrics as wm


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die Word2Vec-Voreinstellung (identisch zur Pipeline)
# ---------------------------------------------------------------------------

def _suggest_w2v_params(num_tokens: int) -> dict:
    """Größenabhängige Word2Vec-Defaults.

    Nutzt - wenn verfügbar - dieselbe Funktion wie die Pipeline
    (``nlp_pipeline.s07_word_vectors.calculate_word2vec_params``), damit
    Vorschau und tatsächlicher Lauf identisch sind.
    """
    try:
        from nlp_pipeline.s07_word_vectors import calculate_word2vec_params
        return calculate_word2vec_params(num_tokens)
    except Exception:
        base = {"workers": 4, "sg": 1, "hs": 0, "sample": 1e-3, "seed": 42}
        if num_tokens < 1_000_000:
            base.update(vector_size=100, window=5, min_count=2, negative=10,
                        epochs=40, _category="KLEIN")
        elif num_tokens < 10_000_000:
            base.update(vector_size=150, window=5, min_count=5, negative=5,
                        epochs=20, _category="MITTEL")
        else:
            base.update(vector_size=300, window=5, min_count=10, negative=5,
                        epochs=15, _category="GROSS")
        return base


def _detect_delimiter_safe(path: Path) -> str:
    """Trennzeichen robust erkennen (App-Funktion, mit Fallback)."""
    for modpath in ("explorer_core.data_store", "pipeline_utils"):
        try:
            mod = __import__(modpath, fromlist=["detect_delimiter"])
            return mod.detect_delimiter(path)
        except Exception:
            continue
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
    """Zählt Tokens/Dokumente der korpus_gen-Grundlage für die Voreinstellung."""
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
    try:
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
    texts = df[content_col].astype(str)
    diag["tokens"] = int(texts.apply(lambda x: len(x.split())).sum())
    return diag


def _parse_pairs(text: str):
    """Parst Zeilen 'w1, w2, score' in eine Liste (w1, w2, float(score))."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (",", ";", "\t"):
            if sep in line:
                parts = [x.strip() for x in line.split(sep)]
                break
        else:
            parts = line.split()
        if len(parts) >= 3:
            try:
                pairs.append((parts[0], parts[1], float(parts[2])))
            except ValueError:
                continue
    return pairs


# ---------------------------------------------------------------------------
# Seite
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Wortvektormodell erstellen", layout="wide")
st.title("🧮 Wortvektormodell erstellen")
st.caption("Word2Vec-Modellbildung (Pipeline-Schritt s07) als eigener Schritt - "
           "getrennt von der Vorverarbeitung. Setzt voraus, dass die "
           "Vorverarbeitung (insbesondere korpus_gen) bereits gelaufen ist.")

store = get_store()
project_root = Path(store.project_root)
runner_path = APP_DIR / "run_pipeline.py"

# 1) Konfiguration --------------------------------------------------------
st.subheader("1 · Konfiguration")
configs = available_configs(project_root)
if not configs:
    st.warning(f"Keine TOML-Konfiguration im Projektordner gefunden "
               f"(`{project_root}`). Pfad ggf. auf der Startseite korrigieren.")
    st.stop()
names = [str(p.relative_to(project_root)) for p in configs]
default_idx = next((i for i, n in enumerate(names) if n.endswith("fadelive_v3.toml")), 0)
chosen_name = st.selectbox("TOML-Konfiguration", names, index=default_idx)
config_path = project_root / chosen_name

# 2) Intervalle -----------------------------------------------------------
st.subheader("2 · Intervalle (optional)")
intervals_raw = st.text_input(
    "Intervalle (optional, kommagetrennt)", value="",
    help="z. B. 1782-1852, 1853-1864. Je Intervall wird ein eigenes "
         "Word2Vec-Modell trainiert (für diachrone Vergleiche). "
         "Leer lassen = nur ein Modell über das Gesamtkorpus.")
intervals = [s.strip() for s in intervals_raw.replace(";", ",").split(",")
             if s.strip()] or None
if intervals:
    st.caption("Aktive Intervalle: " + ", ".join(intervals))

# 3) Hyperparameter -------------------------------------------------------
st.subheader("3 · Hyperparameter")
import tomllib
input_dir = output_dir = None
pattern = "korpus_gen*.csv"
try:
    with config_path.open("rb") as f:
        _cfg = tomllib.load(f)
    c10 = _cfg.get("step10_s07_word_vector_model", {})
    if c10.get("input_dir"):
        input_dir = (project_root / c10["input_dir"]).resolve()
    if c10.get("output_dir"):
        output_dir = (project_root / c10["output_dir"]).resolve()
    pattern = c10.get("pattern", pattern)
except Exception:
    pass

diag = {"tokens": 0, "docs": 0, "file": "", "sep": "", "content_col": None,
        "searched": [], "n_files": 0}
if input_dir is not None:
    diag = _count_corpus_tokens(input_dir, pattern, extra_dirs=(output_dir,))
tokens, docs, fname = diag["tokens"], diag["docs"], diag["file"]
suggested = _suggest_w2v_params(tokens)
cat = str(suggested.get("_category", "?"))

if tokens > 0:
    st.caption(f"Grundlage `{fname}` (Trennzeichen {diag['sep']!r}): "
               f"~{tokens:,} Tokens, {docs:,} Dokumente → Kategorie **{cat}**. "
               "Felder unten sind entsprechend voreingestellt (Quellen: "
               "Altszyler et al. 2017, HistWords/COHA, gensim-Defaults).")
else:
    st.caption("Noch keine `korpus_gen`-Datei gefunden (entsteht in der "
               "Vorverarbeitung, Schritt 5). Voreinstellung anhand der "
               "Default-Kategorie; beim Lauf wird die Token-Zahl neu bestimmt.")

w2v_params = None
auto = st.checkbox("Automatisch (empfohlen, größenabhängig)", value=True,
                   key="w2v_auto")
if not auto:
    cA, cB, cC = st.columns(3)
    vsize = cA.number_input(
        "vector_size", 16, 600, int(suggested["vector_size"]), 2, key="w2v_vs",
        help="Dimension der Wortvektoren. Kleiner (50-100) bei kleinem Korpus, "
             "~300 bei großem; mehr Dimensionen erfassen feinere "
             "Bedeutungsunterschiede, brauchen aber mehr Daten.")
    window = cB.number_input(
        "window", 1, 20, int(suggested["window"]), 1, key="w2v_win",
        help="Größe des Kontextfensters (Wörter links und rechts). ~5 ist "
             "Standard; größer erfasst eher thematische, kleiner eher "
             "syntaktische Nähe.")
    mincount = cC.number_input(
        "min_count", 1, 100, int(suggested["min_count"]), 1, key="w2v_mc",
        help="Mindesthäufigkeit, die ein Wort haben muss, um ins Modell "
             "aufgenommen zu werden. Klein (2-5) bei kleinem Korpus, sonst "
             "wird das Vokabular zu klein.")
    cD, cE, cF = st.columns(3)
    negative = cD.number_input(
        "negative", 0, 30, int(suggested["negative"]), 1, key="w2v_neg",
        help="Anzahl der Negativbeispiele beim Negative Sampling. Mehr (10-20) "
             "stabilisiert kleine Korpora; bei großen reichen 2-5 (Mikolov et al.).")
    epochs = cE.number_input(
        "epochs", 1, 200, int(suggested["epochs"]), 1, key="w2v_ep",
        help="Anzahl der Trainingsdurchläufe über das Korpus. Mehr (30-40) bei "
             "kleinen Korpora für stabilere Vektoren; bei großen genügen weniger.")
    algo = cF.selectbox(
        "Algorithmus", ["Skip-gram (sg=1)", "CBOW (sg=0)"],
        index=0 if int(suggested.get("sg", 1)) == 1 else 1, key="w2v_sg",
        help="Skip-gram (sg=1, SGNS) sagt den Kontext aus dem Wort vorher - "
             "Standard für kleine und diachrone Korpora; CBOW (sg=0) ist "
             "umgekehrt und schneller bei großen Korpora.")
    sample = st.select_slider(
        "sample (Subsampling häufiger Wörter)", options=[1e-5, 1e-4, 1e-3],
        value=float(suggested.get("sample", 1e-3)),
        format_func=lambda v: f"{v:.0e}", key="w2v_sample",
        help="Subsampling-Schwelle für sehr häufige Wörter: kleinere Werte "
             "unterdrücken Hochfrequenzwörter stärker. 1e-3 ist der gensim-Default.")
    w2v_params = {
        "vector_size": int(vsize), "window": int(window),
        "min_count": int(mincount), "negative": int(negative),
        "epochs": int(epochs), "sg": 1 if algo.startswith("Skip") else 0,
        "sample": float(sample),
    }
    st.caption("Diese Werte überschreiben die automatische Voreinstellung für "
               "alle Modelle (Gesamtkorpus und Intervalle).")

# 4) Starten --------------------------------------------------------------
st.subheader("4 · Modell trainieren")
if not runner_path.exists():
    st.error(f"Runner nicht gefunden: {runner_path}.")
    st.stop()

if st.button("▶️ Wortvektormodell trainieren (s07)", type="primary"):
    log_lines: list[str] = []
    log_box = st.empty()
    returncode = None
    try:
        with st.status("Training läuft …", expanded=True) as status:
            for kind, payload in stream_pipeline(
                runner_path=runner_path, project_root=project_root,
                config_path=config_path, steps=["10"],
                intervals=intervals, w2v_params=w2v_params,
            ):
                if kind == "log":
                    log_lines.append(str(payload))
                    log_box.code("\n".join(log_lines[-400:]), language="text")
                elif kind == "done":
                    returncode = payload
            if returncode == 0:
                status.update(label="Training abgeschlossen ✅", state="complete")
            else:
                status.update(label=f"Mit Fehler beendet (Code {returncode})",
                              state="error")
    except Exception as exc:
        show_error(exc)
    else:
        if returncode == 0:
            st.success("Fertig. Das Modell liegt in `output/`. "
                       "Qualität unten prüfen oder auf der Seite 'Wortvektoren' "
                       "weiter erkunden.")
        else:
            st.error("Das Training ist nicht sauber durchgelaufen. Bitte das Log "
                     "prüfen (häufig: fehlende `korpus_gen`-Datei - erst die "
                     "Vorverarbeitung laufen lassen).")
        st.download_button("⬇️ Log herunterladen",
                           "\n".join(log_lines).encode("utf-8"),
                           file_name="word2vec_log.txt", mime="text/plain")
        outputs = list_output_tree(project_root)
        if outputs:
            with st.expander(f"Erzeugte Dateien in output/ ({len(outputs)})"):
                st.code("\n".join(outputs), language="text")

# 5) Qualität -------------------------------------------------------------
st.divider()
st.subheader("5 · Qualität des Modells")
st.caption("Zuerst das erzeugte Modell laden, dann Kennzahlen und eine "
           "wissenschaftliche Metrik prüfen. Für die ausführliche Analyse "
           "(Vergleich, Stabilität, diachrone Drift) siehe die Seite 'Wortvektoren'.")

if st.button("Modell laden & prüfen", key="wvm_check"):
    st.session_state["wvm_loaded"] = True

if st.session_state.get("wvm_loaded"):
    kv = None
    try:
        kv = get_models().load()
    except Exception as e:
        show_error(e)

    if kv is not None:
        import pandas as pd

        # --- Deskriptive Kennzahlen (kein Qualitätsurteil, nur Überblick) ---
        s = wm.model_summary(kv)
        m1, m2, m3 = st.columns(3)
        m1.metric("Vokabulargröße", f"{s['vocab_size']:,}".replace(",", "."))
        m2.metric("Vektordimension", s["vector_size"])
        m3.metric("Kategorie (Korpusgröße)", cat)

        # --- Qualitativer Plausibilitätscheck: nächste Nachbarn ---
        st.markdown("##### Plausibilitätscheck: nächste Nachbarn")
        probe = st.text_input("Testwort", key="wvm_probe", placeholder="z. B. natur")
        if probe.strip():
            try:
                nn = wm.nearest_neighbors(kv, probe.strip(), topn=10)
                st.dataframe(pd.DataFrame(nn, columns=["Wort", "Ähnlichkeit"]),
                             hide_index=True, use_container_width=True)
                st.caption("Qualitativer Check: Wirken die Nachbarn semantisch "
                           "sinnvoll, spricht das für die Modellqualität.")
            except KeyError:
                st.warning("Dieses Wort ist nicht im Vokabular des Modells.")

        # --- Wissenschaftliche Metrik: Word-Similarity-Korrelation (Spearman) ---
        st.markdown("##### Wissenschaftliche Metrik: Word-Similarity-Korrelation")
        st.caption(
            "Intrinsischer Standard zur Bewertung von Wortvektoren (Finkelstein "
            "et al. 2002, WordSim-353; Schnabel et al. 2015): Spearman-"
            "Rangkorrelation zwischen der Modell-Kosinusähnlichkeit und "
            "menschlichen Ähnlichkeitsurteilen für Wortpaare. Werte näher an 1 = "
            "besser. Benötigt einen Goldstandard aus bewerteten Wortpaaren "
            "(CSV mit Spalten w1,w2,score - oder direkt eingeben).")
        up = st.file_uploader("CSV mit Spalten w1,w2,score", type=["csv"],
                              key="wvm_pairs_csv")
        pairs_text = st.text_area(
            "oder Paare direkt eingeben (je Zeile: w1, w2, score)",
            value="koenig, koenigin, 0.9\nmann, frau, 0.85\ntag, nacht, 0.6",
            key="wvm_pairs_text")
        if st.button("Spearman-Korrelation berechnen", key="wvm_pairs_btn",
                     type="primary"):
            try:
                if up is not None:
                    dfp = pd.read_csv(up)
                    pairs = [(str(r.iloc[0]), str(r.iloc[1]), float(r.iloc[2]))
                             for _, r in dfp.iterrows()]
                else:
                    pairs = _parse_pairs(pairs_text)
                res = wm.evaluate_pairs(kv, pairs)
                if res["spearman"] is None:
                    st.warning(f"Zu wenige Paare im Vokabular (verwendet: "
                               f"{res['n_used']}, OOV: {res['n_oov']}). Bitte Paare "
                               "wählen, deren Wörter im Modell vorkommen.")
                else:
                    st.metric("Spearman-Korrelation", f"{res['spearman']:.3f}")
                    st.caption(f"verwendete Paare: {res['n_used']} | "
                               f"übersprungen (OOV): {res['n_oov']} | "
                               f"p = {res['pvalue']:.3g}")
                    st.caption("Vorbehalt: Für historisches Deutsch sind absolute "
                               "Werte vorsichtig zu lesen - gängige Benchmarks "
                               "(WordSim-353, SimLex-999, Gur350) stammen aus "
                               "modernem Sprachgebrauch (Benchmark-Mismatch). "
                               "Aussagekräftiger ist der Vergleich mehrerer eigener "
                               "Modelle auf demselben Paar-Satz.")
            except Exception as e:
                show_error(e)
