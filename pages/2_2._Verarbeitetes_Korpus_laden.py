from pathlib import Path

import streamlit as st

from explorer_core.data_store import PATH_LABELS, PATH_CATEGORIES
from ui_helpers import get_store, get_models, get_schema, timed

store = get_store()
schema = get_schema()
models = get_models()


def _sync_path_fields(keys=None):
    """Überträgt die aktuellen Store-Pfade in die Textfelder unten (Session-
    State), damit ein von den Buttons geladener Pfad dort sofort als Pfad
    erscheint. Muss VOR dem Rendern der Textfelder laufen – die Buttons stehen
    oberhalb der Felder."""
    for k in (store.paths if keys is None else keys):
        st.session_state[f"path_{k}"] = str(store.paths[k])

# ----------------------------------------------------------------------------
# 1) Projektordner
# ----------------------------------------------------------------------------
st.header("1 · Projektordner")
st.caption("Der Ordner, der `output/`, `resources/` und `korpus/` enthält.")

col_root, col_btn = st.columns([4, 1])
with col_root:
    root_input = st.text_input("Projektordner", value=str(store.project_root),
                               label_visibility="collapsed")
with col_btn:
    if st.button("Übernehmen", use_container_width=True):
        root = Path(root_input).expanduser()
        if root.exists():
            store.set_project_root(root)
            models.set_path(store.paths["w2v_model"])
            _sync_path_fields()
            st.success(f"Projektordner gesetzt: {root}")
        else:
            st.error("Ordner existiert nicht.")

if st.button("🔍 Dateien automatisch suchen (Auto-Discovery)"):
    found = store.auto_discover()
    models.set_path(store.paths["w2v_model"])
    _sync_path_fields()
    n_found = sum(1 for p in found.values() if p is not None)
    st.success(f"{n_found}/{len(found)} Dateien gefunden und übernommen.")

# ----------------------------------------------------------------------------
# 2) Datenquellen
# ----------------------------------------------------------------------------
st.header("2 · Datenquellen")

# --- Ergebnis-Ordner bequem auswählen (setzt die jeweiligen Pfade) ---
st.subheader("Ergebnis-Ordner auswählen")


def _report(applied, folder, source_label, warn=True):
    found = [PATH_LABELS[k] for k, p in applied.items() if p is not None]
    missing = [PATH_LABELS[k] for k, p in applied.items() if p is None]
    if found:
        st.success(f"{source_label} aus '{folder}': {', '.join(found)}")
    elif not warn:
        st.info(f"In '{folder}' sind keine passenden Dateien vorhanden.")
    if missing and warn:
        st.warning(f"Im Ordner '{folder}' nicht gefunden: {', '.join(missing)}")


col_ts, col_dtti, col_tm = st.columns(3)

with col_ts:
    termset_dirs = store.list_termset_dirs()
    if termset_dirs:
        ts_choice = st.selectbox(
            "Termset laden", options=termset_dirs, key="ts_load_choice",
            help="Wählt ein Termset aus output/processed_termset/ und setzt die "
                 "Document-Termset-Topics-Verarbeitungen (Term-Topic-Ranking, "
                 "-Score, -Year-Matrix) auf die dort gefundenen Dateien.")
        if st.button("Termset übernehmen", use_container_width=True,
                     key="ts_load_btn"):
            _report(store.apply_termset_dir(ts_choice), ts_choice,
                    "Termset geladen")
            _sync_path_fields()
    else:
        st.caption("Noch keine Ordner unter `output/processed_termset/`.")

with col_dtti:
    dtti_dirs = store.list_dtti_dirs()
    if dtti_dirs:
        dtti_choice = st.selectbox(
            "Document-Term-Topic-Index laden", options=dtti_dirs,
            key="dtti_load_choice",
            help="Wählt eine Kombination <Termset>/<Topic-Modell> aus "
                 "output/processed_termset/ und setzt die Document-Termset-"
                 "Topics-Verarbeitungen auf genau diesen Ordner.")
        if st.button("DTTI-Index übernehmen", use_container_width=True,
                     key="dtti_load_btn"):
            _report(store.apply_termset_dir(dtti_choice), dtti_choice,
                    "DTTI-Index geladen")
            _sync_path_fields()
    else:
        st.caption("Noch keine <Termset>/<Topic-Modell>-Ordner vorhanden.")

with col_tm:
    model_dirs = store.list_topic_model_dirs()
    if model_dirs:
        tm_choice = st.selectbox(
            "Topic-Modell laden", options=model_dirs, key="tm_load_choice",
            help="Wählt ein Modell aus resources/topic-models/ und setzt dessen "
                 "Topic-Model-Dateien (Document-Topic-Matrix, Topic-Word-Matrix) "
                 "aus resources/topic-models/<Modell>/ sowie die Verarbeiteten "
                 "Topics (Topic-Ranking pro Jahr/Text) aus "
                 "output/processed_topics/<Modell>/. Nur vorhandene Dateien.")
        if st.button("Topic-Modell übernehmen", use_container_width=True,
                     key="tm_load_btn"):
            _report(store.apply_topic_model_dir(tm_choice), tm_choice,
                    "Topic-Modell geladen", warn=False)
            _sync_path_fields()
    else:
        st.caption("Noch keine Modelle unter `resources/topic-models/`.")

# --- Manuelle Pfade, nach Kategorien gruppiert (Override) ---
# status erst hier (nach den Buttons) bestimmen, damit die ✅/❌-Icons die
# soeben geladenen Pfade berücksichtigen.
status = store.status()
with st.expander("Dateipfade anzeigen / anpassen",
                 expanded=not all(status.values())):
    st.caption("Grüner Haken = Datei vorhanden. Pfade können hier direkt "
               "überschrieben werden (z. B. für ein anderes Termset). Über die "
               "Buttons oben geladene Pfade erscheinen hier automatisch.")
    for category, keys in PATH_CATEGORIES.items():
        st.markdown(f"**{category}**")
        for key in keys:
            sk = f"path_{key}"
            # Feld wird über den Session-State gesteuert (key-only, kein value=),
            # damit die Buttons oben den Wert per _sync_path_fields setzen können.
            if sk not in st.session_state:
                st.session_state[sk] = str(store.paths[key])
            label = PATH_LABELS[key]
            icon = "✅" if status.get(key) else "❌"
            new_path = st.text_input(f"{icon} {label}", key=sk)
            if new_path != str(store.paths[key]):
                store.set_path(key, new_path)
                if key == "w2v_model":
                    models.set_path(new_path)

# ----------------------------------------------------------------------------
# 3) Laden & Prüfen
# ----------------------------------------------------------------------------
st.header("3 · Laden & Prüfen")

if st.button("Alle Daten laden & prüfen", type="primary"):
    store.invalidate()
    checks = [
        ("Metadaten", store.load_metadata),          # zuerst: registriert Spalten
        ("Korpus", store.load_corpus),
        ("Tokenverteilung pro Jahr", store.load_tokens_year),
        ("DTM", store.load_dtm),
        ("TF-IDF", store.load_tfidf),
        ("Kosinus-Matrix", store.load_cosine),
        ("Document-Topic-Matrix", store.load_topics_dist),
        ("Topic-Word-Matrix", store.load_topic_words),
        ("Termset", store.load_termset),
        ("Term-Topic-Ranking", store.load_ranks),
        ("Term-Topic-Score", store.load_relevance),
        ("Term-Topic-Year-Matrix", store.load_counts_per_year),
        ("Document-Term-Topic-Rank pro Jahr", store.load_top10_year_value),
        ("Document-Term-Topic-Rank pro Text", store.load_top10_value_per_text),
    ]
    ok = 0
    for name, loader in checks:
        try:
            df = loader()
            st.success(f"{name}: {df.shape[0]:,} × {df.shape[1]} geladen")
            ok += 1
        except Exception as e:
            st.warning(f"{name}: {e}")
    try:
        kv = get_models().load()
        st.success(f"Word2Vec-Modell: {len(kv)} Wörter")
        ok += 1
    except Exception as e:
        st.warning(f"Word2Vec-Modell: {e}")

    st.info(f"**{ok}/{len(checks) + 1}** Quellen geladen. Nicht jede Seite "
            "braucht alle Quellen – fehlende Dateien betreffen nur die "
            "jeweiligen Analysen.")

    # Spaltenanalyse der DTM (Mapping über die Metadatendatei)
    try:
        dtm = store.load_dtm()
        meta_cols = schema.metadata_columns_in(dtm)
        term_cols = schema.term_columns_in(dtm)
        st.markdown(
            f"**Spaltenanalyse DTM:** {len(meta_cols)} Metadaten-Spalten, "
            f"{len(term_cols)} Term-Spalten "
            f"(Mapping über die Metadatendatei{'' if schema.metadata_registered else ' – nicht geladen, Heuristik aktiv'})."
        )
    except Exception:
        pass

# ----------------------------------------------------------------------------
# 4) Ergebnisse herunterladen (Statistiken + gespeicherte Bilder)
# ----------------------------------------------------------------------------
st.header("4 · Ergebnisse herunterladen")
st.caption(
    "Bündelt alle Statistik-CSVs sowie die auf den Analyse-Seiten erzeugten "
    "Bilder samt Hyperparameter-Dateien (`output/statistics/`, inkl. "
    "`bilder/`) in eine ZIP-Datei."
)

_stats_dir = Path(store.project_root) / "output" / "statistics"
if st.button("📦 ZIP zusammenstellen", key="stats_zip_btn"):
    import io
    import zipfile
    try:
        with timed("ZIP erstellen"):
            buf = io.BytesIO()
            n_files = 0
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                if _stats_dir.exists():
                    for f in sorted(_stats_dir.rglob("*")):
                        if f.is_file():
                            zf.write(f, arcname=str(f.relative_to(_stats_dir)))
                            n_files += 1
            st.session_state["stats_zip"] = buf.getvalue()
            st.session_state["stats_zip_n"] = n_files
    except Exception as e:
        st.error(f"⚠️ ZIP konnte nicht erstellt werden: {e}")

if "stats_zip" in st.session_state:
    _n = st.session_state["stats_zip_n"]
    _data = st.session_state["stats_zip"]
    if _n == 0:
        st.info("Keine Dateien in `output/statistics/` gefunden – erst die "
                "Token-Statistik (Seite 'Token-Statistik erstellen') ausführen "
                "bzw. auf den Analyse-Seiten Bilder erzeugen.")
    else:
        st.download_button(
            f"⬇️ Statistiken + Bilder herunterladen "
            f"({_n} Dateien, {len(_data) / 1e6:.1f} MB)",
            _data,
            file_name="signifier_statistik.zip",
            mime="application/zip",
            key="dl_stats_zip",
        )

st.divider()
st.caption(
    "💡 **Neu hier?** Reihenfolge: (1) Projektordner setzen → (2) "
    "'Dateien automatisch suchen' → (3) 'Laden & Prüfen' → links eine "
    "Analyse-Seite öffnen. Eigene Spaltennamen? Seite **Schema** öffnen."
)
