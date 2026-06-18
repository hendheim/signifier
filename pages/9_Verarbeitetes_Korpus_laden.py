from pathlib import Path

import streamlit as st

from explorer_core.data_store import PATH_LABELS
from ui_helpers import get_store, get_models, get_schema, show_error

store = get_store()
schema = get_schema()
models = get_models()

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
            st.success(f"Projektordner gesetzt: {root}")
        else:
            st.error("Ordner existiert nicht.")

if st.button("🔍 Dateien automatisch suchen (Auto-Discovery)"):
    found = store.auto_discover()
    models.set_path(store.paths["w2v_model"])
    n_found = sum(1 for p in found.values() if p is not None)
    st.success(f"{n_found}/{len(found)} Dateien gefunden und übernommen.")

# ----------------------------------------------------------------------------
# 2) Dateipfade
# ----------------------------------------------------------------------------
st.header("2 · Datenquellen")
status = store.status()

with st.expander("Dateipfade anzeigen / anpassen",
                 expanded=not all(status.values())):
    st.caption("Grüner Haken = Datei vorhanden. Pfade können hier direkt "
               "überschrieben werden (z. B. für ein anderes Termset).")
    for key, label in PATH_LABELS.items():
        icon = "✅" if status.get(key) else "❌"
        new_path = st.text_input(f"{icon} {label}", value=str(store.paths[key]),
                                 key=f"path_{key}")
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
        ("DTM", store.load_dtm),
        ("TF-IDF", store.load_tfidf),
        ("Kosinus-Matrix", store.load_cosine),
        ("Topic-Verteilung", store.load_topics_dist),
        ("Termset", store.load_termset),
        ("Topic-Words", store.load_topic_words),
        ("Rankings", store.load_ranks),
        ("Counts/Jahr", store.load_counts_per_year),
        ("Tokens/Jahr", store.load_tokens_year),
        ("TopDocs", store.load_global_topdocs),
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

st.divider()
st.caption(
    "💡 **Neu hier?** Reihenfolge: (1) Projektordner setzen → (2) "
    "'Dateien automatisch suchen' → (3) 'Laden & Prüfen' → links eine "
    "Analyse-Seite öffnen. Eigene Spaltennamen? Seite **Schema** öffnen."
)
