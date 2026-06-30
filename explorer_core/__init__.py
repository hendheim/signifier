#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core
=============

UI-freie Logikschicht des FaDeLive-Dashboards.

Module
------
- ``schema``           : konfigurierbares Metadatenschema (YAML + Auto-Detect)
- ``data_store``       : Daten-/Modell-Lader mit Caching
- ``analysis_terms``   : Frequenz, TF-IDF, Konkordanz, Kollokation, Wortverläufe
- ``analysis_vectors`` : Embeddings, Netzwerke, Cluster, Wortwolke, Dendrogramme
- ``analysis_texts``   : UMAP-Streudiagramme der Texte
- ``analysis_topics``  : Topicverläufe, Tag-Topic-Analysen
- ``pipeline_runner``  : Pipeline (s01–s07) als Subprozess + Live-Log
- ``tagging``          : POS-Liste laden/validieren/taggen/speichern
- ``tag_processing``   : Wrapper um s01_process_stop_pos_tag.py (Modul 3)

Alle Module sind frei von Tkinter/Streamlit und können auch in Notebooks
oder eigenen Skripten verwendet werden (didaktisch nützlich).
"""

from .schema import MetadataSchema, find_column  # noqa: F401
from .data_store import (  # noqa: F401
    DataStore, ModelStore, DEFAULT_PATHS, PATH_LABELS, PATH_CATEGORIES,
    read_csv_auto, detect_delimiter, detect_project_root,
)
