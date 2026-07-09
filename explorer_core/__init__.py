#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core
=============

UI-freie Logikschicht des signifier-Dashboards.

Alle Module sind frei von Tkinter/Streamlit und können auch in Notebooks
oder eigenen Skripten verwendet werden (didaktisch nützlich).
"""

from .schema import MetadataSchema, find_column  # noqa: F401
from .data_store import (  # noqa: F401
    DataStore, ModelStore, DEFAULT_PATHS, PATH_LABELS, PATH_CATEGORIES,
    read_csv_auto, detect_delimiter, detect_project_root,
)
