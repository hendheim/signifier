#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.viz_export
========================

Dünne, additive Hülle um ``ui_helpers.fig_with_download``: Sie rendert die
Grafik weiterhin über die bestehende Funktion und bietet ZUSÄTZLICH die
verwendeten Hyperparameter als separate ``.txt``-Datei zum Download an – mit
demselben Basisdateinamen wie die Grafik (``<name>.png`` ⇒ ``<name>.txt``).

Damit ist sichergestellt, dass zu *jeder* gespeicherten Grafik (Wortverläufe,
Streudiagramme, Dendrogramme, …) die Parameter nachvollziehbar mit ausgegeben
werden, ohne die zentrale ``fig_with_download``-Funktion selbst ändern zu
müssen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Mapping, Any

import streamlit as st

from ui_helpers import fig_with_download


def format_hyperparams(params: Optional[Mapping[str, Any]],
                       filename: str, title: Optional[str] = None) -> str:
    """Formatiert die Hyperparameter als lesbaren Klartext."""
    head = title or f"Hyperparameter – {filename}"
    lines = [head, "=" * len(head),
             f"Grafik-Datei : {filename}.png",
             f"Erzeugt      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ""]
    if params:
        width = max((len(str(k)) for k in params), default=0)
        for k, v in params.items():
            lines.append(f"{str(k).ljust(width)} : {v}")
    else:
        lines.append("(keine Parameter übergeben)")
    return "\n".join(lines) + "\n"


def save_figure(fig, filename: str,
                params: Optional[Mapping[str, Any]] = None,
                key: Optional[str] = None,
                title: Optional[str] = None) -> None:
    """Wie ``fig_with_download``, gibt aber zusätzlich eine
    Hyperparameter-``.txt`` mit gleichem Basisnamen zum Download aus.

    Parameters
    ----------
    fig:
        Die anzuzeigende/zu speichernde Matplotlib-Figur.
    filename:
        Basisdateiname OHNE Endung (z. B. ``"texte_streudiagramm_pca"``).
    params:
        Dict der verwendeten Hyperparameter (wird in ``<filename>.txt``
        geschrieben).
    key:
        Streamlit-Key-Präfix (für PNG- und TXT-Button eindeutig gemacht).
    """
    fig_with_download(fig, filename, key=key)
    txt = format_hyperparams(params, filename, title=title)
    st.download_button(
        "⬇️ Hyperparameter (TXT)",
        txt.encode("utf-8"),
        file_name=f"{filename}.txt",
        mime="text/plain",
        key=(f"{key}_hp" if key else None),
    )
