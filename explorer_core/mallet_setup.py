#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.mallet_setup
==========================

Hilfsfunktionen, um MALLET halb-automatisch einzurichten – aufrufbar aus
einer Streamlit-Seite:

- ``find_java()``            : prüft, ob eine Java-Laufzeit vorhanden ist.
- ``setup_mallet(...)``      : lädt das MALLET-Binary (oder nimmt eine lokale
                              ZIP), entpackt es, macht ``bin/mallet`` ausführbar
                              und liefert den Starter-Pfad.
- ``locate_mallet(dir)``     : findet ``bin/mallet`` / ``bin/mallet.bat``.
- ``mallet_status(dir)``     : kompакter Status (Java + Starter).

Java selbst wird NICHT installiert (OS-/rechteabhängig) – fehlt es, sollte die
UI eine Installationsanleitung zeigen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Optional
from urllib.request import urlretrieve

# Standard-Download (konfigurierbar – URL kann sich ändern)
DEFAULT_MALLET_URL = "https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip"


def find_java() -> Optional[str]:
    """Pfad zur ``java``-Laufzeit oder ``None``."""
    return shutil.which("java")


def java_version() -> Optional[str]:
    """Versions-String von Java (oder ``None``), best effort."""
    java = find_java()
    if not java:
        return None
    try:
        out = subprocess.run([java, "-version"], capture_output=True,
                             text=True, timeout=15)
        # `java -version` schreibt nach stderr
        text = (out.stderr or out.stdout or "").strip()
        return text.splitlines()[0] if text else None
    except Exception:
        return None


def _is_url(source: str) -> bool:
    return str(source).startswith(("http://", "https://", "ftp://", "file://"))


def locate_mallet(install_dir: Path) -> Optional[Path]:
    """Sucht den MALLET-Starter unterhalb von ``install_dir``."""
    install_dir = Path(install_dir)
    candidates = ["bin/mallet", "bin/mallet.bat"]
    # direkt …/bin/mallet
    for c in candidates:
        p = install_dir / c
        if p.exists():
            return p
    # oder eine Ebene tiefer (z. B. entpacktes mallet-2.0.8/bin/mallet)
    for sub in sorted(install_dir.glob("*/")):
        for c in candidates:
            p = sub / c
            if p.exists():
                return p
    return None


def _make_executable(path: Path) -> None:
    if os.name != "nt":
        try:
            mode = path.stat().st_mode
            path.chmod(mode | 0o111)  # +x für u/g/o
        except Exception:
            pass


def setup_mallet(source: str = DEFAULT_MALLET_URL,
                 target_dir: Path = Path("resources/mallet"),
                 progress: Optional[Callable[[float], None]] = None
                 ) -> dict:
    """Lädt/entpackt MALLET und macht den Starter ausführbar.

    ``source`` ist eine URL **oder** der Pfad zu einer bereits
    heruntergeladenen ZIP (für offline). ``progress`` (optional) erhält den
    Download-Fortschritt als Anteil 0..1.

    Returns
    -------
    dict mit ``install_dir``, ``mallet_path`` (oder None), ``java`` (oder None).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # ZIP beschaffen
    if _is_url(source):
        tmp = Path(tempfile.mkdtemp()) / "mallet.zip"

        def _hook(block, block_size, total):
            if progress and total > 0:
                progress(min(1.0, (block * block_size) / total))

        urlretrieve(source, tmp, reporthook=_hook if progress else None)
        zip_path = tmp
    else:
        zip_path = Path(source)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP nicht gefunden: {zip_path}")

    # Entpacken
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    launcher = locate_mallet(target_dir)
    if launcher is not None:
        _make_executable(launcher)
    install_dir = launcher.parent.parent if launcher else target_dir

    return {"install_dir": str(install_dir),
            "mallet_path": str(launcher) if launcher else None,
            "java": find_java()}


def mallet_status(install_dir: Path) -> dict:
    """Kompakter Status für die Anzeige."""
    launcher = locate_mallet(Path(install_dir))
    return {"java": find_java(), "java_version": java_version(),
            "mallet_path": str(launcher) if launcher else None,
            "ready": bool(launcher and find_java())}
