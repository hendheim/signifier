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


def windows_safe_launcher(launcher: Path) -> Path:
    """Auf Windows den ``.bat``-Starter bevorzugen.

    Das endungslose Unix-Skript ``bin/mallet`` lässt sich auf Windows NICHT per
    ``subprocess`` ausführen (``WinError 2``/``193``). Liegt daneben
    ``mallet.bat``, wird dieser zurückgegeben. Auf anderen Systemen unverändert.
    """
    launcher = Path(launcher)
    if os.name == "nt" and launcher.suffix.lower() != ".bat":
        bat = launcher.with_suffix(".bat")
        if bat.exists():
            return bat
    return launcher


def locate_mallet(install_dir: Path) -> Optional[Path]:
    """Sucht den MALLET-Starter unterhalb von ``install_dir``.

    Auf Windows wird ``bin/mallet.bat`` bevorzugt (das endungslose Unix-Skript
    ``bin/mallet`` ist dort nicht per ``subprocess`` startbar).
    """
    install_dir = Path(install_dir)
    candidates = (["bin/mallet.bat", "bin/mallet"] if os.name == "nt"
                  else ["bin/mallet", "bin/mallet.bat"])
    # direkt …/bin/… und eine Ebene tiefer (z. B. entpacktes mallet-2.0.8/bin/…)
    for base in [install_dir, *sorted(install_dir.glob("*/"))]:
        for c in candidates:
            p = base / c
            if p.exists():
                return windows_safe_launcher(p)
    return None


def _make_executable(path: Path) -> None:
    if os.name != "nt":
        try:
            mode = path.stat().st_mode
            path.chmod(mode | 0o111)  # +x für u/g/o
        except Exception:
            pass


def _zip_is_complete(zip_path: Path) -> bool:
    """ZIP intakt UND enthält die MALLET-Abhängigkeiten (``lib/mallet-deps.jar``)?

    Der UMass-Server liefert die ZIP gelegentlich **truncirt** aus; ohne diese
    Prüfung würde eine kaputte Installation (ohne Klassen/JARs) entpackt.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if zf.testzip() is not None:           # erster beschädigter Eintrag
                return False
            return any(n.endswith("lib/mallet-deps.jar") for n in zf.namelist())
    except Exception:
        return False


def install_complete(install_dir: Path) -> bool:
    """Vollständig = Starter vorhanden UND ``lib/mallet-deps.jar`` da."""
    launcher = locate_mallet(Path(install_dir))
    if launcher is None:
        return False
    return (launcher.parent.parent / "lib" / "mallet-deps.jar").exists()


def setup_mallet(source: str = DEFAULT_MALLET_URL,
                 target_dir: Path = Path("resources/mallet"),
                 progress: Optional[Callable[[float], None]] = None
                 ) -> dict:
    """Lädt/entpackt MALLET und macht den Starter ausführbar.

    ``source`` ist eine URL **oder** der Pfad zu einer bereits
    heruntergeladenen ZIP (für offline). ``progress`` (optional) erhält den
    Download-Fortschritt als Anteil 0..1.

    Download und Entpacken werden **auf Vollständigkeit geprüft** – ein
    truncierter Download (UMass-Server liefert das zeitweise) führt nicht mehr
    zu einer stillen, kaputten Installation, sondern zu einem klaren Fehler.

    Returns
    -------
    dict mit ``install_dir``, ``mallet_path`` (oder None), ``java`` (oder None).
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    def _hook(block, block_size, total):
        if progress and total > 0:
            progress(min(1.0, (block * block_size) / total))

    # ZIP beschaffen (URL: mit Validierung + Wiederholung gegen truncierte Downloads)
    if _is_url(source):
        tmp = Path(tempfile.mkdtemp()) / "mallet.zip"
        last_err: Optional[Exception] = None
        for _ in range(4):
            try:
                urlretrieve(source, tmp, reporthook=_hook if progress else None)
            except Exception as exc:               # u. a. ContentTooShortError
                last_err = exc
            if _zip_is_complete(tmp):
                break
        else:
            raise RuntimeError(
                "MALLET-Download unvollständig/fehlerhaft (Server liefert evtl. "
                f"abgeschnittene Dateien). Letzter Fehler: {last_err}. Tipp: eine "
                "lokal heruntergeladene mallet-2.0.8.zip als Quelle angeben.")
        zip_path = tmp
    else:
        zip_path = Path(source)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP nicht gefunden: {zip_path}")
        if not _zip_is_complete(zip_path):
            raise RuntimeError(f"ZIP unvollständig/fehlerhaft (kein "
                               f"mallet-deps.jar / beschädigt): {zip_path}")

    # Entpacken + Vollständigkeit prüfen
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    launcher = locate_mallet(target_dir)
    if launcher is None or not install_complete(target_dir):
        raise RuntimeError("MALLET-Entpacken unvollständig: Starter bzw. "
                           "lib/mallet-deps.jar fehlt.")
    _make_executable(launcher)
    install_dir = launcher.parent.parent

    return {"install_dir": str(install_dir),
            "mallet_path": str(launcher),
            "java": find_java()}


def mallet_status(install_dir: Path) -> dict:
    """Kompakter Status für die Anzeige."""
    launcher = locate_mallet(Path(install_dir))
    complete = install_complete(install_dir)
    return {"java": find_java(), "java_version": java_version(),
            "mallet_path": str(launcher) if launcher else None,
            "complete": complete,   # Starter + mallet-deps.jar vorhanden?
            "ready": bool(launcher and find_java() and complete)}
