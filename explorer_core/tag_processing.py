#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_core.tag_processing
============================

UI-freie Logik für die Dashboard-Seite "POS-Tags verarbeiten".

Dünner Wrapper um das vorhandene Skript ``tt_s01_stop_pos_tag.py``
(Modul 3). Die Funktionen dort werden NICHT neu geschrieben, sondern
unverändert aufgerufen:

- ``run_pipeline(pos_file, tfidf_file, output_dir, max_combo_size)``
- ``add_tags_to_tfidf(pos_file, tfidf_file, output_file)``
- ``combine_tags_in_pos_file(input_file, output_file)``
- ``build_tag_stats_and_matrices(input_file, output_dir, tagset_output, max_combo_size)``
- ``sort_tagset_by_tfidf(tagset_file, tfidf_file, output_file)``

Diese Schritte sind leichtgewichtig (nur pandas) und laufen daher direkt im
Prozess. Die ``print``-Ausgaben ([1/5]…[5/5]) werden mitgeschnitten und an
die Oberfläche zurückgegeben.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
from typing import List, Optional, Tuple

# Mögliche Speicherorte des Skripts (echtes Repo-Layout vs. flach).
_SCRIPT_CANDIDATES = [
    "nlp_pipeline/tt_s01_stop_pos_tag.py",
    "tt_s01_stop_pos_tag.py",
]


def locate_script(project_root: Path) -> Optional[Path]:
    """Findet ``tt_s01_stop_pos_tag.py`` im Projektordner."""
    root = Path(project_root)
    for rel in _SCRIPT_CANDIDATES:
        candidate = root / rel
        if candidate.exists():
            return candidate
    # Letzter Versuch: rekursiv suchen
    hits = list(root.rglob("tt_s01_stop_pos_tag.py"))
    return hits[0] if hits else None


def _load_module(script_path: Path):
    """Lädt das Skript als Modul (ohne es zu installieren)."""
    spec = importlib.util.spec_from_file_location("tt_s01_stop_pos_tag", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Konnte {script_path} nicht laden.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_outputs(pos_file: Path, output_dir: Path) -> List[Path]:
    """Die Dateien, die ``run_pipeline`` erzeugt (Basisname ohne _vN)."""
    stem = Path(pos_file).stem
    # Basisnamen wie im Skript bilden ("…_v3" → "…")
    parts = stem.rsplit("_v", 1)
    base = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else stem
    out = Path(output_dir)
    return [
        out / f"{base}_tfidf_tagged.csv",
        out / f"{base}_combined.csv",
        out / f"{base}_tagset.csv",
        out / f"{base}_tagset_sorted.csv",
        out / "tag_stats",
    ]


def run_full(
    project_root: Path,
    pos_file: Path,
    tfidf_file: Path,
    output_dir: Path,
    max_combo_size: int = 3,
) -> Tuple[str, List[Path]]:
    """Führt die komplette Modul-3-Pipeline aus und schneidet das Log mit.

    Returns:
        (Log-Text, Liste tatsächlich erzeugter Dateien/Ordner)
    """
    script = locate_script(project_root)
    if script is None:
        raise FileNotFoundError(
            "tt_s01_stop_pos_tag.py wurde im Projektordner nicht gefunden."
        )
    mod = _load_module(script)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        mod.run_pipeline(
            pos_file=Path(pos_file),
            tfidf_file=Path(tfidf_file),
            output_dir=Path(output_dir),
            max_combo_size=int(max_combo_size),
        )
    created = [p for p in _expected_outputs(pos_file, output_dir) if p.exists()]
    return buffer.getvalue(), created


def run_step(
    project_root: Path,
    step: str,
    **kwargs,
) -> Tuple[str, List[Path]]:
    """Führt einen Einzelschritt aus.

    step ∈ {"add-tags-to-tfidf", "combine-tags", "build-tag-stats", "sort-tagset"}
    Die jeweils nötigen kwargs entsprechen den Funktionsargumenten im Skript.
    """
    script = locate_script(project_root)
    if script is None:
        raise FileNotFoundError("tt_s01_stop_pos_tag.py nicht gefunden.")
    mod = _load_module(script)

    buffer = io.StringIO()
    created: List[Path] = []
    with contextlib.redirect_stdout(buffer):
        if step == "add-tags-to-tfidf":
            mod.add_tags_to_tfidf(
                pos_file=Path(kwargs["pos_file"]),
                tfidf_file=Path(kwargs["tfidf_file"]),
                output_file=Path(kwargs["output_file"]),
            )
            created = [Path(kwargs["output_file"])]
        elif step == "combine-tags":
            mod.combine_tags_in_pos_file(
                input_file=Path(kwargs["input_file"]),
                output_file=Path(kwargs["output_file"]),
            )
            created = [Path(kwargs["output_file"])]
        elif step == "build-tag-stats":
            mod.build_tag_stats_and_matrices(
                input_file=Path(kwargs["input_file"]),
                output_dir=Path(kwargs["output_dir"]),
                tagset_output=Path(kwargs["tagset_output"]),
                max_combo_size=int(kwargs.get("max_combo_size", 3)),
            )
            created = [Path(kwargs["output_dir"]), Path(kwargs["tagset_output"])]
        elif step == "sort-tagset":
            mod.sort_tagset_by_tfidf(
                tagset_file=Path(kwargs["tagset_file"]),
                tfidf_file=Path(kwargs["tfidf_file"]),
                output_file=Path(kwargs["output_file"]),
            )
            created = [Path(kwargs["output_file"])]
        else:
            raise ValueError(f"Unbekannter Schritt: {step}")
    created = [p for p in created if p.exists()]
    return buffer.getvalue(), created
