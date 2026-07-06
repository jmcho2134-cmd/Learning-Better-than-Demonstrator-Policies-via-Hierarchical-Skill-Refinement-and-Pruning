#!/usr/bin/env python
"""
m1_config.py
============

Tiny shared helpers for Module 1: locating the project root, loading the YAML
config, resolving config-relative paths, and deriving the phase label space.

Kept dependency-light on purpose: the only third-party import is ``yaml``
(PyYAML), which the README lists as a runtime dependency to ``pip install``.

Import pattern used by every CLI entry point (so the ``replay`` / ``feature_bank``
/ ``phase_segmenter`` / ``goal`` packages resolve regardless of the current
working directory)::

    import os, sys
    sys.path.insert(0, PROJECT_ROOT)     # computed per-script (see add_root_to_path)
    from m1_config import load_config, resolve_path, get_phase_names
"""

import os
import sys


# The project root is simply the directory that holds THIS file.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def add_root_to_path():
    """Ensure the project root is importable (idempotent).

    Every CLI script calls this before importing sibling packages so that
    ``python phase_segmenter/train.py`` works the same as running from the root.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def load_config(path):
    """Load the YAML config into a plain dict.

    Raises a clear error (not a bare ImportError) if PyYAML is missing, since
    that is the single most likely first-run failure.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime-only guard
        raise SystemExit(
            "PyYAML is required to read the M1 config but is not installed.\n"
            "  pip install pyyaml\n"
            f"(original import error: {exc})"
        )
    if not os.path.isfile(path):
        raise SystemExit(f"config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config did not parse to a mapping: {path}")
    return cfg


def resolve_path(path):
    """Resolve a (possibly relative) config path against the project root.

    Absolute paths pass through unchanged. This makes ``./data/processed`` etc.
    stable no matter where the user launches the script from.
    """
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def get_phase_names(cfg):
    """Return the ordered list of phase names, honoring ``use_stabilize``.

    The list index IS the integer phase id used everywhere (weak labels, model
    logits, z_t). K = len(names).
    """
    phases = cfg.get("phases", {})
    names = list(phases.get("names", ["approach", "grasp", "lift", "transport", "place"]))
    if phases.get("use_stabilize", False) and "stabilize" not in names:
        names = names + ["stabilize"]
    return names
