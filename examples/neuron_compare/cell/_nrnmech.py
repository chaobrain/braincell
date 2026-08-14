"""Locating compiled NEURON mechanism libraries across NEURON versions.

Every cell model under this directory points at an ``nrnivmodl`` build of the
matching ``Cerebellum_mod/<CELL>`` tree. Where the built library lands depends
on the NEURON version, so the path cannot be written as a constant.
"""

from __future__ import annotations

from pathlib import Path


def nrnmech_path(build_dir: Path) -> Path:
    """Return the compiled mechanism library inside an ``nrnivmodl`` build dir.

    NEURON <= 8 builds through libtool and emits ``x86_64/.libs/libnrnmech.so``;
    NEURON >= 9 emits ``x86_64/libnrnmech.so``. Prefer whichever is present, and
    fall back to the modern layout when nothing has been compiled yet, so the
    reported path names the file a fresh build would produce.

    Parameters
    ----------
    build_dir : Path
        The ``x86_64`` directory ``nrnivmodl`` writes into.

    Returns
    -------
    Path
        Path to ``libnrnmech.so``. May not exist if nothing was compiled.
    """
    legacy = build_dir / ".libs" / "libnrnmech.so"
    return legacy if legacy.exists() else build_dir / "libnrnmech.so"
