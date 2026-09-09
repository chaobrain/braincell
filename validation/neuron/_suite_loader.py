# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Shared plumbing for the out-of-package NEURON comparison suites.

``cable/`` and ``channel_no_conc/`` are parallel suites with the same layout
(``engine/``, ``workflows/``, ``tests/``) and the same two loading problems,
so both used to carry their own copy of the code below. They are not
importable as ordinary packages -- the tests load engine modules by path --
which is what makes the loader necessary in the first place.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import brainstate

SUITES_ROOT = Path(__file__).resolve().parent


def use_double_precision() -> None:
    """Force float64 for a suite that compares against NEURON.

    NEURON integrates in double precision, and these suites compare against
    it at tolerances as tight as 2e-5 mV. Under the float32 default, rounding
    accumulates through the capacitive voltage update -- roughly 0.03 mV of
    monotonic drift over a 4 ms run, growing with step count -- which swamps
    those tolerances.

    ``brainstate.environ.set`` is process-global, so whichever module runs
    first decides for everyone. Each suite calls this from the helper module
    every one of its tests imports, rather than relying on a sibling suite
    having happened to set it: run either package alone and the comparison
    tests would otherwise fail.
    """
    brainstate.environ.set(precision=64)


def load_suite_module(suite_root: Path, path: Path, name: str):
    """Load a suite module under ``name`` with its real package attached.

    The engine modules import their siblings relatively (``from .compare
    import ...``) and fall back to bare absolute names on ImportError.
    Loading them with no package makes the relative import fail every time,
    so the fallback registers ``compare`` / ``experiment_schema`` /
    ``outputs`` as top-level modules -- names both suites use. Whichever
    suite ran first then owned the name and the other imported the wrong
    file.

    Loading under a dotted name inside ``<suite>.<subpackage>`` lets the
    relative imports resolve, so the fallback never runs and no *engine
    module basename* is claimed. The module is also aliased under the
    caller's bare ``name``, but those are suite-prefixed and unique, so they
    collide with nothing. The package is taken from the file's own
    directory, so this works for ``engine/`` and ``workflows/`` alike. The
    caller's ``name`` is kept as the leaf, so callers that load the same file
    twice still get independent module objects and cannot leak state into
    each other.

    Parameters
    ----------
    suite_root : Path
        The suite directory, e.g. ``.../neuron_compare/cable``. Its basename
        becomes the first component of the qualified module name.
    path : Path
        The module file to load.
    name : str
        Leaf name to register the module under.

    Returns
    -------
    ModuleType
        The executed module.
    """
    if str(SUITES_ROOT) not in sys.path:
        sys.path.insert(0, str(SUITES_ROOT))

    package = f"{suite_root.name}.{path.resolve().parent.name}"
    qualified = f"{package}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
