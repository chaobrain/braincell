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

"""Compile reference MOD sources into workflow-owned artifact directories.

Run ``python -m validation.neuron._mechanisms MODEL --kind cell`` before
opening a whole-cell comparison. Channel sweeps use ``--kind channel``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from validation.neuron._paths import DATA_ROOT, mechanism_build_dir, model_dir
from validation.neuron.cell._nrnmech import nrnmech_path


def compile_mechanisms(model: str, kind: str = "cell", *, files=None) -> Path:
    """Build a model or explicit MOD subset and return its library parent.

    ``files`` selects an isolated content-addressed subset, used by ion
    notebooks. Default cell builds include channel, ion and synapse sources.
    Existing libraries are reused only for the same sources and NEURON version.
    """
    import neuron

    source_root = DATA_ROOT / "mechanisms" if model == "testing" else model_dir(model) / "mechanisms"
    if kind not in ("cell", "channel"):
        raise ValueError("kind must be 'cell' or 'channel'.")
    if files is not None:
        sources = sorted(Path(p).resolve() for p in files)
    else:
        if model == "testing":
            categories = ("testing",)
        elif kind == "channel":
            categories = ("channel",)
        else:
            categories = ("channel", "ion", "synapse")
        sources = sorted(p for category in categories for p in (source_root / category).glob("*.mod"))
    if not sources:
        raise FileNotFoundError(f"No MOD sources selected for {model}.")
    if len({p.name for p in sources}) != len(sources):
        raise ValueError("Selected MOD files must have distinct basenames.")
    signature = {
        "neuron": neuron.__version__,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    variant = kind if files is None else "subset-" + hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:16]
    build_root = mechanism_build_dir(model, variant)
    manifest = build_root / "sources.json"
    library = nrnmech_path(build_root / "x86_64")
    if library.exists() and manifest.exists() and json.loads(manifest.read_text()) == signature:
        return build_root
    compiler = Path(sys.executable).parent / "nrnivmodl"
    executable = str(compiler) if compiler.is_file() else shutil.which("nrnivmodl")
    if executable is None:
        raise FileNotFoundError("nrnivmodl is unavailable; install NEURON in the active Python environment.")
    build_root.mkdir(parents=True, exist_ok=True)
    # Clear only generated files in this workflow-owned build directory.
    for old in build_root.glob("*.mod"):
        old.unlink()
    for generated in (build_root / "x86_64",):
        if generated.exists():
            shutil.rmtree(generated)
    for source in sources:
        shutil.copy2(source, build_root / source.name)
    env = os.environ.copy()
    for name, tool in (("CPP", "cpp"), ("CC", "cc"), ("CXX", "c++")):
        if command := shutil.which(tool):
            env.setdefault(name, command)
    subprocess.run([executable], cwd=build_root, env=env, check=True)
    if not nrnmech_path(build_root / "x86_64").is_file():
        raise FileNotFoundError(f"NEURON did not produce a mechanism library in {build_root}.")
    manifest.write_text(json.dumps(signature, indent=2) + "\n")
    return build_root


def main() -> None:
    """Compile a cerebellar model's cell or channel mechanisms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--kind", choices=("cell", "channel"), default="cell")
    args = parser.parse_args()
    print(compile_mechanisms(args.model, args.kind))


if __name__ == "__main__":
    main()
