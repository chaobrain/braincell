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


import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import brainstate  # noqa: E402
import jax  # noqa: E402


# Collection leaves the precision where the session started.
#
# ``brainstate.environ.set(precision=64)`` is process-wide, and the NEURON
# comparison suites under validation/neuron call it when imported (it also
# serves their ``python <suite>.py`` runs). Collecting one of them with
# braincell's tests therefore ran every test of the session in float64; a test
# that checks float32 dtypes failed, and the rest silently tested float64 only.
# validation/neuron/conftest.py gives those suites float64 while
# their own tests run.
_SESSION_PRECISION = brainstate.environ.get_precision()
_SESSION_X64 = bool(jax.config.jax_enable_x64)


def pytest_collection_finish(session):
    if brainstate.environ.get_precision() != _SESSION_PRECISION:
        brainstate.environ.set(precision=_SESSION_PRECISION)
    if bool(jax.config.jax_enable_x64) != _SESSION_X64:
        jax.config.update("jax_enable_x64", _SESSION_X64)


# CI shards (.github/ci_shards.py): with BRAINCELL_TEST_SHARD=<index>/<count>,
# keep only the tests of the files assigned to that shard. Every shard collects
# the whole session, so each collected file runs in exactly one shard.
def pytest_collection_modifyitems(config, items):
    value = os.environ.get("BRAINCELL_TEST_SHARD")
    if not value:
        return
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_braincell_ci_shards", Path(__file__).with_name(".github") / "ci_shards.py"
    )
    shards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shards)
    index, count = shards.parse_shard(value)
    root = Path(config.rootpath)
    name_of = {item: Path(item.path).relative_to(root).as_posix() for item in items}
    keep = shards.shard_files(list(name_of.values()), index, count)
    selected = [item for item in items if name_of[item] in keep]
    deselected = [item for item in items if name_of[item] not in keep]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
