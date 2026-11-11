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

import gc  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402

# Precision isolation between tests (OPT-12).
#
# jaxlib caches the canonical form of a host numpy array per array object and
# hands it back for as long as that form is alive, whatever
# ``jax_enable_x64`` is by then. Tracing under x64 keeps a float64 view of a
# long-lived numpy constant (brainunit's ``gas_constant.mantissa``, for one)
# in a jaxpr literal; while that jaxpr exists, a float32 program given the
# same constant receives a float64 buffer and fails with
# ``RuntimeProgramInputMismatch``. Switching the other way fails the same
# way. A finished test's jaxprs sit in reference cycles until the cyclic
# collector runs, so failures depended on test order.
#
# So: collect garbage at a test's first precision switch unless the previous
# test already collected after itself, and after every test that switched.
# A full collection takes 0.1-0.4 s, too slow to run after every test. A test
# that switches precision with objects from the other precision still alive
# is not covered; run such a check in a child process.
_switched_in_this_test = False
_collected_after_last_test = False


def _update_and_collect_on_precision_switch(name, value, *, _update=jax.config.update):
    global _switched_in_this_test
    if name == "jax_enable_x64" and bool(value) != bool(jax.config.jax_enable_x64):
        if not _switched_in_this_test and not _collected_after_last_test:
            gc.collect()
        _switched_in_this_test = True
    _update(name, value)


jax.config.update = _update_and_collect_on_precision_switch


@pytest.fixture(autouse=True)
def _collect_garbage_after_a_precision_switch():
    global _switched_in_this_test, _collected_after_last_test
    _switched_in_this_test = False
    yield
    _collected_after_last_test = _switched_in_this_test
    if _switched_in_this_test:
        gc.collect()
