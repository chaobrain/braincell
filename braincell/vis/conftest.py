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

"""Fixtures shared across the ``braincell.vis`` test modules.

Non-fixture helpers belong in the sibling ``_testing.py`` and are imported
explicitly; a pytest fixture is the one kind of helper that cannot be, since
pytest resolves it by name from the collected ``conftest.py`` chain. Placing
it here rather than in ``_testing.py`` keeps the three benchmark modules free
of an import that exists only to satisfy the fixture lookup.

This file sits at ``vis/`` level so ``vis/layout/`` inherits it too.
"""

import pytest


@pytest.fixture
def clean_layout_cache():
    """Clear the shared layout cache before a benchmark, close figures after.

    The layout cache is process-global, so without this a benchmark would
    time a cache hit populated by whichever test happened to run first.

    ``matplotlib`` is imported inside the body, not at module scope: a
    ``conftest.py`` is imported when pytest collects the directory, so a
    top-level import would turn a missing optional dependency into a
    collection error for all of ``braincell/vis/`` rather than for the
    individual modules that actually need it.
    """
    import matplotlib.pyplot as plt

    from braincell.vis.layout import get_default_layout_cache

    get_default_layout_cache().clear()
    yield
    plt.close("all")
