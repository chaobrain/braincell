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

"""Locations and expectations for the on-disk morphology fixtures.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. It is the single source for the fixture directory:
resolving ``parents[N]`` by hand at each call site meant four copies at two
different depths, one of which is easy to get silently wrong when a test file
moves between package levels.

Consumed by the reader tests under ``braincell/io`` and re-exported from
``braincell/vis/_testing.py`` for the visualization tests that render real
reconstructions.
"""

from pathlib import Path

from braincell.morph.branch import BRANCH_TYPES

__all__ = ["ALLOWED_TYPES", "FIXTURE_DIR", "VALID_SWC_FIXTURES"]

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "morphology"
"""SWC and ASC fixtures shipped in the repository checkout.

``MANIFEST.in`` prunes ``data/`` from the source distribution, so this path
only resolves from a source tree. Tests that use it have no skip guard and
will error rather than skip when the fixtures are absent — that is deliberate,
so a checkout with missing data fails loudly instead of quietly passing.
"""

VALID_SWC_FIXTURES = ("grc.swc", "io.swc")
"""SWC fixtures small enough to read and render in every parametrized sweep."""

ALLOWED_TYPES = BRANCH_TYPES
"""Every branch type a reader may produce from the shipped fixtures.

Taken from the :mod:`braincell.morph.branch` registry rather than copied, so a
newly registered branch type cannot leave the assertion silently checking a
stale set.
"""
