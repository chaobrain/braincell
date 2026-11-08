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

"""Pixel-baseline comparison for Matplotlib figures.

This module is deliberately self-contained: it imports nothing from
``braincell`` so that it can be copied verbatim into another package's test
suite. A test builds a figure, then calls :func:`assert_figure_matches` with the
path of the committed baseline PNG.

* Rendering is pinned by :func:`baseline_style`: Agg, default ``rcParams``,
  the bundled DejaVu Sans font, no hinting and a fixed dpi.
* Pixels are compared with :func:`matplotlib.testing.compare.compare_images`
  (RMS difference over 0-255, ``tol=0`` means identical).
* Pixels are compared on Linux only. Font rasterisation differs between
  operating systems, so every other platform skips with
  :class:`unittest.SkipTest`.
* Setting the environment variable named by ``update_env`` to ``1`` rewrites
  the baseline from the figure instead of comparing, so a Matplotlib upgrade
  that changes the rendering is one command to accept.
"""

import contextlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

__all__ = ["BASELINE_DPI", "baseline_style", "assert_figure_matches"]

#: Resolution every baseline is rendered at; with a fixed ``figsize`` it fixes
#: the image size in pixels.
BASELINE_DPI = 100


@contextlib.contextmanager
def baseline_style():
    """Pin Matplotlib's rendering for the duration of the ``with`` block.

    Switches to the Agg backend's default ``rcParams`` plus the font and
    reproducibility settings Matplotlib itself uses for its image tests, and
    restores the caller's ``rcParams`` on exit.
    """
    import matplotlib as mpl
    from matplotlib.testing import set_font_settings_for_testing, set_reproducibility_for_testing

    defaults = {k: v for k, v in mpl.rcParamsDefault.items() if k not in ("backend", "backend_fallback")}
    with mpl.rc_context(defaults):
        set_font_settings_for_testing()
        set_reproducibility_for_testing()
        yield


def _updating(update_env: str) -> bool:
    return os.environ.get(update_env, "").strip().lower() in ("1", "true", "yes")


def assert_figure_matches(
    fig,
    baseline,
    *,
    tol: float = 0.0,
    update_env: str,
    dpi: int = BASELINE_DPI,
) -> None:
    """Assert that ``fig`` renders to the same pixels as the PNG at ``baseline``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render. Give it a fixed ``figsize`` and draw it inside
        :func:`baseline_style`.
    baseline : str or os.PathLike
        Baseline PNG. Missing baselines fail with a hint to generate them.
    tol : float
        RMS tolerance passed to ``compare_images``; ``0`` demands identical pixels.
    update_env : str
        Name of the environment variable that, set to ``1``, rewrites the
        baseline instead of comparing.
    dpi : int
        Render resolution.

    Raises
    ------
    unittest.SkipTest
        On any platform other than Linux.
    AssertionError
        When the baseline is missing or the pixels differ by more than ``tol``.
    """
    from matplotlib.testing.compare import compare_images

    if not sys.platform.startswith("linux"):
        raise unittest.SkipTest("pixel baselines are compared on Linux only (font rendering differs elsewhere)")

    baseline = Path(baseline)
    if _updating(update_env):
        baseline.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(baseline, dpi=dpi, format="png")
        return
    if not baseline.is_file():
        raise AssertionError(f"missing baseline {baseline}; generate it with {update_env}=1")

    result_dir = Path(tempfile.mkdtemp(prefix="baseline-"))
    actual = result_dir / baseline.name
    fig.savefig(actual, dpi=dpi, format="png")
    # compare_images writes ``<actual stem>-failed-diff.png`` beside ``actual``.
    message = compare_images(str(baseline), str(actual), tol=tol)
    if message is None:
        shutil.rmtree(result_dir, ignore_errors=True)
        return
    raise AssertionError(f"{baseline.name} differs from its baseline (images kept in {result_dir}):\n{message}")
