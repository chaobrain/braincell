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

"""Interactive pick / hover hooks for the vis backends.

This module defines a backend-agnostic :class:`VisHooks` container plus the
:class:`PickInfo` payload delivered to user callbacks. The matplotlib and
PyVista backends (``backend_matplotlib.py`` / ``backend_pyvista.py``) consume
these at render time and register the appropriate mouse-event callbacks.

The hooks are intentionally coarse: a callback receives enough context to
identify *which branch / segment / x-coordinate* was picked, plus the
underlying scalar value when the scene was built with colour-by-values.
Callers that need the raw matplotlib artist or PyVista mesh can read them
from :attr:`PickInfo.artist`.

Typical usage::

    def on_pick(info):
        print(f"picked branch {info.branch_name} at x={info.x:.3f}")

    plot2d(morpho, hooks=VisHooks(on_pick=on_pick))
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

PickCallback = Callable[["PickInfo"], None]
LeaveCallback = Callable[[], None]


@dataclass(frozen=True)
class PickInfo:
    """Payload delivered to a :class:`VisHooks` callback.

    Parameters
    ----------
    branch_index : int
        Index of the branch inside ``morpho.branches``.
    branch_name : str
        Human-readable branch name.
    branch_type : str
        Branch type string (``"soma"``, ``"apical_dendrite"``, …).
    segment_index : int or None
        Segment index within the branch (``0`` for the proximal segment),
        or ``None`` if the artist does not resolve to a single segment.
    x : float or None
        Fractional arc-length coordinate along the branch in ``[0, 1]``,
        or ``None`` if the pick cannot be placed precisely.
    value : float or None
        Scalar at the picked segment when the scene carries a
        colour-by-values overlay. ``None`` when no values were supplied.
    position_um : numpy.ndarray or None
        Pick location in scene coordinates (2-D for matplotlib, 3-D for
        PyVista). ``None`` when the backend did not report a location.
    artist : object or None
        The underlying backend artist (e.g. ``LineCollection``,
        ``pyvista.PolyData``). Opaque to generic code but useful for
        tests and advanced callers.
    """

    branch_index: int
    branch_name: str
    branch_type: str
    segment_index: int | None = None
    x: float | None = None
    value: float | None = None
    position_um: np.ndarray | None = None
    artist: Any = None


@dataclass(frozen=True)
class VisHooks:
    """Bundle of interactive callbacks understood by the vis backends.

    Hooks are optional; passing ``VisHooks()`` (the default) wires
    nothing. When any callback is set the backend takes an interactive
    path — for matplotlib that means enabling ``pick_event`` and/or
    ``motion_notify_event`` on the figure canvas; for PyVista it means
    calling :meth:`pyvista.Plotter.enable_point_picking`.

    Parameters
    ----------
    on_pick : callable or None
        Called with a :class:`PickInfo` whenever the user clicks a
        branch. Use this for "click to inspect" tooling.
    on_hover : callable or None
        Called with a :class:`PickInfo` while the mouse hovers over a
        branch. Only delivered by the matplotlib backend at present;
        PyVista does not currently expose a hover event.
    on_leave : callable or None
        Called with no arguments when the mouse moves off a branch
        after having hovered over it. Matplotlib only.

    Notes
    -----
    Callbacks are invoked synchronously from the event loop. Keep them
    cheap; heavy work belongs in a ``QueueTimer`` or async task.
    """

    on_pick: PickCallback | None = None
    on_hover: PickCallback | None = None
    on_leave: LeaveCallback | None = None

    def is_active(self) -> bool:
        """Return ``True`` if any callback is wired."""
        return any(cb is not None for cb in (self.on_pick, self.on_hover, self.on_leave))
