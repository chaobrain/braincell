# -*- coding: utf-8 -*-

from __future__ import annotations

"""Shared ion-side templates.

This module contains mixins used by concrete ion classes such as
``SodiumFixed`` or ``CalciumDetailed``. The public lifecycle still lives on
``Ion``; these mixins only provide helper methods and lifecycle hooks for
common ion patterns:

- fixed ``Ci/Co/E``
- fixed ``Ci/Co`` with ``E`` initialized from Nernst
- dynamic ``Ci`` with Nernst-computed ``E``
"""

import brainstate
import braintools
import brainunit as u

from braincell.quad import DiffEqState

__all__ = [
    "FixedIon",
    "InitNernstIon",
    "DynamicNernstIon",
]


class FixedIon:
    """Helper mixin for ions with fixed ``Ci/Co/E`` state."""

    def _init_fixed_ion(self, *, Ci=None, Co=None, E=None, valence=None):
        """Materialize one fixed ion payload onto ``self``.

        ``Ci``/``Co``/``valence`` fall back to species-level defaults when
        omitted. ``E`` must always be provided explicitly by the concrete fixed
        class, typically via its constructor default.
        """
        if E is None:
            raise ValueError(f"{type(self).__name__} requires an explicit fixed reversal potential E.")

        self.Ci = braintools.init.param(
            type(self).default_Ci if Ci is None else Ci,
            self.varshape,
            allow_none=False,
        )
        self.Co = braintools.init.param(
            type(self).default_Co if Co is None else Co,
            self.varshape,
            allow_none=False,
        )
        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.valence = braintools.init.param(
            type(self).default_valence if valence is None else valence,
            self.varshape,
            allow_none=False,
        )


class InitNernstIon:
    """Helper mixin for ions with fixed ``Ci/Co`` and stored Nernst ``E``.

    ``E`` is stored as a regular attribute and refreshed during
    ``init_state()`` / ``reset_state()`` via ion-level hooks.
    """

    def _init_nernst_ion(self, *, Ci=None, Co=None, temp=None, valence=None):
        """Initialize the fixed concentrations needed by a stored-Nernst ion."""
        if temp is None:
            raise ValueError(f"{type(self).__name__} requires an explicit temperature value.")

        self.Ci = braintools.init.param(
            type(self).default_Ci if Ci is None else Ci,
            self.varshape,
            allow_none=False,
        )
        self.Co = braintools.init.param(
            type(self).default_Co if Co is None else Co,
            self.varshape,
            allow_none=False,
        )
        self.valence = braintools.init.param(
            type(self).default_valence if valence is None else valence,
            self.varshape,
            allow_none=False,
        )
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.E = None

    def _update_reversal(self):
        """Recompute and store ``E`` from the current ``Ci/Co/temp/valence``."""
        Ci = self.Ci.value if isinstance(self.Ci, brainstate.State) else self.Ci
        Co = self.Co.value if isinstance(self.Co, brainstate.State) else self.Co
        valence = self.valence.value if isinstance(self.valence, brainstate.State) else self.valence
        temp = self.temp.value if isinstance(self.temp, brainstate.State) else self.temp
        self.E = (
            u.gas_constant * temp / (valence * u.faraday_constant)
        ) * u.math.log(Co / Ci)

    def _ion_init_state_hook(self, V, batch_size: int = None):
        _ = (V, batch_size)
        self._update_reversal()

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        _ = (V, batch_size)
        self._update_reversal()


class DynamicNernstIon:
    """Helper mixin for ions with dynamic ``Ci`` and computed Nernst ``E``.

    ``Ci`` is created as a :class:`DiffEqState` during ``init_state()`` and is
    reset from the stored initializer during ``reset_state()``. ``E`` is a
    derived property computed on demand from the current ``Ci``.
    """

    #: When true, the template precomputes the aggregate ion current and passes
    #: it to ``derivative(..., total_current=...)``. Leave false for models that
    #: do not need current-driven concentration dynamics.
    uses_total_current = False

    def _init_dynamic_nernst_ion(self, *, Co=None, temp=None, valence=None, Ci_initializer=None):
        """Initialize the static fields and remember the ``Ci`` initializer."""
        if temp is None:
            raise ValueError(f"{type(self).__name__} requires an explicit temperature value.")

        self.Co = braintools.init.param(
            type(self).default_Co if Co is None else Co,
            self.varshape,
            allow_none=False,
        )
        self.valence = braintools.init.param(
            type(self).default_valence if valence is None else valence,
            self.varshape,
            allow_none=False,
        )
        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self._Ci_initializer = type(self).default_Ci if Ci_initializer is None else Ci_initializer

    @property
    def E(self):
        """Compute ``E`` from the current dynamic ``Ci`` via Nernst."""
        Ci = self.Ci.value if isinstance(self.Ci, brainstate.State) else self.Ci
        Co = self.Co.value if isinstance(self.Co, brainstate.State) else self.Co
        valence = self.valence.value if isinstance(self.valence, brainstate.State) else self.valence
        temp = self.temp.value if isinstance(self.temp, brainstate.State) else self.temp
        return (u.gas_constant * temp / (valence * u.faraday_constant)) * u.math.log(Co / Ci)

    def _ion_init_state_hook(self, V, batch_size: int = None):
        """Create the runtime ``Ci`` state from the stored initializer."""
        _ = V
        self.Ci = DiffEqState(
            braintools.init.param(self._Ci_initializer, self.varshape, batch_size),
        )

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        """Reset the dynamic ``Ci`` state back to its initializer."""
        _ = V
        value = braintools.init.param(
            self._Ci_initializer,
            self.varshape,
            batch_size,
        )
        self.Ci.value = value
        if isinstance(batch_size, int):
            assert value.shape[0] == batch_size

    def _ion_compute_derivative_hook(self, V):
        """Populate ``Ci.derivative`` using the concrete ion model."""
        total_current = None
        if type(self).uses_total_current:
            total_current = self.current(V, include_external=True)
        self.Ci.derivative = self.derivative(
            self.Ci.value,
            V,
            total_current=total_current,
        )

    def derivative(self, Ci, V, total_current=None):
        """Return ``dCi/dt`` for the concrete dynamic ion model."""
        raise NotImplementedError
