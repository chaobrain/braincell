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

"""Chloride ion species: fixed, Nernst-derived and the JE2011 radial pool."""

import math
from typing import Optional

import braintools
import brainunit as u

from braincell._base_ion import Ion
from braincell._typing import Initializer, Size
from braincell.ion._base import FixedIon, InitNernstIon, KineticIon, Factor, Reaction, Source, Species
from braincell.mech import register_ion

__all__ = [
    'Chloride',
    'ChlorideFixed',
    'ChlorideInitNernst',
    'Cldif_JE2011',
]


class Chloride(Ion):
    """Base class for modeling the chloride ion species.

    It holds the defaults shared by the concrete chloride models and has
    no dynamics of its own. The family key is ``"cl"``, as in NEURON.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target.
    name : str or None, optional
        Runtime instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion.

    Notes
    -----
    The default concentrations are the values the Jedlicka et al. (2011)
    simulation sets in hoc; they are conventions, not measurements.

    Attributes
    ----------
    ion_symbol : str
        Symbol used for runtime family lookup. Set to ``'Cl'``.
    default_Ci : brainunit.Quantity
        Default intracellular chloride concentration, ``5 mM``.
    default_Co : brainunit.Quantity
        Default extracellular chloride concentration, ``133.5 mM``.
    default_valence : int
        Default ionic valence, ``-1``.
    """

    __module__ = 'braincell.ion'
    ion_symbol = 'Cl'
    default_Ci = 5.0 * u.mM
    default_Co = 133.5 * u.mM
    default_valence = -1


@register_ion("ChlorideFixed")
class ChlorideFixed(Chloride, FixedIon):
    """Chloride ion with a fixed reversal potential and fixed concentrations.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target.
    E : array-like or callable, optional
        Fixed reversal potential. Defaults to ``-87.5 mV``, the Nernst
        value of the class-default concentrations at 36 degrees Celsius
        (``-87.505 mV``) rounded to 0.1 mV. Passing ``None`` raises
        :class:`ValueError`.
    Ci, Co : array-like or callable or None, optional
        Intra- and extracellular concentrations. ``None`` falls back to
        :attr:`Chloride.default_Ci` and :attr:`Chloride.default_Co`.
    valence : array-like or callable or None, optional
        Ionic valence. ``None`` falls back to
        :attr:`Chloride.default_valence` (``-1``).
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion.

    Raises
    ------
    ValueError
        If ``E`` is explicitly passed as ``None``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.ion import ChlorideFixed
        >>> float(ChlorideFixed(size=1).E.to_decimal(u.mV))
        -87.5
    """

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: Size,
        E: Optional[Initializer] = -87.5 * u.mV,
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)


@register_ion("ChlorideInitNernst")
class ChlorideInitNernst(Chloride, InitNernstIon):
    r"""Chloride ion with fixed concentrations and a Nernst-initialized ``E``.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target.
    temp : array-like or callable, optional
        Absolute temperature of the Nernst equation. Defaults to 36
        degrees Celsius in kelvin, as for the sodium, potassium and
        calcium siblings.
    Ci, Co : array-like or callable or None, optional
        Intra- and extracellular concentrations. ``None`` falls back to
        :attr:`Chloride.default_Ci` and :attr:`Chloride.default_Co`.
    valence : array-like or callable or None, optional
        Ionic valence. ``None`` falls back to ``-1``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion.

    Raises
    ------
    ValueError
        If ``temp`` is explicitly passed as ``None``.

    Notes
    -----
    ``E = RT/(zF) log(Co/Ci)`` with ``z = -1`` is computed on
    ``init_state`` and ``reset_state``: ``-87.505 mV`` at 36 and
    ``-87.222 mV`` at 35 degrees Celsius for the default concentrations.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.ion import ChlorideInitNernst
        >>> cl = ChlorideInitNernst(size=1, temp=u.celsius2kelvin(35.0))
        >>> cl.init_state(-65.0 * u.mV)
        >>> cl.E.to_decimal(u.mV)
        Array(-87.221985, dtype=float32)
    """

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(36.0),
        Ci: Optional[Initializer] = None,
        Co: Optional[Initializer] = None,
        valence: Optional[Initializer] = None,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size, name=name, **channels)
        self._init_nernst_ion(Ci=Ci, Co=Co, temp=temp, valence=valence)


def _cldif_shells(n: int = 4):
    """Return ``(vrat, frat)`` of ``cldif.mod``'s ``factors()`` for a 1 um diameter."""
    r = 0.5
    dr2 = r / (n - 1) / 2.0
    vrat = [0.0] * n
    frat = [0.0] * n
    frat[0] = 2.0 * r
    for i in range(n - 1):
        vrat[i] += math.pi * (r - dr2 / 2.0) * 2.0 * dr2
        r -= dr2
        frat[i + 1] = 2.0 * math.pi * r / (2.0 * dr2)
        r -= dr2
        vrat[i + 1] = math.pi * (r + dr2 / 2.0) * 2.0 * dr2
    return tuple(vrat), tuple(frat)


_VRAT, _FRAT = _cldif_shells()
_SHELLS = ("Ci", "cl1", "cl2", "cl3")


def _shell_factor(i):
    return Factor(f"vol{i}", lambda self: self._diameter() ** 2 * _VRAT[i])


def _diffusion(i):
    rate = lambda self, V, x: self.DCl * _FRAT[i + 1]
    return Reaction({_SHELLS[i]: 1}, {_SHELLS[i + 1]: 1}, rate, rate)


def _current_influx(self, V, x, total_current=None):
    """``-pi d I / F``: an inward (positive) current is a Cl- efflux, ``z = -1``."""
    if total_current is None:
        return 0.0 * self.cli0 * self._diameter() ** 2 / self.tau
    return -(total_current * u.math.pi * self._diameter()) / u.faraday_constant


def _pump(self, V, x, total_current=None):
    """First-order relaxation to ``cli0``, explicit in ``Ci`` like NMODL's ``<<``."""
    ci = getattr(self, "_ci_step_start", None)
    if ci is None:
        ci = x["Ci"]
    return self._diameter() ** 2 * _VRAT[0] * (self.cli0 - ci) / self.tau


@register_ion("Cldif_JE2011")
class Cldif_JE2011(Chloride, KineticIon):
    r"""Radial-diffusion chloride pool of Jedlicka et al. (2011), ``cldif.mod``.

    The compartment is split into four concentric shells [1]_, ``Ci`` (the
    outermost, read by the Nernst potential) and ``cl1``-``cl3``.
    With :math:`d` the compartment's ``diam_arc_mean``, the shell
    volume per unit length is :math:`V_i = d^2\,\mathrm{vrat}_i`, and in
    the scaled variables :math:`y_i = V_i c_i`

    .. math::

        \dot y_0 = -\frac{\pi d}{F} I_{Cl}
            + V_0 \frac{c_{rest} - c_0^{(n)}}{\tau}
            + D\,\mathrm{frat}_1 (c_1 - c_0),

        \dot y_i = D\,\mathrm{frat}_i (c_{i-1} - c_i)
            + D\,\mathrm{frat}_{i+1} (c_{i+1} - c_i),
            \quad i = 1, 2, 3,\ \mathrm{frat}_4 = 0,

    with :math:`I_{Cl}` the inward-positive chloride current density,
    :math:`c_{rest}` = ``cli0``, and ``vrat``, ``frat`` from ``cldif.mod``.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target.
    temp : array-like or callable, optional
        Absolute temperature. Defaults to 35 degrees Celsius in kelvin, as
        in the published hoc.
    DCl : array-like or callable, optional
        Radial diffusion coefficient. Defaults to ``2 um**2/ms``.
    tau : array-like or callable, optional
        Time constant of the "pump" relaxation of ``Ci`` to ``cli0``.
        Defaults to ``3000 ms``.
    cli0 : array-like or callable, optional
        Resting concentration and initial value of all shells. Defaults to
        ``5 mM`` as in the published hoc (the ``.mod`` default is 8 mM).
    Co : array-like or callable or None, optional
        Extracellular concentration. ``None`` falls back to
        :attr:`Chloride.default_Co` (``133.5 mM``).
    species_initializers : dict or None, optional
        Initial values by shell name (``"Ci"``, ``"cl1"``, ``"cl2"``,
        ``"cl3"``); a shell not named starts at ``cli0``.
    solver : str, optional
        Independent integrator. Defaults to ``"backward_euler"``.
    substeps : int, optional
        Solver substeps per parent step. Defaults to ``1``.
    name : str or None, optional
        Runtime ion instance name. Defaults to ``None``.
    **channels
        Channel instances to attach to this ion.

    Raises
    ------
    ValueError
        If ``temp`` is ``None``, ``substeps < 1``, or
        ``species_initializers`` names an unknown shell.
    AttributeError
        From ``init_state`` when ``diam_arc_mean`` (set by a Cell) is missing.

    Notes
    -----
    The pump term reads ``Ci`` at the start of each solver substep, so it
    is explicit as NMODL's ``~ cl[0] << (flux)`` is, while diffusion is
    implicit under ``backward_euler``. Evaluating it at the current
    ``Ci`` (implicit) changes ``Ci`` by a relative 8e-7 over 1 s; it is
    not wrapped in ``stop_gradient``, so gradients through ``Ci`` stay
    exact. Called outside a solver step (``compute_derivative``), the
    pump reads the current ``Ci``.

    ``cldif.mod`` also diffuses chloride along the cable
    (``LONGITUDINAL_DIFFUSION``). BrainCell integrates each ion per
    control volume, with no ion flux between them, so this class matches
    ``cldifus`` on a single compartment only; on a multi-compartment
    dendrite local accumulation is larger than in NEURON.

    References
    ----------
    .. [1] Jedlicka, P., Deller, T., Gutkin, B. S., & Backus, K. H. (2011).
           Activity-dependent intracellular chloride accumulation and
           diffusion controls GABA(A) receptor-mediated synaptic
           transmission. Hippocampus, 21(8), 885-898.
           doi:10.1002/hipo.20804

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.ion import Cldif_JE2011
        >>> cl = Cldif_JE2011(size=1)
        >>> cl.diam_arc_mean = 1.0 * u.um  # set by a Cell
        >>> cl.init_state(-70.0 * u.mV)
        >>> cl.E.to_decimal(u.mV)
        Array([-87.221985], dtype=float32)
    """

    __module__ = 'braincell.ion'

    uses_total_current = True
    factors = tuple(_shell_factor(i) for i in range(4))
    species = tuple(Species(name, init=5.0 * u.mM, factor=f"vol{i}") for i, name in enumerate(_SHELLS))
    reactions = tuple(_diffusion(i) for i in range(3))
    sources = (Source("Ci", _current_influx), Source("Ci", _pump))

    def __init__(
        self,
        size: Size,
        temp: Initializer = u.celsius2kelvin(35.0),
        DCl: Initializer = 2.0 * u.um**2 / u.ms,
        tau: Initializer = 3000.0 * u.ms,
        cli0: Initializer = 5.0 * u.mM,
        Co: Optional[Initializer] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels,
    ):
        super().__init__(size=size, name=name, **channels)
        self.DCl = braintools.init.param(DCl, self.varshape, allow_none=False)
        self.tau = braintools.init.param(tau, self.varshape, allow_none=False)
        self.cli0 = braintools.init.param(cli0, self.varshape, allow_none=False)
        initializers = self._resolve_species_initializers(
            Ci_initializer=None,
            species_initializers=species_initializers,
        )
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _default_species_initializers(self, Ci_initializer) -> dict[str, object]:
        return dict.fromkeys(_SHELLS, self.cli0)

    def _diameter(self):
        if not hasattr(self, "diam_arc_mean"):
            raise AttributeError(f"{type(self).__name__} requires 'diam_arc_mean' before kinetic state initialization.")
        return self.diam_arc_mean

    def _step_solver(self, V, recursive_child: bool = True):
        self._ci_step_start = self.Ci.value
        try:
            super()._step_solver(V, recursive_child)
        finally:
            del self._ci_step_start

    def _ion_init_state_hook(self, V, batch_size: int = None):
        self._diameter()
        super()._ion_init_state_hook(V, batch_size=batch_size)
