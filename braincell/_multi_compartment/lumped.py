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

"""Lumped runtime of a :class:`~braincell.Cell` with ``cv_policy=Lumped()``.

A lumped Cell keeps the cable Cell's declaration phase (paint / place rules,
the one-CV discretization, ``cvs``) and replaces only the runtime. At
``init_state`` this module lowers the single CV into:

- :class:`LumpedRuntime`, a point-neuron engine with the voltage, derivative
  and update of :class:`~braincell.SingleCompartment`, whose mechanisms are
  instantiated with ``size=pop_size`` -- there is no CV axis;
- :class:`LumpedClamps`, which evaluates the placed current clamps with the
  same helpers as the cable runtime and returns their total current.

Parameters are resolved on the one CV by the cable rules (callables of
:class:`~braincell.mech.CVContext` evaluated at the CV, ion initializers
sampled once) and checked against the same unit prototypes, with the same
error messages, as on a cable Cell. Channels are bound to ions by the same
code as the cable runtime (``root_type``, ``ion_name`` / ``ion_names``,
placeholder ions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._compute.bindings import apply_subsolver_schedule, install_channel_on_ions
from braincell._compute.bridge import _ION_GEOMETRY_ATTRS, broadcast_to_shape
from braincell._compute.ions import (
    _build_ion_alias_map,
    _placeholder_family_keys,
    _runtime_ion_species_key,
    _supported_ion_runtime_params,
)
from braincell._compute.layouts import (
    _allocate_current_clamp_buffer,
    _allocate_current_clamp_delay_buffer,
    _allocate_state_buffer,
    _eval_function_clamp,
    _eval_sine_clamp,
    _mechanism_var_names,
    _spatial_density_error_prefix,
    _time_ge,
    _time_lt,
)
from braincell._compute.parameters import (
    density_error_context,
    density_parameter_names,
    density_parameter_schema,
    density_parameter_value,
    spatial_callable_source,
)
from braincell._single_compartment import SingleCompartment
from braincell.ion import build_placeholder_ions
from braincell.mech import CurrentClamp, Density, FunctionClamp, SineClamp, get_registry

if TYPE_CHECKING:
    from braincell._multi_compartment.cell import Cell

__all__ = ["CLAMP_TYPES", "LumpedClamps", "LumpedRuntime", "build_lumped_runtime"]

#: Point mechanisms a lumped Cell accepts in ``place``.
CLAMP_TYPES = (CurrentClamp, SineClamp, FunctionClamp)

#: Solvers that need the cable point tree and so cannot step a lumped Cell.
CABLE_ONLY_SOLVERS = frozenset({"staggered", "dhs_voltage"})


class LumpedRuntime(SingleCompartment):
    """Point-neuron engine of a lumped Cell.

    It is :class:`~braincell.SingleCompartment` with the membrane area of the
    Cell's one CV in place of ``2 pi radius length``. The Cell calls
    :meth:`update` with the clamp current density as ``I_ext``.

    Parameters
    ----------
    size : tuple of int
        The Cell's ``pop_size``.
    area : Quantity
        Membrane area of the CV.
    C : Quantity
        Specific membrane capacitance of the CV.
    V_th : Quantity
        Spike threshold, scalar or shaped ``size``.
    V_initializer : Quantity
        Initial voltage shaped ``size``.
    spk_fun : callable
        Surrogate spike function.
    solver : callable
        Step function from :func:`braincell.quad.get_integrator`.
    ion_channels : dict
        Ions and root-level channels, keyed by attribute name.
    """

    __module__ = "braincell._multi_compartment.lumped"

    def __init__(self, size, *, area, C, V_th, V_initializer, spk_fun, solver, ion_channels):
        super().__init__(
            size,
            C=C,
            V_th=V_th,
            V_initializer=V_initializer,
            spk_fun=spk_fun,
            solver=solver,
            **ion_channels,
        )
        self._cv_area = area

    @property
    def area(self):
        """Membrane area of the Cell's one CV."""
        return self._cv_area


def build_lumped_runtime(cell: "Cell", *, V_init, V_th) -> LumpedRuntime:
    """Lower the one CV of a lumped ``cell`` into a :class:`LumpedRuntime`.

    Parameters
    ----------
    cell : Cell
        A Cell whose ``cv_policy`` is :class:`~braincell.Lumped`.
    V_init, V_th : Quantity
        Materialized initial voltage and threshold, shaped ``cell.pop_size``.

    Returns
    -------
    LumpedRuntime
        Engine with mechanisms of size ``cell.pop_size``.

    Raises
    ------
    ValueError
        On the same declaration errors as the cable runtime: unsupported ion
        parameters, ion instance names that clash, an ion binding that cannot
        be resolved, a callable parameter that does not return a scalar, a
        parameter whose units do not match its prototype.
    TypeError
        A parameter without units where the prototype has them, as on the
        cable runtime.
    """
    (cv,) = cell.cvs
    (context,) = cell.cv_contexts
    pop_size = tuple(cell.pop_size)
    registry = get_registry()

    ions: dict[str, object] = {}
    family_candidates: dict[str, list[str]] = {}
    class_candidates: dict[str, list[str]] = {}
    schedule_pairs: list[tuple[Density, object]] = []
    channel_declarations: list[Density] = []

    # Discretization has already rejected an ion or channel painted twice on
    # the CV, so each instance name appears once here.
    for declaration in cv.density_mech:
        if declaration.category == "channel":
            channel_declarations.append(declaration)
            continue
        runtime_cls = registry.get("ion", declaration.class_name)
        species_key = _runtime_ion_species_key(runtime_cls)
        name = declaration.instance_name
        if name in _placeholder_family_keys() and name != species_key:
            raise ValueError(
                f"Ion instance name {name!r} conflicts with canonical family key for a different ion family."
            )
        invalid = set(declaration.params) - set(_supported_ion_runtime_params(runtime_cls))
        if invalid:
            raise ValueError(
                f"Ion instance {name!r} uses unsupported runtime ion params {sorted(invalid)!r} "
                f"on {runtime_cls.__name__!r}."
            )
        node = runtime_cls(size=pop_size, name=name, **_resolve_params(declaration, context, cell.paint_rules))
        ions[name] = node
        family_candidates.setdefault(species_key, []).append(name)
        class_candidates.setdefault(runtime_cls.__name__, []).append(name)
        schedule_pairs.append((declaration, node))

    for family_key, default_ion in build_placeholder_ions(size=pop_size).items():
        if family_key in family_candidates:
            continue
        ions[family_key] = default_ion
        family_candidates[family_key] = [family_key]
        class_candidates.setdefault(type(default_ion).__name__, []).append(family_key)

    ion_aliases = _build_ion_alias_map(
        ions=ions,
        ion_family_candidates=family_candidates,
        ion_class_candidates=class_candidates,
    )
    frozen_candidates = {key: tuple(value) for key, value in family_candidates.items()}
    _attach_ion_geometry(ions, cv=cv, pop_size=pop_size)

    root_channels: dict[str, object] = {}
    for index, declaration in enumerate(channel_declarations):
        runtime_cls = registry.get("channel", declaration.class_name)
        node = runtime_cls(size=pop_size, **_resolve_params(declaration, context, cell.paint_rules))
        _, owner = install_channel_on_ions(
            node,
            runtime_cls=runtime_cls,
            mechanism=declaration,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=frozen_candidates,
            layout_id=index,
        )
        if owner is None:
            root_channels[_unique_key(root_channels, ions, declaration.instance_name)] = node
        schedule_pairs.append((declaration, node))

    apply_subsolver_schedule(schedule_pairs, solver=cell.subsolver, substeps=cell.substeps)

    return LumpedRuntime(
        pop_size,
        area=cv.area,
        C=cv.cm,
        V_th=V_th,
        V_initializer=V_init,
        spk_fun=cell.spk_fun,
        solver=cell.solver,
        ion_channels={**ions, **root_channels},
    )


def _resolve_params(declaration: Density, context, paint_rules) -> dict[str, object]:
    """Resolve and check declared parameters on the one CV, as the cable runtime does."""
    schema = density_parameter_schema(declaration)
    # The cable runtime checks the parameters it stores as numeric buffers;
    # configuration such as ``species_initializers`` is passed through.
    checked = set(density_parameter_names(declaration)) & set(schema)
    owner, source = density_error_context(declaration, paint_rules)
    params = {}
    for name in declaration.params:
        value = density_parameter_value(declaration, name)
        value_source = source
        if callable(value) and not isinstance(value, braintools.init.Initialization):
            value_source = spatial_callable_source(value) + (f"; {source}" if source else ".")
            value = _evaluate_spatial_param(declaration, name, value, context)
        # A Channel initializer is sampled by the runtime class itself.
        if name in checked and not isinstance(value, braintools.init.Initialization):
            schema[name].validate(value, name, owner=owner, source=value_source)
        params[name] = value
    return params


def _evaluate_spatial_param(declaration: Density, name: str, fn, context):
    """Evaluate a ``fn(CVContext)`` parameter at the CV and require a scalar."""
    prefix = _spatial_density_error_prefix(declaration, name, context)
    try:
        result = fn(context)
    except Exception as exc:
        raise ValueError(f"{prefix} callable raised {type(exc).__name__}: {exc}") from exc
    shape = tuple(getattr(result, "shape", ()))
    if shape not in ((), (1,)):
        raise TypeError(f"{prefix} must return a scalar, got shape {shape!r}.")
    if isinstance(result, u.Quantity):
        return u.math.reshape(result, ())
    try:
        return float(np.asarray(result, dtype=np.float64).reshape(()))
    except Exception as exc:
        raise TypeError(f"{prefix} must return a numeric scalar or a Quantity, got {result!r}.") from exc


def _attach_ion_geometry(ions: dict[str, object], *, cv, pop_size: tuple[int, ...]) -> None:
    """Give each ion the CV geometry (``area``, ``diam_mid``, ...) shaped ``pop_size``."""
    for ion in ions.values():
        for attr_name in _ION_GEOMETRY_ATTRS:
            setattr(ion, attr_name, broadcast_to_shape(getattr(cv, attr_name), pop_size, name=f"ion.{attr_name}"))


def _unique_key(taken: dict[str, object], ions: dict[str, object], name: str) -> str:
    if name not in taken and name not in ions:
        return name
    suffix = 2
    while f"{name}_{suffix}" in taken or f"{name}_{suffix}" in ions:
        suffix += 1
    return f"{name}_{suffix}"


class LumpedClamps:
    """Current clamps placed on a lumped Cell.

    Buffers are allocated by the cable runtime's clamp helpers on a one-row
    layout, so step, sine and function clamps keep their cable semantics
    (timing tolerance, delays, per-member values). Step-clamp start and end
    times are static, so they are tabulated once here rather than summed on
    every step, and the per-step work is elementwise.

    Parameters
    ----------
    cell : Cell
        A declared lumped Cell. Its clamp store fixes the order of the
        components returned by :meth:`evaluate`.
    """

    def __init__(self, cell: "Cell", *, density_unit, with_components: bool = True) -> None:
        self.pop_size = tuple(cell.pop_size)
        self.with_components = bool(with_components)
        (cv,) = cell.cvs
        # nA -> density_unit over the CV area, applied once to the static tables.
        self.density_unit = density_unit
        self._per_area = float((1.0 * u.nA / cv.area).to_decimal(density_unit))
        self.n_population = int(np.prod(self.pop_size, dtype=int))
        store = cell._get_clamp_store()
        self.n_component = len(store.id)
        self.state_buffers: dict[tuple[int, str], object] = {}
        self._entries: list[tuple[int, object]] = []
        # ``Cell.place`` admits only clamps on a lumped Cell, and only on the
        # whole population (population views raise), so every placement
        # covers every member. The clamp store numbers rows in placement
        # order, so concatenating per-placement currents gives store order.
        for placement in cell.point_placements:
            key = int(placement.id)
            self._allocate(key, placement.mechanism)
            self._entries.append((key, self._tabulate(key, placement.mechanism)))

    def __len__(self) -> int:
        return len(self._entries)

    def _allocate(self, key: int, mechanism) -> None:
        for var_name in _mechanism_var_names(mechanism):
            if isinstance(mechanism, CurrentClamp) and var_name == "delay":
                self.state_buffers[(key, var_name)] = _allocate_current_clamp_delay_buffer(
                    mechanism=mechanism, pop_size=self.pop_size, n_active=1
                )
            elif isinstance(mechanism, CurrentClamp) and var_name in ("durations", "amplitudes"):
                quantity, mask = _allocate_current_clamp_buffer(
                    mechanism=mechanism, var_name=var_name, pop_size=self.pop_size, n_active=1
                )
                self.state_buffers[(key, var_name)] = quantity
                self.state_buffers[(key, f"_mask_{var_name}")] = mask
            else:
                self.state_buffers[(key, var_name)] = _allocate_state_buffer(
                    mechanism, var_name=var_name, shape=self.pop_size + (1,)
                )

    def _tabulate(self, key: int, mechanism):
        """Static step table of a CurrentClamp (``_eval_current_clamp`` math)."""
        if not isinstance(mechanism, CurrentClamp):
            return type(mechanism).__name__
        durations = np.asarray(self.state_buffers[(key, "durations")].to_decimal(u.ms))[..., 0, :]
        ends = np.cumsum(durations, axis=-1)
        starts = np.concatenate([np.zeros_like(ends[..., :1]), ends[..., :-1]], axis=-1)
        return {
            "delay": np.asarray(self.state_buffers[(key, "delay")].to_decimal(u.ms))[..., 0],
            "starts": starts,
            "ends": ends,
            "amplitudes": np.asarray(self.state_buffers[(key, "amplitudes")].to_decimal(u.nA))[..., 0, :],
            "mask": np.asarray(self.state_buffers[(key, "_mask_durations")])[..., 0, :],
        }

    def _placement_current(self, key: int, table, t):
        """Current of one placement at time ``t`` in nA (decimal), shaped ``pop_size``."""
        if isinstance(table, dict):
            local_t = jnp.expand_dims(jnp.asarray(t.to_decimal(u.ms)) - table["delay"], axis=-1)
            active = (
                _time_ge(local_t, 0.0)
                & _time_ge(local_t, table["starts"])
                & _time_lt(local_t, table["ends"])
                & table["mask"]
            )
            return jnp.sum(jnp.where(active, table["amplitudes"], 0.0), axis=-1)
        rows = np.zeros((1,), dtype=np.int64)
        if table == "FunctionClamp":
            value = _eval_function_clamp(self, layout_id=key, local_index=0, t=t)
            return jnp.broadcast_to(jnp.asarray(value.to_decimal(u.nA)), self.pop_size)
        delay = self.state_buffers[(key, "delay")]
        local_t = (t - u.Quantity(delay.mantissa[..., rows], delay.unit)).in_unit(u.ms)
        value = _eval_sine_clamp(self, layout_id=key, rows=rows, local_t=local_t)
        return jnp.broadcast_to(value.to_decimal(u.nA)[..., 0], self.pop_size)

    def evaluate(self, t):
        """Return ``(density, components)`` at time ``t``.

        Returns
        -------
        density : Quantity
            Summed clamp current per population member divided by the CV
            area, in ``density_unit``, shaped ``pop_size``.
        components : Quantity or None
            Current of each logical clamp in clamp-store row order, nA;
            ``None`` unless ``with_components``.
        """
        total = 0.0
        parts = []
        for key, table in self._entries:
            value = self._placement_current(key, table, t)
            total = total + value
            if self.with_components:
                parts.append(value.reshape(-1))
        density = u.Quantity(jnp.broadcast_to(jnp.asarray(total) * self._per_area, self.pop_size), self.density_unit)
        if not self.with_components:
            return density, None
        components = jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=float)
        return density, u.Quantity(components, u.nA)
