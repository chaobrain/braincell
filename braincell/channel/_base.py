# -*- coding: utf-8 -*-


from dataclasses import dataclass
import inspect
import warnings
import numpy as np
from typing import Any, Optional
from typing import ClassVar

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

from braincell._base import Channel
from braincell._misc import is_traced_value
from braincell.quad.protocol import DiffEqState
from braincell.quad.protocol import IndependentIntegration

__all__ = [
    "Gate",
    "Transition",
    "HH",
    "OhmicHH",
    "Markov",
    "ghk_flux",
]


def ghk_flux(V, ci, co, z, temp):
    """Unit-aware GHK flux helper with a small-zeta stable branch."""
    zeta = (z * u.faraday_constant * V) / (u.gas_constant * temp)
    exp_term = u.math.exp(-zeta)
    numerator = ci - co * exp_term
    small_branch = (z * u.faraday_constant) * numerator * (1 + zeta / 2)
    regular_branch = (z * zeta * u.faraday_constant) * numerator / (1 - exp_term)
    return u.math.where(u.math.abs(1 - exp_term) <= 1e-6, small_branch, regular_branch)


def _resolve_value(owner, value):
    """Resolve a piece of gate metadata against the owning channel instance.

    Gate metadata is declared at class level but often refers to a
    constructor parameter. A ``str`` names that attribute
    (``q10="q10"``); a callable is applied to the instance
    (``q10=lambda self: self.q10``), which predates the string form and
    stays supported; anything else is a literal.
    """
    if isinstance(value, str):
        try:
            return getattr(owner, value)
        except AttributeError:
            raise AttributeError(
                f"{type(owner).__name__}: gate metadata references attribute {value!r}, "
                f"which the instance does not define."
            ) from None
    return value(owner) if callable(value) else value


def _rate_ion_count(owner_type: type, rate_name: str) -> int | None:
    """Return how many ion arguments ``rate_name`` declares, or ``None`` for ``*args``.

    Results are memoised on the owning class rather than in a module-level
    cache, so a garbage-collected channel class takes its entry with it.
    """
    cache = owner_type.__dict__.get("_rate_ion_counts")
    if cache is None:
        cache = {}
        setattr(owner_type, "_rate_ion_counts", cache)
    if rate_name in cache:
        return cache[rate_name]

    signature = inspect.signature(getattr(owner_type, rate_name))
    params = tuple(signature.parameters.values())
    positional = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
        count = None
    else:
        positional_count = sum(1 for param in params if param.kind in positional)
        count = max(0, positional_count - 2)

    cache[rate_name] = count
    return count


def _call_rate(owner, rate_name: str, V, *ions):
    """Call a rate method with only the ion arguments it declares.

    Shared by both templates. A rate or gate method on a multi-ion channel
    often depends on a single ion: declaring ``f_m_inf(self, V, K)`` on a
    ``JointTypes[Potassium, Sodium, NonSpecific]`` channel receives just
    potassium, while a ``*args`` signature still receives every ion.

    ``MixIons`` builds ion arguments in the declaration order of the joint
    root type (``braincell/_base_ion.py``), so taking a prefix is well
    defined.
    """
    ion_count = _rate_ion_count(type(owner), rate_name)
    if ion_count is None:
        return getattr(owner, rate_name)(V, *ions)
    if ion_count > len(ions):
        raise TypeError(f"{type(owner).__name__}.{rate_name} expects {ion_count} ion argument(s), got {len(ions)}.")
    return getattr(owner, rate_name)(V, *ions[:ion_count])


def _resolve_declarations(raw, factory):
    """Normalise a class-level declaration tuple into ``factory`` instances.

    Both templates accept either the dataclass itself or the plain tuple of
    its fields, so ``gates = (("m", 3),)`` and ``gates = (Gate("m", 3),)``
    mean the same thing.
    """
    return tuple(item if isinstance(item, factory) else factory(*item) for item in raw)


def _check_identifier(cls: type, name: str, kind: str) -> None:
    """Reject a declared name that could not be bound as an attribute.

    Gate and Markov state names become attributes on the instance, so a name
    that is not an identifier fails much later and far less legibly.
    """
    if not name.isidentifier():
        raise ValueError(f"{cls.__name__}: {kind} name {name!r} is not a valid Python identifier.")


def _bind_state(owner, name: str, value, kind: str) -> None:
    """Attach a ``DiffEqState`` to ``owner`` without clobbering a parameter.

    ``init_state`` assigns one state per gate / Markov state by name. Without
    this guard a gate named after a constructor parameter silently replaces
    that parameter, which is impossible to diagnose downstream. Re-running
    ``init_state`` is legitimate, so an existing ``DiffEqState`` is replaced.
    """
    existing = getattr(owner, name, None)
    if existing is not None and not isinstance(existing, DiffEqState):
        raise ValueError(
            f"{type(owner).__name__}: {kind} {name!r} collides with the existing attribute "
            f"{name!r} of type {type(existing).__name__}. Rename the {kind} or the attribute."
        )
    setattr(owner, name, value)


@dataclass(frozen=True)
class Gate:
    """Metadata for one HH gate.

    Parameters
    ----------
    name : str
        Gate name. The state is stored under this attribute and the rate
        methods are looked up as ``f_<name>_inf`` / ``f_<name>_tau`` or
        ``f_<name>_alpha`` / ``f_<name>_beta``.
    power : int
        Exponent applied to the gate in :meth:`HH.conductance_factor`.
    phi : float or str or callable, optional
        Explicit temperature factor. Mutually exclusive with ``q10``. A
        ``str`` names an instance attribute; see :func:`_resolve_value`.
    q10 : float or str or callable, optional
        Q10 coefficient, used together with ``temp_ref`` and ``self.temp``.
        Resolved the same way as ``phi``.
    temp_ref : Quantity or str or callable, optional
        Reference temperature for ``q10``, as an absolute temperature.
        Resolved the same way as ``phi``.
    time_unit : Unit, default ``u.ms``
        Unit assumed when a rate method returns a bare (dimensionless)
        value. ``f_<name>_tau`` returning ``5.0`` is read as ``5 * time_unit``
        and ``f_<name>_alpha`` returning ``0.3`` as ``0.3 / time_unit``.
        Rate methods that return a properly united quantity are used as-is
        and ignore this field.
    clip : bool
        Project the gate into ``[0, 1]`` when evaluating
        :meth:`HH.conductance_factor`. Off by default: NEURON does not clip
        HH gates either, so clipping would diverge from the reference
        mechanisms this catalogue is validated against. Enable it per gate
        when a solver is expected to overshoot and an odd ``power`` would
        otherwise yield a negative conductance. The stored state is never
        rewritten -- only the value used for the conductance product.
    """

    name: str
    power: int = 1
    phi: Any | None = None
    q10: Any | None = None
    temp_ref: Any | None = None
    time_unit: Any = u.ms
    clip: bool = False

    def __post_init__(self):
        has_phi = self.phi is not None
        has_q10 = self.q10 is not None
        has_temp_ref = self.temp_ref is not None

        if has_phi and (has_q10 or has_temp_ref):
            raise ValueError(f"Gate {self.name!r}: phi cannot be provided together with q10/temp_ref.")
        if has_q10 != has_temp_ref:
            raise ValueError(f"Gate {self.name!r}: q10 and temp_ref must be provided together.")
        if u.get_dim(self.time_unit) != u.get_dim(u.ms):
            raise ValueError(f"Gate {self.name!r}: time_unit must have a time dimension, got {self.time_unit!r}.")


def _as_gate_quantity(value, gate: Gate, label: str, *, inverse: bool):
    """Interpret one gate rate return value against the gate's ``time_unit``.

    The two forms differ only in which side of the unit the value lands on:
    a time constant is read as ``value * time_unit``, a transition rate as
    ``value / time_unit``. A value that already carries a unit is checked
    against the expected dimension and passed through untouched.
    """
    expected = 1 / gate.time_unit if inverse else gate.time_unit
    if u.get_dim(value) == u.DIMENSIONLESS:
        return value / gate.time_unit if inverse else value * gate.time_unit
    if u.get_dim(value) != u.get_dim(expected):
        raise ValueError(
            f"Gate {gate.name!r}: {label} must be dimensionless (read as "
            f"{'1/' if inverse else ''}{gate.time_unit!r}) or carry "
            f"{'an inverse-time' if inverse else 'a time'} dimension, "
            f"got dimension {u.get_dim(value)}."
        )
    return value


def _as_time(value, gate: Gate, label: str):
    """Interpret a gate time constant, honouring an explicit unit if present."""
    return _as_gate_quantity(value, gate, label, inverse=False)


def _as_rate(value, gate: Gate, label: str):
    """Interpret a gate transition rate, honouring an explicit unit if present."""
    return _as_gate_quantity(value, gate, label, inverse=True)


def _as_markov_rate(value, owner_type: type, rate_name: str):
    """Normalise a Markov transition rate to a bare per-millisecond value.

    Markov rates are conventionally bare and implicitly per-millisecond:
    ``compute_derivative`` divides the accumulated derivative by ``u.ms`` once
    at the end, and the steady-state solvers strip units with
    ``u.get_magnitude``. Both of those are applied blind, so whatever a rate
    drags in survives them -- a rate built from a dimensioned ``phi`` used to
    yield a ``mV / ms`` derivative, and a silently unit-stripped generator
    matrix. Check here, where the offending rate method can still be named.
    """
    dim = u.get_dim(value)
    if dim == u.DIMENSIONLESS:
        return value
    if dim != u.get_dim(1 / u.ms):
        raise ValueError(
            f"{owner_type.__name__}.{rate_name} must be dimensionless (read as 1/ms) or carry "
            f"an inverse-time dimension, got dimension {dim}. Check that phi/q10 are dimensionless."
        )
    # `u.ms ** -1` rather than `1 / u.ms`: the former is a Unit, which is what
    # `to_decimal` accepts; the latter is a Quantity and raises.
    return value.to_decimal(u.ms**-1)


def _check_derivative(value, label: str):
    """Check that a state derivative came out as an inverse time.

    The per-rate checks police what the rate methods return, but the
    derivative also picks up ``phi`` and ``conserve``, which are read off the
    instance and are never checked at construction. A dimensioned ``phi``
    therefore contaminates the derivative silently -- ``phi = 2 * u.mV``
    yields ``mV / ms`` -- and the integrator carries it without complaint.
    Catch it here, where the gate or state is still known, rather than letting
    it surface as a bare unit mismatch far downstream.
    """
    if u.get_dim(value) != u.get_dim(1 / u.ms):
        raise ValueError(
            f"{label}: derivative must have an inverse-time dimension, got "
            f"dimension {u.get_dim(value)}. Check that phi/q10 are dimensionless."
        )
    return value


@dataclass(frozen=True)
class Transition:
    """One directed/reversible transition used by Markov channels."""

    src: str
    dst: str
    forward: str
    backward: str | None = None


class HH(Channel):
    """HH gate dynamics with per-gate auto-detected form.

    For each gate ``g`` exactly one of the following method pairs must exist:

    - ``f_<g>_inf`` and ``f_<g>_tau``
    - ``f_<g>_alpha`` and ``f_<g>_beta``
    """

    gates: ClassVar[tuple[Gate | tuple[Any, ...], ...]] = ()
    _resolved_gates: ClassVar[tuple[Gate, ...]] = ()
    _gate_forms: ClassVar[dict[str, str]] = {}
    #: Populated lazily, per concrete class, by :func:`_rate_ion_count`.
    #: Annotated but deliberately not assigned: a default here would be one
    #: mutable dict shared by every subclass that never writes its own.
    _rate_ion_counts: ClassVar[dict[str, int | None]]

    def __init_subclass__(cls, **kwargs):
        """Resolve and validate ``gates`` once, when the subclass is created.

        Every check here used to fire at ``reset_state`` time or not at all.
        Classes declaring no gates are skipped so abstract intermediates
        (``HH`` itself, ``OhmicHH``, family base classes) stay definable.
        """
        super().__init_subclass__(**kwargs)

        resolved = _resolve_declarations(cls.gates, Gate)
        cls._resolved_gates = resolved
        if not resolved:
            cls._gate_forms = {}
            return

        seen: set[str] = set()
        forms: dict[str, str] = {}
        for gate in resolved:
            _check_identifier(cls, gate.name, "gate")
            if gate.name in seen:
                raise ValueError(f"{cls.__name__}: gate {gate.name!r} is declared more than once.")
            seen.add(gate.name)

            has_inf_tau = hasattr(cls, f"f_{gate.name}_inf") and hasattr(cls, f"f_{gate.name}_tau")
            has_alpha_beta = hasattr(cls, f"f_{gate.name}_alpha") and hasattr(cls, f"f_{gate.name}_beta")
            if has_inf_tau and has_alpha_beta:
                raise ValueError(
                    f"{cls.__name__}: gate {gate.name!r} defines both inf/tau and alpha/beta forms; choose one."
                )
            if has_inf_tau:
                forms[gate.name] = "inf_tau"
            elif has_alpha_beta:
                forms[gate.name] = "alpha_beta"
            else:
                raise ValueError(
                    f"{cls.__name__}: gate {gate.name!r} must define either "
                    f"f_{gate.name}_inf + f_{gate.name}_tau or f_{gate.name}_alpha + f_{gate.name}_beta."
                )
        cls._gate_forms = forms

    def _iter_gates(self) -> tuple[Gate, ...]:
        return type(self)._resolved_gates

    def _gate_state(self, gate: Gate) -> DiffEqState:
        return getattr(self, gate.name)

    def _gate_value(self, gate: Gate):
        return self._gate_state(gate).value

    def gate_phi(self, gate: Gate):
        """Resolve one gate's temperature factor.

        Resolution order is intentionally simple:

        1. explicit ``phi``
        2. ``q10`` + ``temp_ref`` using ``self.temp``
        3. default ``1.0``
        """
        if gate.phi is not None:
            return _resolve_value(self, gate.phi)
        if gate.q10 is not None:
            q10 = _resolve_value(self, gate.q10)
            temp_ref = _resolve_value(self, gate.temp_ref)
            return q10 ** (((self.temp - temp_ref) / u.kelvin) / 10.0)
        return 1.0

    def _gate_form(self, gate: Gate) -> str:
        return type(self)._gate_forms[gate.name]

    def _call_rate(self, rate_name: str, V, *ions):
        return _call_rate(self, rate_name, V, *ions)

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for gate in self._iter_gates():
            _bind_state(
                self,
                gate.name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
                "gate",
            )

    def conductance_factor(self, V, *ions):
        _ = (V, ions)
        product = 1.0
        for gate in self._iter_gates():
            value = self._gate_value(gate)
            if gate.clip:
                value = u.math.clip(value, 0.0, 1.0)
            product = product * (value if gate.power == 1 else value**gate.power)
        return product

    def reset_state(self, V, *ions, batch_size: int = None):
        for gate in self._iter_gates():
            form = self._gate_form(gate)
            if form == "inf_tau":
                value = self._call_rate(f"f_{gate.name}_inf", V, *ions)
            else:
                # Normalised so a mis-dimensioned rate is reported against its
                # gate here too, not as a bare unit mismatch from the sum.
                alpha = _as_rate(self._call_rate(f"f_{gate.name}_alpha", V, *ions), gate, "alpha")
                beta = _as_rate(self._call_rate(f"f_{gate.name}_beta", V, *ions), gate, "beta")
                value = alpha / (alpha + beta)
            self._gate_state(gate).value = value
            if isinstance(batch_size, int):
                assert value.shape[0] == batch_size

    def compute_derivative(self, V, *ions):
        for gate in self._iter_gates():
            value = self._gate_value(gate)
            phi = self.gate_phi(gate)
            form = self._gate_form(gate)
            if form == "inf_tau":
                inf = self._call_rate(f"f_{gate.name}_inf", V, *ions)
                tau = _as_time(self._call_rate(f"f_{gate.name}_tau", V, *ions), gate, "tau")
                derivative = phi * (inf - value) / tau
            else:
                alpha = _as_rate(self._call_rate(f"f_{gate.name}_alpha", V, *ions), gate, "alpha")
                beta = _as_rate(self._call_rate(f"f_{gate.name}_beta", V, *ions), gate, "beta")
                derivative = phi * (alpha * (1.0 - value) - beta * value)
            self._gate_state(gate).derivative = _check_derivative(derivative, f"Gate {gate.name!r}")


class OhmicHH(HH):
    r"""HH gating driven by an ohmic current, :math:`g \cdot \prod g_i^{p_i} \cdot (E - V)`.

    Most voltage-gated channels in the catalogue share exactly this current
    expression, so :class:`HH` covers only the gating and this subclass adds
    the driving force. Channels whose current is not ohmic -- GHK flux and
    permeability-scaled calcium channels -- inherit :class:`HH` directly and
    implement :meth:`current` themselves.

    The maximal conductance is read from ``self.g_max``. A channel that names
    it differently, or that scales the driving force some other way, overrides
    :meth:`current`.

    See Also
    --------
    HH : Gating-only template, for channels with a non-ohmic current.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.channel import Gate, OhmicHH
        >>> from braincell.ion import Sodium
        >>> class MyNa(OhmicHH):
        ...     root_type = Sodium
        ...     gates = (Gate("m", power=3), Gate("h"))
        ...     def __init__(self, size, g_max=10.0 * (u.mS / u.cm**2)):
        ...         super().__init__(size=size)
        ...         self.g_max = g_max
        ...     def f_m_inf(self, V, Na):
        ...         return 0.6
        ...     def f_m_tau(self, V, Na):
        ...         return 0.4
        ...     def f_h_inf(self, V, Na):
        ...         return 0.3
        ...     def f_h_tau(self, V, Na):
        ...         return 2.0

    The gates supply ``m ** 3 * h`` and :class:`OhmicHH` supplies
    ``g_max * m ** 3 * h * (E - V)``, so no ``current`` method is needed.
    """

    __module__ = "braincell.channel"

    def reversal_potential(self, V, *ions):
        """Return the reversal potential driving the current.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential. Unused by the default implementation; present
            so overrides can depend on it.
        *ions : IonInfo
            Ion states, in the declaration order of ``root_type``.

        Returns
        -------
        ArrayLike
            Reversal potential, as a voltage. Defaults to that of the first
            ion, which for a joint root type is the first entry of
            ``root_type.__args__``. Override to read a fixed ``self.E`` or a
            non-leading ion.
        """
        _ = V
        return ions[0].E

    def current(self, V, *ions):
        """Return the ohmic current through the channel.

        Parameters
        ----------
        V : ArrayLike
            Membrane potential.
        *ions : IonInfo
            Ion states, in the declaration order of ``root_type``.

        Returns
        -------
        ArrayLike
            ``g_max * conductance_factor(V, *ions) * (E - V)``, where ``E``
            comes from :meth:`reversal_potential`.
        """
        return self.g_max * self.conductance_factor(V, *ions) * (self.reversal_potential(V, *ions) - V)


class Markov(Channel, IndependentIntegration):
    """Probability-state channel kinetics described by transition pairs.

    ``pairs`` define one conserved probability pool. One state is eliminated
    and reconstructed algebraically from the others; ``dependent_state``
    names it. Declaring it is required in practice -- the fallback to "the
    last state discovered while scanning ``pairs``" is order-dependent, so
    reordering ``pairs`` would silently eliminate a different state, and it
    now emits a ``DeprecationWarning``.

    ``state_values()`` returns the raw stored states plus the reconstructed
    dependent state. ``compute_derivative()`` uses ``_kinetic_state_values()``,
    which clips each independent state to ``[0, 1]`` before evaluating the
    transition graph while still reconstructing the dependent state from the
    raw stored sum. Clipping is governed by :attr:`clip_states` and projects
    only the values fed to the kinetics; the stored states are untouched.

    Transition rates may return either a bare value, read as
    per-millisecond, or a properly united inverse time; anything else is
    rejected against the rate method's own name. This mirrors
    :attr:`Gate.time_unit` on the HH side, except that the unit is fixed at
    ``u.ms`` rather than declared per transition -- the catalogue's Markov
    channels all come from NEURON ``.mod`` sources written in milliseconds.
    """

    pairs: ClassVar[tuple[Transition | tuple[Any, ...], ...]] = ()
    conserve: ClassVar[Any] = 1.0
    dependent_state: ClassVar[str | None] = None
    default_solver: ClassVar[str] = "backward_euler"
    default_substeps: ClassVar[int] = 1
    #: Project independent states into ``[0, 1]`` before evaluating the
    #: transition graph. On by default: a probability pool that leaves the
    #: simplex makes the kinetics meaningless, whereas an HH gate merely
    #: becomes inaccurate. See :attr:`Gate.clip` for the HH-side switch.
    clip_states: ClassVar[bool] = True
    _resolved_pairs: ClassVar[tuple[Transition, ...]] = ()
    _resolved_state_names: ClassVar[tuple[str, ...]] = ()
    #: See :attr:`HH._rate_ion_counts` -- same lazy per-class cache.
    _rate_ion_counts: ClassVar[dict[str, int | None]]

    def __init_subclass__(cls, **kwargs):
        """Resolve and validate ``pairs`` once, when the subclass is created.

        Classes declaring no transitions are skipped so abstract
        intermediates stay definable.
        """
        super().__init_subclass__(**kwargs)

        resolved = _resolve_declarations(cls.pairs, Transition)
        cls._resolved_pairs = resolved
        if not resolved:
            cls._resolved_state_names = ()
            return

        names: list[str] = []
        seen: set[str] = set()
        for pair in resolved:
            for name in (pair.src, pair.dst):
                if name not in seen:
                    _check_identifier(cls, name, "state")
                    names.append(name)
                    seen.add(name)
        if len(names) < 2:
            raise ValueError(f"{cls.__name__}: Markov requires at least two states, got {names}.")
        cls._resolved_state_names = tuple(names)

        declared = cls.dependent_state
        if declared is not None and declared not in seen:
            raise ValueError(
                f"{cls.__name__}: dependent_state {declared!r} is not one of the declared states {sorted(seen)}."
            )

        for pair in resolved:
            for rate in (pair.forward, pair.backward):
                if rate is not None and not hasattr(cls, rate):
                    raise ValueError(
                        f"{cls.__name__}: transition {pair.src!r} -> {pair.dst!r} references "
                        f"rate method {rate!r}, which is not defined."
                    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
        solver: str | None = None,
        substeps: int | None = None,
    ):
        super().__init__(size=size, name=name)
        if solver is None:
            solver = type(self).default_solver
        if substeps is None:
            substeps = type(self).default_substeps
        IndependentIntegration.__init__(self, solver=solver)

        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1.")

    def make_integration(self, *args, **kwargs):
        with brainstate.environ.context(dt=brainstate.environ.get_dt() / self.substeps):
            brainstate.transform.for_loop(
                lambda i: self.solver(self, *args, **kwargs),
                u.math.arange(self.substeps),
            )

    def _iter_pairs(self) -> tuple[Transition, ...]:
        return type(self)._resolved_pairs

    def _state_names(self) -> tuple[str, ...]:
        return type(self)._resolved_state_names

    def _dependent_state_name(self) -> str:
        state_names = self._state_names()
        if len(state_names) < 2:
            raise ValueError("Markov requires at least two states.")
        declared = type(self).dependent_state
        if declared is not None:
            return declared
        warnings.warn(
            f"{type(self).__name__} does not declare `dependent_state`, so it falls back to "
            f"{state_names[-1]!r} -- the last state discovered while scanning `pairs`. Reordering "
            f"`pairs` would silently eliminate a different state and change the conditioning of "
            f"the steady-state solve. Set `dependent_state` explicitly; the implicit fallback "
            f"will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return state_names[-1]

    def _independent_state_names(self) -> tuple[str, ...]:
        dependent = self._dependent_state_name()
        return tuple(name for name in self._state_names() if name != dependent)

    def _state_zero(self):
        independent = self._independent_state_names()
        if not independent:
            raise ValueError("Markov requires at least one independent state.")
        return u.math.zeros_like(getattr(self, independent[0]).value)

    def _conserve_value(self):
        return _resolve_value(self, type(self).conserve)

    def _independent_state_values(self):
        return {name: getattr(self, name).value for name in self._independent_state_names()}

    def _dependent_state_value(self, states):
        total = None
        for value in states.values():
            total = value if total is None else (total + value)
        if total is None:
            total = 0.0
        return self._conserve_value() - total

    def _project_independent_state(self, name: str, value):
        _ = name
        if not type(self).clip_states:
            return value
        return u.math.clip(value, 0.0, 1.0)

    def _kinetic_state_values(self):
        raw_states = self._independent_state_values()
        states = {name: self._project_independent_state(name, value) for name, value in raw_states.items()}
        states[self._dependent_state_name()] = self._dependent_state_value(raw_states)
        return states

    def _call_rate(self, rate_name: str, V, *ions):
        return _call_rate(self, rate_name, V, *ions)

    def _transition_rate(self, rate_name: str, V, *ions):
        """Call one transition rate and normalise it to a bare per-ms value.

        Every consumer of a transition rate goes through here -- the
        derivative and both steady-state solvers -- so the dimension check
        cannot be bypassed by adding a fourth.
        """
        return _as_markov_rate(self._call_rate(rate_name, V, *ions), type(self), rate_name)

    def pre_integral(self, V, *ions):
        _ = (V, ions)

    def post_integral(self, V, *ions):
        _ = (V, ions)

    @property
    def state_names(self) -> tuple[str, ...]:
        return self._independent_state_names()

    @property
    def redundant_state(self) -> str:
        return self._dependent_state_name()

    @property
    def state_pairs(self) -> tuple[tuple[str, str, str, str | None], ...]:
        return tuple((pair.src, pair.dst, pair.forward, pair.backward) for pair in self._iter_pairs())

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for name in self._independent_state_names():
            _bind_state(
                self,
                name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
                "Markov state",
            )

    def reset_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for name in self._independent_state_names():
            value = braintools.init.param(u.math.zeros, self.varshape, batch_size)
            getattr(self, name).value = value
            if isinstance(batch_size, int):
                assert value.shape[0] == batch_size

    def _solve_steady_state(self, V, *ions):
        """Return Markov steady-state probabilities for reset-time initialization.

        The host implementation avoids compiling a collection of small XLA
        kernels during cold reset. Traced contexts keep the JAX implementation
        so symbolic execution remains available where JAX requires it.
        """
        try:
            return self._solve_steady_state_host(V, *ions)
        except (
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError,
        ):
            return self._solve_steady_state_jax(V, *ions)

    def _solve_steady_state_jax(self, V, *ions):
        """Return Markov steady-state probabilities using JAX arrays."""
        state_names = self._state_names()
        dependent_index = state_names.index(self._dependent_state_name())
        template = jnp.asarray(u.get_magnitude(self._state_zero()))
        template_shape = template.shape
        flat_size = int(template.size)

        def _flatten_like(value, label: str):
            array = jnp.asarray(u.get_magnitude(value))
            if array.shape != template_shape:
                if array.size == 1:
                    array = jnp.full(template_shape, array.reshape(()), dtype=array.dtype)
                else:
                    try:
                        array = jnp.broadcast_to(array, template_shape)
                    except ValueError as err:
                        raise ValueError(
                            f"{type(self).__name__}.{label} could not be broadcast "
                            f"to steady-state shape {template_shape}."
                        ) from err
            return array.reshape(flat_size)

        conserve = _flatten_like(self._conserve_value(), "conserve")
        pair_rates = []
        rates = []
        # Resolve each rate once; steady-state assembly reuses the arrays below.
        for pair in self._iter_pairs():
            forward = _flatten_like(self._transition_rate(pair.forward, V, *ions), pair.forward)
            backward = None
            if pair.backward is not None:
                backward = _flatten_like(self._transition_rate(pair.backward, V, *ions), pair.backward)
            pair_rates.append((pair, forward, backward))
            rates.append(forward)
            if backward is not None:
                rates.append(backward)

        dtype = jnp.result_type(template, conserve, *rates) if rates else jnp.result_type(template, conserve)
        conserve = conserve.astype(dtype)
        generator = jnp.zeros((flat_size, len(state_names), len(state_names)), dtype=dtype)

        for pair, forward, backward in pair_rates:
            src = state_names.index(pair.src)
            dst = state_names.index(pair.dst)
            forward = forward.astype(dtype)
            generator = generator.at[:, src, src].add(-forward)
            generator = generator.at[:, dst, src].add(forward)
            if backward is not None:
                backward = backward.astype(dtype)
                generator = generator.at[:, src, dst].add(backward)
                generator = generator.at[:, dst, dst].add(-backward)

        lhs = generator.at[:, dependent_index, :].set(jnp.ones((flat_size, len(state_names)), dtype=dtype))
        rhs = jnp.zeros((flat_size, len(state_names)), dtype=dtype).at[:, dependent_index].set(conserve)
        try:
            solution = jnp.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        except Exception as err:
            raise ValueError(f"{type(self).__name__} steady-state linear system could not be solved.") from err

        traced = is_traced_value(solution)
        if not traced:
            if not bool(jnp.all(jnp.isfinite(solution))):
                raise ValueError(f"{type(self).__name__} steady-state solve returned non-finite values.")

            tol = 1e-7
            if bool(jnp.any(solution < -tol)) or bool(jnp.any(solution > conserve[:, None] + tol)):
                raise ValueError(f"{type(self).__name__} steady-state solve returned out-of-range probabilities.")

        solution = jnp.clip(solution, 0.0, None)
        totals = solution.sum(axis=1, keepdims=True)
        if not traced and not bool(jnp.all(totals > 0.0)):
            raise ValueError(f"{type(self).__name__} steady-state solve collapsed to zero probability mass.")
        solution = solution * (conserve[:, None] / totals)

        return {name: solution[:, index].reshape(template_shape) for index, name in enumerate(state_names)}

    def _solve_steady_state_host(self, V, *ions):
        """Return Markov steady-state probabilities using a host NumPy solve."""
        state_names = self._state_names()
        dependent_index = state_names.index(self._dependent_state_name())
        template = jnp.asarray(u.get_magnitude(self._state_zero()))
        template_shape = template.shape
        flat_size = int(template.size)
        template_host = np.asarray(jax.device_get(template))

        def _flatten_like(value, label: str):
            array = np.asarray(jax.device_get(u.get_magnitude(value)))
            if array.shape != template_shape:
                if array.size == 1:
                    array = np.full(template_shape, array.reshape(()), dtype=array.dtype)
                else:
                    try:
                        array = np.broadcast_to(array, template_shape)
                    except ValueError as err:
                        raise ValueError(
                            f"{type(self).__name__}.{label} could not be broadcast "
                            f"to steady-state shape {template_shape}."
                        ) from err
            return array.reshape(flat_size)

        conserve = _flatten_like(self._conserve_value(), "conserve")
        pair_rates = []
        rates = []
        # Resolve each rate once; steady-state assembly reuses the arrays below.
        for pair in self._iter_pairs():
            forward = _flatten_like(self._transition_rate(pair.forward, V, *ions), pair.forward)
            backward = None
            if pair.backward is not None:
                backward = _flatten_like(self._transition_rate(pair.backward, V, *ions), pair.backward)
            pair_rates.append((pair, forward, backward))
            rates.append(forward)
            if backward is not None:
                rates.append(backward)

        if rates:
            dtype = np.dtype(jnp.result_type(template_host, conserve, *rates))
        else:
            dtype = np.dtype(jnp.result_type(template_host, conserve))
        conserve = conserve.astype(dtype, copy=False)
        generator = np.zeros((flat_size, len(state_names), len(state_names)), dtype=dtype)

        for pair, forward, backward in pair_rates:
            src = state_names.index(pair.src)
            dst = state_names.index(pair.dst)
            forward = forward.astype(dtype, copy=False)
            generator[:, src, src] -= forward
            generator[:, dst, src] += forward
            if backward is not None:
                backward = backward.astype(dtype, copy=False)
                generator[:, src, dst] += backward
                generator[:, dst, dst] -= backward

        lhs = generator.copy()
        lhs[:, dependent_index, :] = 1
        rhs = np.zeros((flat_size, len(state_names)), dtype=dtype)
        rhs[:, dependent_index] = conserve
        try:
            solution = np.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        except Exception as err:
            raise ValueError(f"{type(self).__name__} steady-state linear system could not be solved.") from err

        if not np.all(np.isfinite(solution)):
            raise ValueError(f"{type(self).__name__} steady-state solve returned non-finite values.")

        tol = 1e-7
        if np.any(solution < -tol) or np.any(solution > conserve[:, None] + tol):
            raise ValueError(f"{type(self).__name__} steady-state solve returned out-of-range probabilities.")

        solution = np.clip(solution, 0.0, None)
        totals = solution.sum(axis=1, keepdims=True)
        if not np.all(totals > 0.0):
            raise ValueError(f"{type(self).__name__} steady-state solve collapsed to zero probability mass.")
        solution = solution * (conserve[:, None] / totals)

        return {name: jnp.asarray(solution[:, index].reshape(template_shape)) for index, name in enumerate(state_names)}

    def reset_steady_state(self, V, *ions, batch_size: int = None):
        states = self._solve_steady_state(V, *ions)
        for name in self._independent_state_names():
            value = states[name]
            getattr(self, name).value = value
            if isinstance(batch_size, int):
                assert value.shape[0] == batch_size

    def state_values(self):
        states = self._independent_state_values()
        states[self._dependent_state_name()] = self._dependent_state_value(states)
        return states

    def compute_derivative(self, V, *ions):
        states = self._kinetic_state_values()
        derivatives = {name: self._state_zero() for name in states}

        for pair in self._iter_pairs():
            forward = self._transition_rate(pair.forward, V, *ions)
            derivatives[pair.src] = derivatives[pair.src] - states[pair.src] * forward
            derivatives[pair.dst] = derivatives[pair.dst] + states[pair.src] * forward

            if pair.backward is not None:
                backward = self._transition_rate(pair.backward, V, *ions)
                derivatives[pair.src] = derivatives[pair.src] + states[pair.dst] * backward
                derivatives[pair.dst] = derivatives[pair.dst] - states[pair.dst] * backward

        for name in self._independent_state_names():
            # The rates are normalised to bare per-ms values, so the single
            # division below is what gives the derivative its unit. `conserve`
            # still reaches the states unchecked, hence the assertion.
            derivative = derivatives[name] / u.ms
            getattr(self, name).derivative = _check_derivative(
                derivative, f"{type(self).__name__} state {name!r}"
            )
