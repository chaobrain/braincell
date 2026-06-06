# -*- coding: utf-8 -*-


from dataclasses import dataclass
from functools import lru_cache
import inspect
from typing import Any, Optional
from typing import ClassVar

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import Channel
from braincell.quad.protocol import DiffEqState
from braincell.quad.protocol import IndependentIntegration

__all__ = [
    "Gate",
    "Transition",
    "HH",
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
    return value(owner) if callable(value) else value


@lru_cache(maxsize=None)
def _rate_ion_count(owner_type: type, rate_name: str) -> int | None:
    signature = inspect.signature(getattr(owner_type, rate_name))
    params = tuple(signature.parameters.values())
    positional = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return None

    positional_count = sum(1 for param in params if param.kind in positional)
    return max(0, positional_count - 2)


@dataclass(frozen=True)
class Gate:
    """Metadata for one HH gate."""

    name: str
    power: int = 1
    phi: Any | None = None
    q10: Any | None = None
    temp_ref: Any | None = None

    def __post_init__(self):
        has_phi = self.phi is not None
        has_q10 = self.q10 is not None
        has_temp_ref = self.temp_ref is not None

        if has_phi and (has_q10 or has_temp_ref):
            raise ValueError(
                f"Gate {self.name!r}: phi cannot be provided together with q10/temp_ref."
            )
        if has_q10 != has_temp_ref:
            raise ValueError(
                f"Gate {self.name!r}: q10 and temp_ref must be provided together."
            )


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

    def _iter_gates(self) -> tuple[Gate, ...]:
        items = []
        for gate in type(self).gates:
            if isinstance(gate, Gate):
                items.append(gate)
            else:
                items.append(Gate(*gate))
        return tuple(items)

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

    def _has_inf_tau(self, gate: Gate) -> bool:
        return hasattr(self, f"f_{gate.name}_inf") and hasattr(self, f"f_{gate.name}_tau")

    def _has_alpha_beta(self, gate: Gate) -> bool:
        return hasattr(self, f"f_{gate.name}_alpha") and hasattr(self, f"f_{gate.name}_beta")

    def _gate_form(self, gate: Gate) -> str:
        has_inf_tau = self._has_inf_tau(gate)
        has_alpha_beta = self._has_alpha_beta(gate)
        if has_inf_tau and has_alpha_beta:
            raise ValueError(
                f"Gate {gate.name!r} defines both inf/tau and alpha/beta forms; choose one."
            )
        if has_inf_tau:
            return "inf_tau"
        if has_alpha_beta:
            return "alpha_beta"
        raise ValueError(
            f"Gate {gate.name!r} must define either inf/tau or alpha/beta methods."
        )

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for gate in self._iter_gates():
            setattr(
                self,
                gate.name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
            )

    def conductance_factor(self, V, *ions):
        _ = (V, ions)
        product = 1.0
        for gate in self._iter_gates():
            value = self._gate_value(gate)
            product = product * (value if gate.power == 1 else value ** gate.power)
        return product

    def reset_state(self, V, *ions, batch_size: int = None):
        for gate in self._iter_gates():
            form = self._gate_form(gate)
            if form == "inf_tau":
                value = getattr(self, f"f_{gate.name}_inf")(V, *ions)
            else:
                alpha = getattr(self, f"f_{gate.name}_alpha")(V, *ions)
                beta = getattr(self, f"f_{gate.name}_beta")(V, *ions)
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
                inf = getattr(self, f"f_{gate.name}_inf")(V, *ions)
                tau = getattr(self, f"f_{gate.name}_tau")(V, *ions)
                derivative = phi * (inf - value) / tau / u.ms
            else:
                alpha = getattr(self, f"f_{gate.name}_alpha")(V, *ions)
                beta = getattr(self, f"f_{gate.name}_beta")(V, *ions)
                derivative = phi * (alpha * (1.0 - value) - beta * value) / u.ms
            self._gate_state(gate).derivative = derivative


class Markov(Channel, IndependentIntegration):
    """Probability-state channel kinetics described by transition pairs.

    ``pairs`` define one conserved probability pool. By default the dependent
    state is the last state whose name is first discovered while scanning
    ``pairs``. Override ``dependent_state`` when that order-based default is
    not the intended hidden state.

    ``state_values()`` returns the raw stored states plus the reconstructed
    dependent state. ``compute_derivative()`` uses ``_kinetic_state_values()``,
    which clips each independent state to ``[0, 1]`` before evaluating the
    transition graph while still reconstructing the dependent state from the
    raw stored sum.
    """

    pairs: ClassVar[tuple[Transition | tuple[Any, ...], ...]] = ()
    conserve: ClassVar[Any] = 1.0
    dependent_state: ClassVar[str | None] = None

    def __init__(
        self,
        size: brainstate.typing.Size,
        name: Optional[str] = None,
        solver: str = "barkward_euler", #"barkward_euler"
        substeps: int = 1,
    ):
        super().__init__(size=size, name=name)
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
        items = []
        for pair in type(self).pairs:
            if isinstance(pair, Transition):
                items.append(pair)
            else:
                items.append(Transition(*pair))
        return tuple(items)

    def _state_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for pair in self._iter_pairs():
            for name in (pair.src, pair.dst):
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        return tuple(names)

    def _dependent_state_name(self) -> str:
        state_names = self._state_names()
        if len(state_names) < 2:
            raise ValueError("Markov requires at least two states.")
        if type(self).dependent_state is not None:
            if type(self).dependent_state not in state_names:
                raise ValueError(
                    f"dependent_state {type(self).dependent_state!r} is not present in Markov states."
                )
            return type(self).dependent_state
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
        return {
            name: getattr(self, name).value
            for name in self._independent_state_names()
        }

    def _dependent_state_value(self, states):
        total = None
        for value in states.values():
            total = value if total is None else (total + value)
        if total is None:
            total = 0.0
        return self._conserve_value() - total

    def _project_independent_state(self, name: str, value):
        _ = name
        return u.math.clip(value, 0.0, 1.0)

    def _kinetic_state_values(self):
        raw_states = self._independent_state_values()
        states = {
            name: self._project_independent_state(name, value)
            for name, value in raw_states.items()
        }
        states[self._dependent_state_name()] = self._dependent_state_value(raw_states)
        return states

    def _call_rate(self, rate_name: str, V, *ions):
        ion_count = _rate_ion_count(type(self), rate_name)
        if ion_count is None:
            return getattr(self, rate_name)(V, *ions)
        if ion_count > len(ions):
            raise TypeError(
                f"{type(self).__name__}.{rate_name} expects {ion_count} ion argument(s), "
                f"got {len(ions)}."
            )
        return getattr(self, rate_name)(V, *ions[:ion_count])

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
        return tuple(
            (pair.src, pair.dst, pair.forward, pair.backward)
            for pair in self._iter_pairs()
        )

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for name in self._independent_state_names():
            setattr(
                self,
                name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
            )

    def reset_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for name in self._independent_state_names():
            value = braintools.init.param(u.math.zeros, self.varshape, batch_size)
            getattr(self, name).value = value
            if isinstance(batch_size, int):
                assert value.shape[0] == batch_size

    def _solve_steady_state(self, V, *ions):
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
                            f"{type(self).__name__}.{label} could not be broadcast to steady-state shape {template_shape}."
                        ) from err
            return array.reshape(flat_size)

        conserve = _flatten_like(self._conserve_value(), "conserve")
        rates = []
        for pair in self._iter_pairs():
            rates.append(_flatten_like(self._call_rate(pair.forward, V, *ions), pair.forward))
            if pair.backward is not None:
                rates.append(_flatten_like(self._call_rate(pair.backward, V, *ions), pair.backward))

        dtype = jnp.result_type(template, conserve, *rates) if rates else jnp.result_type(template, conserve)
        conserve = conserve.astype(dtype)
        generator = jnp.zeros((flat_size, len(state_names), len(state_names)), dtype=dtype)

        for pair in self._iter_pairs():
            src = state_names.index(pair.src)
            dst = state_names.index(pair.dst)
            forward = _flatten_like(self._call_rate(pair.forward, V, *ions), pair.forward).astype(dtype)
            generator = generator.at[:, src, src].add(-forward)
            generator = generator.at[:, dst, src].add(forward)
            if pair.backward is not None:
                backward = _flatten_like(self._call_rate(pair.backward, V, *ions), pair.backward).astype(dtype)
                generator = generator.at[:, src, dst].add(backward)
                generator = generator.at[:, dst, dst].add(-backward)

        lhs = generator.at[:, dependent_index, :].set(jnp.ones((flat_size, len(state_names)), dtype=dtype))
        rhs = jnp.zeros((flat_size, len(state_names)), dtype=dtype).at[:, dependent_index].set(conserve)
        try:
            solution = jnp.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        except Exception as err:
            raise ValueError(f"{type(self).__name__} steady-state linear system could not be solved.") from err

        if not bool(jnp.all(jnp.isfinite(solution))):
            raise ValueError(f"{type(self).__name__} steady-state solve returned non-finite values.")

        tol = 1e-7
        if bool(jnp.any(solution < -tol)) or bool(jnp.any(solution > conserve[:, None] + tol)):
            raise ValueError(f"{type(self).__name__} steady-state solve returned out-of-range probabilities.")

        solution = jnp.clip(solution, 0.0, None)
        totals = solution.sum(axis=1, keepdims=True)
        if not bool(jnp.all(totals > 0.0)):
            raise ValueError(f"{type(self).__name__} steady-state solve collapsed to zero probability mass.")
        solution = solution * (conserve[:, None] / totals)

        return {
            name: solution[:, index].reshape(template_shape)
            for index, name in enumerate(state_names)
        }

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
            forward = self._call_rate(pair.forward, V, *ions)
            derivatives[pair.src] = derivatives[pair.src] - states[pair.src] * forward
            derivatives[pair.dst] = derivatives[pair.dst] + states[pair.src] * forward

            if pair.backward is not None:
                backward = self._call_rate(pair.backward, V, *ions)
                derivatives[pair.src] = derivatives[pair.src] + states[pair.dst] * backward
                derivatives[pair.dst] = derivatives[pair.dst] - states[pair.dst] * backward

        for name in self._independent_state_names():
            getattr(self, name).derivative = derivatives[name] / u.ms
