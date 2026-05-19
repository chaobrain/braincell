# -*- coding: utf-8 -*-

"""Shared ion-side templates.

This module contains mixins used by concrete ion classes such as
``SodiumFixed``, ``CalciumDetailed``, or kinetic-ion subclasses built from
declarative reaction-network pieces. The public lifecycle still lives on
``Ion``; these mixins only provide helper methods and lifecycle hooks for
common ion patterns:

- fixed ``Ci/Co/E``
- fixed ``Ci/Co`` with ``E`` initialized from Nernst
- dynamic ``Ci`` with Nernst-computed ``E``
- kinetic ion species with algebraic conservation constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import brainstate
import braintools
import brainunit as u

from braincell.quad import get_integrator
from braincell.quad.protocol import DiffEqState, IndependentIntegration

__all__ = [
    "Factor",
    "Species",
    "Reaction",
    "Source",
    "Conserve",
    "FixedIon",
    "InitNernstIon",
    "DynamicNernstIon",
    "KineticIon",
]


@dataclass(frozen=True)
class Factor:
    """Named constant factor for visible-to-amount conversion.

    Parameters
    ----------
    name : str
        Factor identifier referenced by :class:`Species`.
    value : callable
        Callable ``value(owner)`` returning the factor for the concrete ion
        instance. This factor is treated as constant during one integration
        step.
    """

    name: str
    value: Callable[[Any], Any]


@dataclass(frozen=True)
class Species:
    """Declare one reaction-network species.

    Parameters
    ----------
    name : str
        Species name. ``"Ci"`` is reserved and must be present exactly once
        for :class:`KineticIon`.
    init : Any
        Visible-space initializer and unit source for this species.
    factor : str or None, optional
        Optional :class:`Factor` name used for visible/amount conversion.
        ``None`` denotes identity conversion.
    """

    name: str
    init: Any
    factor: str | None = None


@dataclass(frozen=True)
class Reaction:
    """Declare one mass-action reaction.

    Parameters
    ----------
    lhs : dict[str, int]
        Left-hand stoichiometry. Keys are species names and values are positive
        integers.
    rhs : dict[str, int]
        Right-hand stoichiometry. Keys are species names and values are
        positive integers.
    forward : callable
        Callable ``forward(owner, V, species_values)`` returning the forward
        reaction coefficient as a quantity or scalar. Runtime multiplies it by
        the left-hand visible species product directly, preserving quantity
        units.
    backward : callable or None, optional
        Optional callable for the reverse direction. ``None`` denotes a
        single-direction reaction. As with ``forward``, its returned value is
        multiplied directly by the right-hand visible species product.
    """

    lhs: dict[str, int]
    rhs: dict[str, int]
    forward: Callable[[Any, Any, dict[str, Any]], Any]
    backward: Callable[[Any, Any, dict[str, Any]], Any] | None = None


@dataclass(frozen=True)
class Source:
    """Declare one source term.

    Parameters
    ----------
    target : str
        Target diffeq species.
    flux : callable
        Callable ``flux(owner, V, species_values)`` returning a contribution to
        the factor-scaled derivative of ``target``. When ``target`` has no
        factor this reduces to the ordinary visible derivative.
    """

    target: str
    flux: Callable[[Any, Any, dict[str, Any]], Any]


@dataclass(frozen=True)
class Conserve:
    """Declare one algebraic conservation relation.

    Parameters
    ----------
    species : tuple[str, ...]
        Species participating in the conserved pool.
    algebraic : str
        The single algebraic species solved from the conservation law.
    total : callable
        Callable ``total(owner, V, species_values)`` returning the conserved
        pool size in factor-scaled units.
    """

    species: tuple[str, ...]
    algebraic: str
    total: Callable[[Any, Any, dict[str, Any]], Any]


class FixedIon(brainstate.mixin.Mixin):
    """Helper mixin for ions with fixed ``Ci/Co/E`` state."""

    def _init_fixed_ion(self, *, Ci=None, Co=None, E=None, valence=None):
        """Materialize one fixed ion payload onto ``self``.

        Parameters
        ----------
        Ci : Any, optional
            Intracellular concentration override. Defaults to the species-level
            ``default_Ci``.
        Co : Any, optional
            Extracellular concentration override. Defaults to the species-level
            ``default_Co``.
        E : Any
            Fixed reversal potential.
        valence : Any, optional
            Ionic valence override. Defaults to the species-level
            ``default_valence``.
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


class InitNernstIon(brainstate.mixin.Mixin):
    """Helper mixin for ions with fixed ``Ci/Co`` and stored Nernst ``E``."""

    def _init_nernst_ion(self, *, Ci=None, Co=None, temp=None, valence=None):
        """Initialize fixed concentrations and stored-Nernst parameters.

        Parameters
        ----------
        Ci : Any, optional
            Intracellular concentration override.
        Co : Any, optional
            Extracellular concentration override.
        temp : Any
            Absolute temperature used by the Nernst equation.
        valence : Any, optional
            Ionic valence override.
        """
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
        self.E = (u.gas_constant * temp / (valence * u.faraday_constant)) * u.math.log(Co / Ci)

    def _ion_init_state_hook(self, V, batch_size: int = None):
        """Refresh the stored Nernst reversal during ion initialization."""
        _ = (V, batch_size)
        self._update_reversal()

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        """Refresh the stored Nernst reversal during ion reset."""
        _ = (V, batch_size)
        self._update_reversal()


class DynamicNernstIon(brainstate.mixin.Mixin):
    """Helper mixin for ions with dynamic ``Ci`` and computed Nernst ``E``."""

    #: When true, the template precomputes the aggregate ion current and passes
    #: it to ``derivative(..., total_current=...)``.
    uses_total_current = False

    def _init_dynamic_nernst_ion(self, *, Co=None, temp=None, valence=None, Ci_initializer=None):
        """Initialize the static fields and remember the ``Ci`` initializer.

        Parameters
        ----------
        Co : Any, optional
            Extracellular concentration override.
        temp : Any
            Absolute temperature used by the Nernst equation.
        valence : Any, optional
            Ionic valence override.
        Ci_initializer : Any, optional
            Initializer for the dynamic ``Ci`` state.
        """
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
            # In the family-phased update path, current-driven ion dynamics
            # consume a step-start cached current snapshot when available.
            # This avoids re-evaluating a newer current after channel or
            # voltage states have already advanced later in the step.
            if hasattr(self, "_cached_total_current"):
                total_current = self._cached_total_current
            else:
                total_current = self.current(V, include_external=True)
        self.Ci.derivative = self.derivative(
            self.Ci.value,
            V,
            total_current=total_current,
        )

    def derivative(self, Ci, V, total_current=None):
        """Return ``dCi/dt`` for the concrete dynamic ion model."""
        raise NotImplementedError


class KineticIon(IndependentIntegration):
    """Template for NMODL-style kinetic ion species.

    Subclasses declare a reaction-network species table, optional conversion
    factors, explicit reaction/source callbacks, and algebraic conservation
    relations. The reserved species name ``"Ci"`` supplies the ion protocol's
    intracellular concentration; ``Co``, ``temp``, and ``valence`` remain
    ion-level fields.

    Notes
    -----
    Species live in visible units during integration. ``factor`` only mediates
    temporary visible/scaled conversion inside conservation and derivative
    mapping; species values are not stored in scaled form. Reaction laws remain
    in the visible domain, matching NEURON's ``KINETIC``/``COMPARTMENT``
    behavior.
    """

    factors: ClassVar[tuple[Factor, ...]] = ()
    species: ClassVar[tuple[Species, ...]] = ()
    reactions: ClassVar[tuple[Reaction, ...]] = ()
    sources: ClassVar[tuple[Source, ...]] = ()
    conserves: ClassVar[tuple[Conserve, ...]] = ()
    uses_total_current: ClassVar[bool] = False

    def _init_kinetic_ion(
        self,
        *,
        Co=None,
        temp=None,
        valence=None,
        species_initializers: dict[str, Any] | None = None,
        solver: str = "rk4",
        substeps: int = 5,
    ):
        """Initialize one declarative kinetic-ion instance.

        Parameters
        ----------
        Co : Any, optional
            Extracellular concentration override.
        temp : Any
            Absolute temperature used by the Nernst equation.
        valence : Any, optional
            Ionic valence override.
        solver : str, optional
            Solver name used when this ion is independently integrated.
        substeps : int, optional
            Number of substeps run inside one parent update.
        """
        if temp is None:
            raise ValueError(f"{type(self).__name__} requires an explicit temperature value.")

        IndependentIntegration.__init__(self, solver=solver)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1.")

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
        self.species_initializers = dict(species_initializers or {})

    @property
    def E(self):
        """Nernst reversal potential from the current ``Ci``."""
        Co = self.Co.value if isinstance(self.Co, brainstate.State) else self.Co
        temp = self.temp.value if isinstance(self.temp, brainstate.State) else self.temp
        valence = self.valence.value if isinstance(self.valence, brainstate.State) else self.valence
        return (u.gas_constant * temp / (valence * u.faraday_constant)) * u.math.log(Co / self.Ci.value)

    def make_integration(self, V):
        """Advance this ion with its own solver and substep schedule."""
        with brainstate.environ.context(dt=brainstate.environ.get_dt() / self.substeps):
            brainstate.transform.for_loop(
                lambda i: self.solver(self, V),
                u.math.arange(self.substeps),
            )

    def species_values(self):
        """Return the current full visible species view."""
        specs = _Specs.for_type(type(self))
        return _Conserve(self, specs, _Species(self, specs)).resolve()

    def _ion_init_state_hook(self, V, batch_size: int = None):
        """Initialize runtime species and project algebraic species."""
        specs = _Specs.for_type(type(self))
        species = _Species(self, specs)
        species.init(batch_size=batch_size)
        _Conserve(self, specs, species).writeback(V)

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        """Reset runtime species and project algebraic species."""
        specs = _Specs.for_type(type(self))
        species = _Species(self, specs)
        species.reset(batch_size=batch_size)
        _Conserve(self, specs, species).writeback(V)

    def _ion_compute_derivative_hook(self, V):
        """Resolve algebraic species and write diffeq derivatives."""
        specs = _Specs.for_type(type(self))
        species = _Species(self, specs)
        conserve = _Conserve(self, specs, species)
        total_current = None
        if type(self).uses_total_current:
            # Reuse a cached total current when a caller has precomputed one;
            # otherwise fall back to the ion's current evaluation path.
            if hasattr(self, "_cached_total_current"):
                total_current = self._cached_total_current
            else:
                total_current = self.current(V, include_external=True)
        _Flux(self, specs, species).compute(V, conserve.resolve(V), total_current=total_current)

    def _ion_post_integral_hook(self, V):
        """Refresh cached algebraic species after one integration step."""
        specs = _Specs.for_type(type(self))
        species = _Species(self, specs)
        _Conserve(self, specs, species).writeback(V)


class _Specs:
    """Validated declarative specifications for one ``KineticIon`` subtype."""

    _cache: ClassVar[dict[type, "_Specs"]] = {}

    @classmethod
    def for_type(cls, ion_type: type) -> "_Specs":
        """Return the cached validated specs for ``ion_type``."""
        cached = cls._cache.get(ion_type)
        if cached is None:
            cached = cls(
                factors=tuple(type_factor if isinstance(type_factor, Factor) else Factor(*type_factor)
                              for type_factor in getattr(ion_type, "factors", ())),
                species=tuple(type_species if isinstance(type_species, Species) else Species(*type_species)
                              for type_species in getattr(ion_type, "species", ())),
                reactions=tuple(type_reaction if isinstance(type_reaction, Reaction) else Reaction(*type_reaction)
                                for type_reaction in getattr(ion_type, "reactions", ())),
                sources=tuple(type_source if isinstance(type_source, Source) else Source(*type_source)
                              for type_source in getattr(ion_type, "sources", ())),
                conserves=tuple(type_conserve if isinstance(type_conserve, Conserve) else Conserve(*type_conserve)
                                for type_conserve in getattr(ion_type, "conserves", ())),
            )
            cls._cache[ion_type] = cached
        return cached

    def __init__(self, *, factors, species, reactions, sources, conserves):
        self.factors_by_name = {factor.name: factor for factor in factors}
        self.species_by_name = {spec.name: spec for spec in species}
        self.reactions = tuple(reactions)
        self.sources = tuple(sources)
        self.conserves = tuple(conserves)
        self._validate(factors=factors, species=species)

        algebraic_names = tuple(conserve.algebraic for conserve in self.conserves)
        self.algebraic_names = algebraic_names
        self.algebraic_set = set(algebraic_names)
        self.diffeq_names = tuple(
            spec.name for spec in species
            if spec.name not in self.algebraic_set
        )
        self.diffeq_set = set(self.diffeq_names)

    def _validate(self, *, factors, species):
        if len(factors) != len(self.factors_by_name):
            raise ValueError("KineticIon factor names must be unique.")
        if len(species) != len(self.species_by_name):
            raise ValueError("KineticIon species names must be unique.")
        if "Ci" not in self.species_by_name:
            raise ValueError("KineticIon requires a species named 'Ci'.")

        for spec in species:
            if spec.factor is not None and spec.factor not in self.factors_by_name:
                raise ValueError(
                    f"KineticIon species {spec.name!r} references unknown factor {spec.factor!r}."
                )

        algebraic_names = []
        for conserve in self.conserves:
            if len(conserve.species) < 2:
                raise ValueError("Each Conserve requires at least two species.")
            if conserve.algebraic not in conserve.species:
                raise ValueError(
                    f"Conserve algebraic species {conserve.algebraic!r} must be present in conserve.species."
                )
            for name in conserve.species:
                if name not in self.species_by_name:
                    raise ValueError(f"Conserve references unknown species {name!r}.")
            algebraic_names.append(conserve.algebraic)
        if len(algebraic_names) != len(set(algebraic_names)):
            raise ValueError("An algebraic species may only appear in one Conserve declaration.")
        if "Ci" in algebraic_names:
            raise ValueError("The reserved species 'Ci' must remain a diffeq species.")

        for reaction in self.reactions:
            if not reaction.lhs and not reaction.rhs:
                raise ValueError("Reaction requires at least one left- or right-hand species.")
            for side_name, stoich in tuple(reaction.lhs.items()) + tuple(reaction.rhs.items()):
                if side_name not in self.species_by_name:
                    raise ValueError(f"Reaction references unknown species {side_name!r}.")
                if not isinstance(stoich, int) or stoich <= 0:
                    raise ValueError("Reaction stoichiometries must be positive integers.")

        for source in self.sources:
            if source.target not in self.species_by_name:
                raise ValueError(f"Source references unknown species {source.target!r}.")
            if source.target in algebraic_names:
                raise ValueError("Source target must be a diffeq species, not an algebraic species.")


class _Species:
    """Runtime adapter for diffeq and algebraic species values."""

    def __init__(self, owner, specs: _Specs):
        self.owner = owner
        self.specs = specs

    def init(self, batch_size: int = None):
        """Materialize runtime species attributes from class declarations."""
        for spec in self.specs.species_by_name.values():
            init = self.owner.species_initializers.get(spec.name, spec.init)
            value = braintools.init.param(init, self.owner.varshape, batch_size)
            if spec.name in self.specs.diffeq_set:
                setattr(self.owner, spec.name, DiffEqState(value))
            else:
                setattr(self.owner, spec.name, brainstate.HiddenState(value))

    def reset(self, batch_size: int = None):
        """Reset runtime species attributes back to their declared initializers."""
        for spec in self.specs.species_by_name.values():
            init = self.owner.species_initializers.get(spec.name, spec.init)
            value = braintools.init.param(init, self.owner.varshape, batch_size)
            if spec.name in self.specs.diffeq_set:
                getattr(self.owner, spec.name).value = value
            else:
                raw = getattr(self.owner, spec.name)
                if isinstance(raw, brainstate.State):
                    raw.value = value
                else:
                    setattr(self.owner, spec.name, brainstate.HiddenState(value))

    def value(self, name: str):
        """Return one species' current visible value."""
        raw = getattr(self.owner, name)
        return raw.value if isinstance(raw, brainstate.State) else raw

    def set_derivative(self, name: str, value):
        """Write one diffeq species derivative."""
        getattr(self.owner, name).derivative = value

    def factor_value(self, name: str):
        """Return one species' concrete factor value, defaulting to ``1``."""
        spec = self.specs.species_by_name[name]
        if spec.factor is None:
            return 1.0
        return self.specs.factors_by_name[spec.factor].value(self.owner)

    def to_scaled(self, name: str, value=None):
        """Convert a visible species value to its factor-scaled form."""
        if value is None:
            value = self.value(name)
        spec = self.specs.species_by_name[name]
        if spec.factor is None:
            return value
        return self.factor_value(name) * value

    def from_scaled(self, name: str, scaled):
        """Convert a factor-scaled value back to the visible domain."""
        spec = self.specs.species_by_name[name]
        if spec.factor is None:
            return scaled
        return scaled / self.factor_value(name)

class _Conserve:
    """Resolve algebraic species from declared conservation relations."""

    def __init__(self, owner, specs: _Specs, species: _Species):
        self.owner = owner
        self.specs = specs
        self.species = species

    def resolve(self, V=None) -> dict[str, Any]:
        """Return a full visible species map that satisfies all constraints."""
        values = {name: self.species.value(name) for name in self.specs.species_by_name}
        for conserve in self.specs.conserves:
            total_scaled = conserve.total(self.owner, V, values)
            algebraic_scaled = total_scaled
            for name in conserve.species:
                if name == conserve.algebraic:
                    continue
                algebraic_scaled = algebraic_scaled - self.species.to_scaled(name, values[name])
            values[conserve.algebraic] = self.species.from_scaled(conserve.algebraic, algebraic_scaled)
        return values

    def writeback(self, V=None):
        """Update cached algebraic species values on the owner object."""
        values = self.resolve(V)
        for name in self.specs.algebraic_names:
            raw = getattr(self.owner, name)
            if isinstance(raw, brainstate.State):
                raw.value = values[name]
            else:
                setattr(self.owner, name, brainstate.HiddenState(values[name]))


class _Flux:
    """Compute diffeq-species derivatives from reactions and sources."""

    def __init__(self, owner, specs: _Specs, species: _Species):
        self.owner = owner
        self.specs = specs
        self.species = species

    def compute(self, V, species_values: dict[str, Any], *, total_current=None) -> None:
        """Accumulate scaled-domain fluxes and write visible derivatives."""
        scaled_derivs = {
            name: 0.0 * self.species.to_scaled(name, species_values[name]) / u.ms
            for name in self.specs.diffeq_names
        }

        for reaction in self.specs.reactions:
            flux = self._reaction_flux(reaction, V, species_values)
            for name, stoich in reaction.lhs.items():
                if name in scaled_derivs:
                    contrib = stoich * flux
                    if hasattr(scaled_derivs[name], "unit") and hasattr(contrib, "in_unit"):
                        contrib = contrib.in_unit(scaled_derivs[name].unit)
                    scaled_derivs[name] = scaled_derivs[name] - contrib
            for name, stoich in reaction.rhs.items():
                if name in scaled_derivs:
                    contrib = stoich * flux
                    if hasattr(scaled_derivs[name], "unit") and hasattr(contrib, "in_unit"):
                        contrib = contrib.in_unit(scaled_derivs[name].unit)
                    scaled_derivs[name] = scaled_derivs[name] + contrib

        for source in self.specs.sources:
            try:
                contrib = source.flux(
                    self.owner,
                    V,
                    species_values,
                    total_current=total_current,
                )
            except TypeError:
                contrib = source.flux(self.owner, V, species_values)
            scaled_derivs[source.target] = scaled_derivs[source.target] + contrib

        for name in self.specs.diffeq_names:
            self.species.set_derivative(name, self.species.from_scaled(name, scaled_derivs[name]))

    def _reaction_flux(self, reaction: Reaction, V, species_values: dict[str, Any]):
        """Return the net reaction flux with its native quantity units."""
        forward = reaction.forward(self.owner, V, species_values)
        for name, stoich in reaction.lhs.items():
            value = species_values[name]
            forward = forward * (value if stoich == 1 else value ** stoich)
        if reaction.backward is None:
            return forward

        backward = reaction.backward(self.owner, V, species_values)
        for name, stoich in reaction.rhs.items():
            value = species_values[name]
            backward = backward * (value if stoich == 1 else value ** stoich)
        return forward - backward
