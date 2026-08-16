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
from braincell.quad.protocol import IndependentIntegration
from braincell.quad.protocol import state, hidden_state

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

    See Also
    --------
    Species : May reference this factor by name via its ``factor`` field.
    KineticIon : Template that consumes ``Factor`` entries through its
        ``factors`` class variable.
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

    See Also
    --------
    Factor : Optional conversion factor referenced by name via ``factor``.
    Reaction : References species by name in ``lhs``/``rhs``.
    Source : References one species by name via ``target``.
    Conserve : References species by name via ``species``/``algebraic``.
    KineticIon : Template that consumes ``Species`` entries through its
        ``species`` class variable.
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

    See Also
    --------
    Species : Declares the species named in ``lhs``/``rhs``.
    KineticIon : Template that consumes ``Reaction`` entries through its
        ``reactions`` class variable.
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

    See Also
    --------
    Species : Declares the species named by ``target``.
    KineticIon : Template that consumes ``Source`` entries through its
        ``sources`` class variable.
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

    See Also
    --------
    Species : Declares the species named in ``species``/``algebraic``.
    KineticIon : Template that consumes ``Conserve`` entries through its
        ``conserves`` class variable.
    """

    species: tuple[str, ...]
    algebraic: str
    total: Callable[[Any, Any, dict[str, Any]], Any]


class FixedIon(brainstate.mixin.Mixin):
    """Mixin for ions with a fixed ``Ci``/``Co``/``E`` triple.

    A concrete ion subclass calls :meth:`_init_fixed_ion` from its own
    ``__init__`` to materialize ``Ci``, ``Co``, ``E``, and ``valence`` as
    plain (non-state) attributes; none of the four evolve during
    simulation. ``Ci``, ``Co``, and ``valence`` each fall back to the
    class-level ``default_Ci``, ``default_Co``, and ``default_valence``
    (resolved via ``type(self)``, so a subclass's own overrides win)
    when passed ``None``. ``E`` has no such fallback and must be
    supplied explicitly.

    Raises
    ------
    ValueError
        If :meth:`_init_fixed_ion` is called with ``E`` left as
        ``None``.

    See Also
    --------
    InitNernstIon : Sibling mixin with fixed ``Ci``/``Co`` but a stored
        reversal potential computed once from the Nernst equation
        instead of supplied directly.
    DynamicNernstIon : Sibling mixin with a dynamic ``Ci`` state and a
        reversal potential recomputed from Nernst on every read.
    """

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
    r"""Mixin for ions with fixed ``Ci``/``Co`` and a stored Nernst ``E``.

    A concrete ion subclass calls :meth:`_init_nernst_ion` from its own
    ``__init__`` to materialize ``Ci``, ``Co``, ``valence``, and
    ``temp`` as plain (non-state) attributes, exactly as
    :class:`FixedIon` does for its own three. ``E`` is not stored
    directly at construction time; see Notes.

    Raises
    ------
    ValueError
        If :meth:`_init_nernst_ion` is called with ``temp`` left as
        ``None``.

    See Also
    --------
    FixedIon : Sibling mixin with a fixed reversal potential supplied
        directly instead of computed from Nernst.
    DynamicNernstIon : Sibling mixin with a dynamic ``Ci`` state and a
        reversal potential recomputed from Nernst on every read, rather
        than stored once.

    Notes
    -----
    ``_init_nernst_ion`` sets ``self.E = None`` immediately, and
    :meth:`_update_reversal` fills it in only later, from the
    ``_ion_init_state_hook`` and ``_ion_reset_state_hook`` lifecycle
    hooks. Between construction and the first state initialization or
    reset, ``E`` is ``None``.

    The stored formula, transcribed as ``_update_reversal`` writes it,
    is

    .. math::

        E = \frac{R \cdot \mathrm{temp}}{\mathrm{valence} \cdot F}
            \log\!\left(\frac{C_o}{C_i}\right)

    where :math:`R` is the gas constant and :math:`F` is the Faraday
    constant. The argument to the logarithm is :math:`C_o / C_i`
    (extracellular over intracellular), and ``valence`` divides inside
    the prefactor rather than appearing as a separate multiplicative
    term.
    """

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
    r"""Mixin for ions with a dynamic ``Ci`` state and Nernst-computed ``E``.

    Unlike :class:`FixedIon` and :class:`InitNernstIon`, a concrete ion
    subclass built on this mixin takes a ``Ci_initializer`` instead of a
    fixed ``Ci`` value: :meth:`_init_dynamic_nernst_ion` materializes
    ``Co``, ``valence``, and ``temp`` as plain (non-state) attributes and
    only *remembers* the initializer, in ``self._Ci_initializer``. The
    runtime ``Ci`` :class:`~brainstate.State` itself is created later, by
    :meth:`_ion_init_state_hook`, and refreshed by
    :meth:`_ion_reset_state_hook`; its time derivative is written by
    :meth:`_ion_compute_derivative_hook`, which delegates to the
    subclass's own :meth:`derivative`.

    Raises
    ------
    ValueError
        If :meth:`_init_dynamic_nernst_ion` is called with ``temp`` left
        as ``None`` -- the same failure mode as
        :meth:`InitNernstIon._init_nernst_ion`, not
        :meth:`FixedIon._init_fixed_ion`, which instead requires an
        explicit ``E``.

    See Also
    --------
    FixedIon : Sibling mixin with a fixed reversal potential and no
        dynamic state.
    InitNernstIon : Sibling mixin with fixed ``Ci``/``Co`` and a
        reversal potential computed once and cached, rather than
        recomputed on every read.

    Notes
    -----
    ``E`` is a property, not a stored attribute: every read recomputes
    it from the current ``Ci``, ``Co``, ``temp``, and ``valence`` via the
    same Nernst formula as :class:`InitNernstIon`,

    .. math::

        E = \frac{R \cdot \mathrm{temp}}{\mathrm{valence} \cdot F}
            \log\!\left(\frac{C_o}{C_i}\right)

    so ``E`` always reflects the live ``Ci`` state rather than a value
    cached at the last lifecycle hook.

    As with the sibling mixins, ``Co`` and ``valence`` fall back to
    ``type(self).default_Co`` and ``type(self).default_valence`` when
    left as ``None``. ``Ci_initializer`` falls back the same way, to
    ``type(self).default_Ci`` -- so omitting it does not leave ``Ci``
    unset, it seeds the dynamic state from the species' fixed default.

    :meth:`derivative` itself raises
    :class:`NotImplementedError` on this mixin; a concrete subclass
    must override it to return ``dCi/dt`` for its own dynamic ion
    model.

    Attributes
    ----------
    uses_total_current : bool, default False
        When ``True``, :meth:`_ion_compute_derivative_hook` precomputes
        the aggregate ion current -- reusing a step-start cached value
        on ``self._cached_total_current`` when present, otherwise
        evaluating ``self.current(V, include_external=True)`` -- and
        passes it to ``derivative(..., total_current=...)``. When
        ``False`` (the default), ``total_current`` is always ``None``.
    """

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
        self.Ci = state(
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

    Raises
    ------
    ValueError
        If :meth:`_init_kinetic_ion` is called with ``temp`` left as
        ``None``, or with ``substeps`` (after falling back to
        :attr:`default_substeps` when not supplied) less than ``1``.

    See Also
    --------
    Factor : Declares one conversion factor consumed via :attr:`factors`.
    Species : Declares one reaction-network species consumed via
        :attr:`species`.
    Reaction : Declares one mass-action reaction consumed via
        :attr:`reactions`.
    Source : Declares one source term consumed via :attr:`sources`.
    Conserve : Declares one algebraic conservation relation consumed via
        :attr:`conserves`.

    Notes
    -----
    Species live in visible units during integration. ``factor`` only mediates
    temporary visible/scaled conversion inside conservation and derivative
    mapping; species values are not stored in scaled form. Reaction laws remain
    in the visible domain, matching NEURON's ``KINETIC``/``COMPARTMENT``
    behavior.

    The NMODL semantics this template reproduces -- ``KINETIC`` reaction
    statements, ``COMPARTMENT`` volume/factor scaling, and the
    ``CONSERVE`` statement for algebraic species -- are documented by
    [1]_, the primary source for NMODL itself. [2]_ is the reference
    text for the surrounding NEURON mechanism model.

    References
    ----------
    .. [1] Hines, M. L., & Carnevale, N. T. (2000). Expanding NEURON's
           repertoire of mechanisms with NMODL. Neural Computation,
           12(5), 995-1007.
           doi:10.1162/089976600300015475
    .. [2] Carnevale, N. T., & Hines, M. L. (2006). The NEURON book.
           Cambridge University Press.
           doi:10.1017/CBO9780511541612

    Attributes
    ----------
    factors : tuple of Factor, default ()
        Conversion factors available to :attr:`species`, by name.
    species : tuple of Species, default ()
        The reaction-network species table. Must contain exactly one
        entry named ``"Ci"``.
    reactions : tuple of Reaction, default ()
        Mass-action reactions among the declared species.
    sources : tuple of Source, default ()
        Source terms contributing directly to a diffeq species'
        derivative.
    conserves : tuple of Conserve, default ()
        Algebraic conservation relations resolving one species per
        entry from the others in its pool.
    uses_total_current : bool, default False
        When ``True``, the template precomputes the aggregate ion
        current and passes it to ``derivative(..., total_current=...)``.
    default_solver : str, default "backward_euler"
        Solver name used when :meth:`_init_kinetic_ion` is not passed an
        explicit ``solver``.
    default_substeps : int, default 1
        Number of substeps run inside one parent update, used when
        :meth:`_init_kinetic_ion` is not passed an explicit
        ``substeps``.
    """

    factors: ClassVar[tuple[Factor, ...]] = ()
    species: ClassVar[tuple[Species, ...]] = ()
    reactions: ClassVar[tuple[Reaction, ...]] = ()
    sources: ClassVar[tuple[Source, ...]] = ()
    conserves: ClassVar[tuple[Conserve, ...]] = ()
    uses_total_current: ClassVar[bool] = False
    default_solver: ClassVar[str] = "backward_euler"
    default_substeps: ClassVar[int] = 1

    def _init_kinetic_ion(
        self,
        *,
        Co=None,
        temp=None,
        valence=None,
        species_initializers: dict[str, Any] | None = None,
        solver: str | None = None,
        substeps: int | None = None,
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

        if solver is None:
            solver = type(self).default_solver
        if substeps is None:
            substeps = type(self).default_substeps
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

    def make_integration(self, V, recursive_child: bool = True):
        """Advance this ion with its own solver and substep schedule."""
        with brainstate.environ.context(dt=brainstate.environ.get_dt() / self.substeps):
            brainstate.transform.for_loop(
                lambda i: self._step_solver(V, recursive_child),
                u.math.arange(self.substeps),
            )

    def _step_solver(self, V, recursive_child: bool = True):
        args = (V,) if recursive_child else (V, recursive_child)
        try:
            self.solver(self, *args, excluded_paths=(("channels",),))
        except TypeError as exc:
            if "excluded_paths" not in str(exc):
                raise
            self.solver(self, *args)

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
                factors=tuple(
                    type_factor if isinstance(type_factor, Factor) else Factor(*type_factor)
                    for type_factor in getattr(ion_type, "factors", ())
                ),
                species=tuple(
                    type_species if isinstance(type_species, Species) else Species(*type_species)
                    for type_species in getattr(ion_type, "species", ())
                ),
                reactions=tuple(
                    type_reaction if isinstance(type_reaction, Reaction) else Reaction(*type_reaction)
                    for type_reaction in getattr(ion_type, "reactions", ())
                ),
                sources=tuple(
                    type_source if isinstance(type_source, Source) else Source(*type_source)
                    for type_source in getattr(ion_type, "sources", ())
                ),
                conserves=tuple(
                    type_conserve if isinstance(type_conserve, Conserve) else Conserve(*type_conserve)
                    for type_conserve in getattr(ion_type, "conserves", ())
                ),
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
        self.diffeq_names = tuple(spec.name for spec in species if spec.name not in self.algebraic_set)
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
                raise ValueError(f"KineticIon species {spec.name!r} references unknown factor {spec.factor!r}.")

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

    def _species_value(self, spec, batch_size: int = None):
        """Materialize one species initializer at the owner's full state shape.

        ``braintools.init.param`` passes a bare scalar through unbroadcast,
        so a species declared as e.g. ``0.0 * u.mol / u.cm**2`` would start
        rank-0 while every sibling species is shaped ``varshape``. That
        shape is not stable: :meth:`_Conserve.writeback` later assigns the
        per-point value, silently growing the state mid-simulation — which
        breaks a ``jit``/``scan`` carry signature and, on a
        :class:`braincell.Cell`, the grouped hidden-state rank contract.
        Broadcasting here keeps one species set homogeneous from the start.
        """
        init = self.owner.species_initializers.get(spec.name, spec.init)
        value = braintools.init.param(init, self.owner.varshape, batch_size)
        target = tuple(self.owner.varshape)
        if batch_size is not None:
            target = (int(batch_size),) + target
        if tuple(getattr(value, "shape", ())) == target:
            return value
        return u.math.broadcast_to(value, target)

    def init(self, batch_size: int = None):
        """Materialize runtime species attributes from class declarations."""
        for spec in self.specs.species_by_name.values():
            value = self._species_value(spec, batch_size)
            if spec.name in self.specs.diffeq_set:
                setattr(self.owner, spec.name, state(value))
            else:
                setattr(self.owner, spec.name, hidden_state(value))

    def reset(self, batch_size: int = None):
        """Reset runtime species attributes back to their declared initializers."""
        for spec in self.specs.species_by_name.values():
            value = self._species_value(spec, batch_size)
            if spec.name in self.specs.diffeq_set:
                getattr(self.owner, spec.name).value = value
            else:
                raw = getattr(self.owner, spec.name)
                if isinstance(raw, brainstate.State):
                    raw.value = value
                else:
                    setattr(self.owner, spec.name, hidden_state(value))

    def algebraic_state(self, value):
        """Allocate an algebraic species state matching its siblings' class.

        :meth:`_Conserve.writeback` runs during simulation, *outside* the
        :func:`~braincell.state_grouping` scope the host establishes around
        ``init_state``. Reading the ambient scope there would silently
        produce a plain :class:`brainstate.HiddenState` on a
        :class:`braincell.Cell`, whose hidden states must all be grouped.
        Deriving the class from an already-allocated sibling species makes
        the decision independent of when the allocation happens.

        Falls back to the scoped factory only when no sibling has been
        allocated yet, which is the genuine initialization path — and that
        one does run inside the host's scope.
        """
        for name in self.specs.species_by_name:
            sibling = getattr(self.owner, name, None)
            if isinstance(sibling, brainstate.HiddenState):
                if isinstance(sibling, brainstate.HiddenGroupState):
                    return brainstate.HiddenGroupState(value)
                return brainstate.HiddenState(value)
        return hidden_state(value)

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
        """Update cached algebraic species values on the owner object.

        Runs every simulation step, so the allocation branch below is dead
        in normal flow — :meth:`_Species.init` has already turned every
        species into a ``State``. It stays defensive rather than raising,
        but routes through :meth:`_Species.algebraic_state` so that a late
        allocation cannot silently pick the wrong hidden-state class.
        """
        values = self.resolve(V)
        for name in self.specs.algebraic_names:
            raw = getattr(self.owner, name)
            if isinstance(raw, brainstate.State):
                raw.value = values[name]
            else:
                setattr(self.owner, name, self.species.algebraic_state(values[name]))


class _Flux:
    """Compute diffeq-species derivatives from reactions and sources."""

    def __init__(self, owner, specs: _Specs, species: _Species):
        self.owner = owner
        self.specs = specs
        self.species = species

    def compute(self, V, species_values: dict[str, Any], *, total_current=None) -> None:
        """Accumulate scaled-domain fluxes and write visible derivatives."""
        scaled_derivs = {
            name: 0.0 * self.species.to_scaled(name, species_values[name]) / u.ms for name in self.specs.diffeq_names
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
            forward = forward * (value if stoich == 1 else value**stoich)
        if reaction.backward is None:
            return forward

        backward = reaction.backward(self.owner, V, species_values)
        for name, stoich in reaction.rhs.items():
            value = species_values[name]
            backward = backward * (value if stoich == 1 else value**stoich)
        return forward - backward
