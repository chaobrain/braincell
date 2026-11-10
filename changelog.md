# Release Notes


## Unreleased

### Breaking Changes

- **`braincell.vis` is removed; plotting lives in `braintools.visualize.morphology`**
  (VIS-4a). The plots moved to braintools (0.5.0) and read neutral data
  (`MorphologyData`, `TopologyGraph`) instead of braincell objects. braincell
  converts its objects with three new methods, and every plot also accepts a
  `Morphology` or a `NodeTree` directly. `import braincell.vis` raises
  `ModuleNotFoundError` and `braincell.vis` is no longer an attribute. There
  is no deprecation period. `import braincell` no longer loads any plotting
  code. With `import braintools.visualize.morphology as bvm`:

  | Old (`braincell.vis.`) | New |
  |---|---|
  | `plot2d`, `plot3d`, `plot_movie`, `plot_traces`, `compare_morphologies`, `compare_values` | `bvm.` the same names, same parameters; the first argument may be a `Morphology` or a `MorphologyData` |
  | `plot_dendrogram`, `plot_topology`, `plot_sholl`, `plot_branch_order_histogram` | `bvm.` the same names, same parameters |
  | `plot_point_topology(node_tree, ...)` | `bvm.plot_topology_graph(node_tree, ...)`; also takes a `TopologyGraph`, and `frame=` for multi-frame values |
  | `plot_cell_topology(cell, level=L, region=R, locset=S, value=V, **style)` | `bvm.plot_topology_graph(cell.topology_data(L, region=R, locset=S, value=V), **style)`; `preset`, `layout`, `coverage_mode`, `highlight_color` and the colour arguments stay on the plot |
  | `ValueSpec` | `bvm.ValueSpec`, with a new `space=` (`"branch"`, `"segment"`, `"centerline"`) |
  | `OverlaySpec` | `bvm.OverlaySpec`; `region` / `locset` take neutral data or evaluated masks |
  | `LayoutConfig` | `bvm.LayoutConfig` without `legacy_root_child_span_rad` |
  | `LayoutCache` | `bvm.LayoutCache`; `get_or_build` takes a `MorphologyData` |
  | `VisHooks`, `PickInfo`, `save_figure`, `VisDefaults`, `configure_defaults`, `get_defaults`, `reset_defaults`, `theme`, `publication_theme`, `PublicationTheme`, `PUBLICATION_BRANCH_TYPE_COLORS`, `PUBLICATION_RC_PARAMS` | `bvm.` the same names, unchanged |
  | `set_defaults` | `bvm.configure_defaults` (it was the same function) |
  | `braincell.vis.traces.TracesResult`, `braincell.vis.movie.MovieResult` | `bvm.TracesResult`, `bvm.MovieResult` |
  | `root_layout="legacy"` (deprecated) | removed; `ValueError` naming `"type_split"` |

  | Also | Now |
  |---|---|
  | `Morphology.vis2d` / `vis3d`, `Branch.vis2d` / `vis3d` | same signatures, drawn by `bvm.plot2d` / `bvm.plot3d`; `chooser=` takes a `braintools.visualize.morphology.backend.BackendChooser` |
  | `pip install braincell[vis]` | still works; it now installs `braintools[morphology]` (networkx, plotly, pyvista) |

  New methods:
  - `Morphology.to_vis_data()` returns a `MorphologyData` (it is also
    `__morphology_data__`), cached until the morphology is edited.
  - `Cell.topology_data(level="node", *, value=None, region=None, locset=None)`
    returns a `TopologyGraph` at the node, CV or branch level. It raises the
    same errors as `plot_cell_topology` did, and a `(T, n)` value array gives
    one frame per row.
  - `NodeTree.to_topology_data()` returns the solver-node graph (it is also
    `__topology_data__`).
- **`braincell.quad` uses the `braintools.quad` registry** (QUAD-2b).
  braincell no longer has an integrator registry of its own. Integrator names
  resolve through `braintools.quad`: braincell's stateful steps are registered
  there under the new `'module'` kind, and the pure-function ODE steppers of
  braintools are wrapped into module steps. The explicit Runge-Kutta steps now
  use the braintools kernels; their trajectories match the previous braincell
  kernels to within 1e-12 relative error in float64, except `ralston4` (see Bug
  Fixes). braintools' embedded methods (`dopri5`, `rk45`, ...) can now be
  selected by name as fixed-step solvers. No deprecation period:

  | Old | New |
  |---|---|
  | `get_registry()`, `IntegratorRegistry`, `IntegratorEntry` | `braincell.quad.integrator_info(name)` (returns a `braintools.quad.MethodInfo`), `braincell.quad.list_integrators()` |
  | `all_integrators` | `{n: get_integrator(n) for n in list_integrators()}` |
  | `@register_integrator(name, aliases=..., category=..., order=..., description=...)` | `braintools.quad.register_method(name, fn, kind='module', aliases=..., category=..., order=..., description=...)`; a pure-function stepper registered with `kind='ode'` is wrapped automatically |
  | `get_integrator("rk2")`, `solver="rk2"`, `braincell.quad.rk2_step` | `"ralston2"` / `ralston2_step` (same numbers), or `"heun2"` / `heun2_step` for Heun's method. `"rk2"` raises `ValueError` and `rk2_step` raises `AttributeError`, both naming the replacements |
  | `get_integrator("explicit")` | `"euler"` |
  | `get_integrator("implicit_euler")`, `solver="implicit_euler"`, `braincell.quad.implicit_euler_step` | `"crank_nicolson"` / `crank_nicolson_step` (QUAD-2c; the trapezoidal rule, now solved to convergence; results change on nonlinear models, see below), or `"backward_euler"` for first-order implicit Euler. The old names raise and name both replacements |
  | `SingleCompartment(solver='rk2')` default | default `solver='ralston2'`; the numbers are unchanged |
  | the other `*_step` functions | unchanged names and calls; `module_step_from_ode(ode_step)` wraps any braintools ODE stepper |

  `get_integrator("rk2")` raises for at least one minor version. Once the error
  is removed, `"rk2"` will resolve to braintools' `rk2`, which is Heun's method
  and not the Ralston method that `rk2` meant in braincell 0.2.
- braincell requires `braintools>=0.5.0`.
- **`implicit_euler` is now `crank_nicolson` and its Newton loop converges**
  (QUAD-2c). The step solved the trapezoidal (Crank-Nicolson) residual but
  was registered as `implicit_euler` with `order=1`, and its Newton loop
  stopped after one iteration. `crank_nicolson` (`order=2`) iterates until
  `max|dy|/(1+|y|) <= eps**0.75` or 20 iterations, with one
  `n_state x n_state` Jacobian per neuron instead of a dense Jacobian over the
  whole population. On linear models the result is unchanged; on nonlinear
  ones it changes: a Hodgkin-Huxley neuron (10 ms, dt = 0.1 ms, float64)
  differs from the old step by 2.6e-2 relative (1.9 mV).

### New Features

- **The same `paint` / `place` verbs on a Cell and its views** (CELL-22).
  `cell.paint(*mechanisms)` without a region paints the whole cell
  (`AllRegion()`), so a point neuron and a morphological one are declared
  the same way. Spatial views declare too, and return the root Cell for
  chaining: `cell.soma.paint(m)` is `cell.paint(branch_in("type",
  ["soma"]), m)`, `cell.on(region).paint(m)` is `cell.paint(region, m)`
  (nested region views paint the intersection), and
  `cell.loc(locset).place(m)` is `cell.place(locset, m)` for any point
  mechanism, clamps included. `cell[indices].loc(locset).place(syn)`
  places on the selected members, as `cell[indices].place(locset, syn)`
  does. Errors at the call:
  - `ValueError` for `paint` on a population view (`cell[0].paint(m)`):
    declarations are the same for every member;
  - `ValueError` for `paint` on a location or `cv[...]` view, `place` on a
    region view, and a location view nested in another spatial view;
  - `TypeError` for `cell.paint()` without a mechanism, and for a region or
    locset passed to a view that already has one.

  Existing calls are unchanged: a region first still paints that region,
  and `cell[indices].place(locset, ...)` still returns the view.

- **Mechanism classes are used by identity; no registration needed**
  (CELL-25). `mech.Channel(MyChannel, ...)`, `mech.Ion(MyIon, ...)` and now
  `mech.Synapse(MySynapse, ...)` accept a class that was never registered:
  the declaration stores an identity key (`MechanismRegistry.key_for`) that
  resolves to that very class, so custom `braincell.Channel` subclasses
  written for `SingleCompartment` models run on a `Cell` unchanged. Two
  unregistered classes that share a `__name__` stay distinct (the second gets
  `module.qualname`). A registered class still resolves to its registry name,
  so `mech.Synapse(ExpSyn) == mech.Synapse("ExpSyn")`, and strings are still
  resolved by the registry at `init_state`. An instance, or a class outside
  the category (a non-`IonChannel`, an `Ion` or `Synapse` passed to
  `Channel`, ...), raises `TypeError` when the declaration is built.
  Previously an unregistered class silently fell back to `cls.__name__` and
  failed at `init_state` with `KeyError`, so no working call changes.

- **`Morphology.from_soma(length=10 * u.um, radius=5 * u.um, name="soma")`**
  builds a morphology of one cylindrical soma branch in one call, replacing
  `Morphology.from_root(Branch.from_lengths(lengths=[L], radii=[r, r],
  type="soma"), name="soma")`. Its membrane area is the lateral area
  `2 * pi * r * L`; the defaults give `SingleCompartment`'s default area.
  It is the geometry of the planned point-neuron `Cell` (CELL-21a). No
  public `Morphology` name may equal a branch type or a common short branch
  name (`dend`, `apic`), because public names are reserved as branch names;
  a test now guards this.

- **Point neurons as `Cell`s: `Cell.point(...)` and `cv_policy=Lumped()`**
  (CELL-21b). `Cell.point(pop_size, length=, radius=, V_init=, ...)` is
  shorthand for `Cell(Morphology.from_soma(length=, radius=), pop_size=...,
  cv_policy=Lumped())`. A lumped cell is declared like any `Cell` (`paint`,
  `place`, `record`, `run`, `update`) but has no CV axis: `V`, `spike` and
  mechanism states have shape `pop_size` (plus a leading batch axis), as in
  `SingleCompartment`, and it reproduces `SingleCompartment` numerically
  (float64 voltages agree to 1e-13 mV with `rk4`, `euler`, `ralston2` and
  `exp_euler`) at about the same cost per step.

  ```python
  cell = bc.Cell.point(100, length=20 * u.um, radius=10 * u.um, V_init=-65 * u.mV)
  cell.paint(bc.filter.AllRegion(), bc.mech.Ion("SodiumFixed", E=50 * u.mV),
             bc.mech.Channel("Na_HH1952", g_max=120 * u.mS / u.cm**2), ...)
  cell.place(bc.mech.CurrentClamp(durations=50 * u.ms, amplitudes=0.1 * u.nA))  # the one site
  cell.record("v", bc.observe.state("v"))
  result = cell.run(dt=0.025 * u.ms, duration=50 * u.ms)  # cell.V.value.shape == (100,)
  ```

  - `place` may omit the locset on a lumped cell; an explicit locset must be
    the CV midpoint (`RootLocation(0.5)`, `cell.cv_midpoints`), otherwise
    `place` raises `ValueError`. Only current clamps can be placed.
  - `Lumped()` on a morphology with more than one branch raises `ValueError`
    when the cell is built or the policy assigned.
  - Initial gates are at steady state for `V_init` (as on cable cells;
    `SingleCompartment.init_state()` leaves them at 0). `V_init=None` starts
    at the resting potential.
  - Views (`cell.channels`, `cell.ions`, `cell.synapses`, `cell.on(...)`,
    `cell.cv[...]`, `cell[...]`, `cell.geometry`), trainable parameters,
    synapses, `add_reduction` and `Network.add_population` raise
    `NotImplementedError` on a lumped cell for now.
- **`Cell(solver=None)` resolves by mode.** The default is now `None`:
  `"staggered"` on a cable cell (unchanged) and `"ralston2"` on a lumped
  cell. A lumped cell accepts every single-step solver of
  `braincell.quad.get_integrator` except `"staggered"` and `"dhs_voltage"`,
  which raise `ValueError`.

- **`parameter_info()` on every runtime Channel, Ion and Synapse, and
  `parameter_units` for parameters without a unit default** (MECH-25).
  `bc.Channel.parameter_info()` and `bc.Ion.parameter_info()` return each
  constructor parameter's default and unit prototype, as
  `Synapse.parameter_info()` already did. A class whose default cannot carry
  a unit declares it, e.g. `parameter_units = {"g_max": u.mS / u.cm**2}` on a
  channel with `g_max=None`; such values were unchecked before.

### Changes & Improvements

- **Parameter unit errors name the mechanism, the value and the declaring
  line** (MECH-25). `init_state()` errors now read like `Channel 'leak' (IL)
  parameter 'g_max' has units incompatible with mS / cm^2; got 0.1 mV.
  Painted on AllRegion() at model.py:12.`, and name the spatial callable a
  value came from. `View.set` and trainable registration errors also name
  the mechanism and the parameter. The exception types and the phrases
  `requires a quantity compatible with`, `has units incompatible with` and
  `must be dimensionless` are unchanged. A braintools initializer passed as a
  Channel parameter in a Cell now says initializers are not supported there,
  instead of asking for a quantity.

- **`Cell.run` costs less per call.** Each call now dispatches one cached
  compiled program: segment start / stop times and the selection of
  fixed-schedule samples moved inside it, and the schedule checks are
  computed once per run shape. A 10-step run of a 100-member cell with one
  voltage recording dropped from 1.12 to 0.42 ms per call on CPU and from
  2.58 to 0.90 ms on GPU (RTX 3090); results are unchanged.
  `RunResult.start_time` is now the cell time at the start of the call
  rather than `stop_time - duration`, so contiguous segments chain exactly.
- `Cell.solver_name` records the canonical name of braincell's own steps:
  `Cell(solver="stagger").solver_name == "staggered"`.
- `ind_exp_euler` computes each state's update with
  `braintools.quad.ode_expeuler_step` (the same formula; trajectories are
  unchanged).
- Docstrings of `CalciumFirstOrder` and the deep cerebellar nucleus
  mechanisms (`*_SU2015_DCN`, `Cdp*_SU2015_DCN`) state their sources more
  precisely (MECH-13). `CalciumFirstOrder` now cites Pinsky & Rinzel (1994),
  whose dendritic calcium equation it implements, and notes that the source's
  calcium variable is dimensionless while the defaults here are in `mM`. The
  DCN docstrings now say which constants the supplementary material of
  Steuber et al. (2011) prints, and `SK_SU2015_DCN` notes that the
  supplement's `tau_z` is 60 times the value in the authors' own code, which
  this class follows. No behaviour changes.
- The docstrings of `Markov` and of `Cell`'s `subsolver` say which
  subsolvers suit stiff Markov channels (MECH-22). The catalogue's sodium
  Markov channels are stiff, so use `backward_euler` (the default; first
  order, voltage-clamp current off by 7-20 % at `dt = 0.025 ms`, as in
  NEURON) or `exp_euler` (exact while the voltage is constant over a step,
  about 4-6 times the cost per step), never an explicit method; and one
  declaration selects `exp_euler` with
  `bc.mech.Channel(..., solver="exp_euler", substeps=1)`. No defaults
  change.

- **Breaking:** braincell now requires `jax>=0.11` (it was `>=0.9`), in
  the base requirement and in the `cpu`, `cuda12`, `cuda13` and `tpu`
  extras. Users on JAX 0.9 or 0.10 must upgrade JAX before upgrading
  braincell.
- CI tests Python 3.12, 3.13, 3.14 and 3.15, and the package classifiers
  declare exactly those four (`requires-python` is unchanged, `>=3.12`).
  Every Python runs against the latest JAX; JAX 0.11.0 (the floor) is
  tested on 3.12 only. A `lint` step fails when the classifiers differ from the Python
  versions CI tests.

### Bug Fixes

- The optional gating current of `Nav1p1_MA2025_BC` and `Nav1p1_RI2021_SC`
  (`gateCurrent != 0`) was 1e18 times too large, for the same reason as the
  Kv entry below: the `(1e12)` and `(1e6)` factors of the `.mod` formula are
  NEURON's unit conversions and were applied again to quantities that carry
  units (MECH-6c). It is now `g_max / gunit * e0 * zgate * dq/dt` and matches
  NEURON to 2.8e-15 relative along a voltage-clamp trace. The default
  `gateCurrent = 0` is unaffected. The docstrings of `Na_HH1952` (the defaults
  run Hodgkin-Huxley's 6.3 degC rates unscaled; `temp_ref` = 6.3 degC
  reproduces NEURON's `hh`), `HCN1_MA2025_BC`, `HCN1_MA2024_PC`,
  `HCN1_RI2021_SC` (the `.mod` Q10 reference is 37 degC; the 23 degC `temp`
  default is an operating temperature) and `HCN_HM1992` (its `+43 mV`
  default is unchecked) are corrected.

- The optional gating current of `Kv1p1_MA2025_BC`, `Kv1p1_MA2024_PC`,
  `Kv1p1_RI2021_SC`, `Kv1p1_MA2020_GoC`, `Kv1p1_MA2020_GrC` and
  `Kv3p3_MA2024_PC` (`gateCurrent != 0`) was 1e18 times too large: the
  `(1e12)` and `(1e6)` factors of the `.mod` formula are NEURON's unit
  conversions (pS to S, C/ms to mA) and were applied again to quantities that
  already carry units (MECH-6b). It is now `g_max / gunit * e0 * 4 * zn * dn/dt`
  and matches NEURON to 1.2e-13 relative along a voltage-clamp trace. The
  default `gateCurrent = 0` is unaffected.

- `Kv1p5_MA2020_GrC` with a nonzero `gnonspec` computes the nonspecific
  reversal as its `.mod` does (MECH-6b): `z = R*(celsius + 273.15)/F` there is
  a number in volts that the formula reads as millivolts, so the reversal is
  1000 times closer to 0 mV than the Nernst-like value the class used. The
  `ino` component now matches NEURON to 5e-16 relative (it was off by up to
  0.9). The default `gnonspec = 0` is unaffected.

- `Cav2p1_RI2021_SC`, `Cav2p1_MA2024_PC` and `Cav2p1_MA2025_BC` compute
  their GHK drive with the constants their `.mod` sources declare
  (`F = 9.6485e4 C/mol`, `R = 8.3145 J/(K mol)`, `T = 273.19 K + celsius`)
  instead of the CODATA constants and exact Kelvin of `ghk_flux` (MECH-6a).
  The calcium current changes by about 3e-4 relative; against NEURON the
  steady-state I-V error drops from 3.1e-4 to 1.5e-15 relative. The
  `_Frozen` variants already used these constants, so their forward current
  now equals the unfrozen classes'.

- Current-driven calcium pools read their current consistently (MECH-2).
  `CdpHVA_SU2015_DCN`, `CdpLVA_SU2015_DCN`,
  `ToyCaBindingIcaSourceKinetic_SU2015_DCN` and
  `ToyCaPumpFactorKinetic_SU2015_DCN` read `kCa`/`kCal` in `1/coulomb`
  instead of taking its bare mantissa, so `3.45e-4 / u.kcoulomb` no longer
  drives `Ci` 1000 times faster than the equal default `3.45e-7 / u.coulomb`
  in a directly constructed pool (`Cell` already converted the unit); defaults
  are unchanged. `Cell.cache_ion_total_currents` no longer retries a
  `TypeError` without `include_external`, which silently dropped the external
  currents from the step-start snapshot.

- `Network.add_population` (and `Population`) reject a `SingleCompartment`
  with a `TypeError` that says to build the neuron as a `Cell`. The check
  accepted any model with a one-dimensional `pop_size`, so a
  `SingleCompartment` passed it and then failed with
  `AttributeError: ... '_bind_network_owner'`. The `state_grouping` docstring
  and a `SingleCompartment` comment no longer claim a `Network` can hold both
  model types, and the `SingleCompartment.compute_derivative` docstring no
  longer claims `I_ext` may be a total current: it must match the unit system
  of `C`. `examples/single_compartment/SC05_*.py` (and SC06, which imports
  it) failed at construction because it assigned to the read-only `area`
  property; the factor is now `area_factor`.

- An integer-valued `V_init` or `V_th` (e.g. `V_init=-65 * u.mV`, also as the
  result of a callable `V_init`) made `Cell.run` fail with a confusing
  "scan body function carry input and carry output must have equal types"
  error, because the voltage was stored as an integer array. The values are now
  converted to the default float dtype (`brainstate.environ.dftype()`) when the
  cell materializes them, so the run matches the float declaration bitwise.

- `Network.run` / `Network.prepare_run` with `event_backend="brainevent"`
  run again (CELL-20). The backend used `brainevent.coomv`, which brainevent
  removed in 0.1.0, so on every brainevent release since then it raised
  `RuntimeError`. It now uses `brainevent.csrmv` (available from brainevent
  0.0.7, the existing floor, through 0.4.1) and matches `"scatter"` to float
  rounding, gradients included. `event_backend="auto"` now always resolves to
  `"scatter"`, which it already did on brainevent >= 0.1.0; with the default
  `brainevent_backend="jax_raw"` the brainevent operator lowers to the same
  gather and scatter-add. A brainevent without `csrmv` is rejected at setup
  with a `RuntimeError` that points to `"scatter"`.

- `ralston4` uses Ralston's exact coefficients. braincell stored them rounded
  to 8 decimal places (e.g. `0.17476028` for `(263 + 24*sqrt(5))/1812 =
  0.1747602822...`), which breaks the fourth-order conditions at the 1e-8
  level. Trajectories change slightly: on a Hodgkin-Huxley neuron (10 ms,
  float64) by at most 6.8e-9 relative at dt = 0.1 ms and 1.6e-9 at
  dt = 0.025 ms.

- `SK_SU2015_DCN`, `Cav1p2_MA2020_GoC`, `Cav1p2_MA2025_BC`,
  `Cav1p3_MA2020_GoC` and `Cav1p3_MA2025_BC` can now run in a `Cell` that
  does not paint a calcium ion. Their calcium-only gates reset to a scalar
  steady state there, and `Cell.run` failed with `TypeError: scan body
  function carry input and carry output must have equal types`.
  `HH.reset_state` now broadcasts every gate to the shape `init_state`
  allocated; gates that already had that shape are unchanged (the voltage
  traces of the other 107 registered channels are bitwise identical).
- `CalciumFirstOrder` can now run. Its `derivative` compared a current
  density with `0 mM` and raised `UnitMismatchError`, with or without a
  calcium channel, and a `Cell` rejected a unit-carrying `alpha` as
  "must be dimensionless" because the constructor defaults were bare
  numbers. `alpha` is now a concentration per time per current density
  and `beta` a rate, with defaults `alpha=0.13 * u.mM / u.ms / (u.uA /
  u.cm**2)` and `beta=0.075 / u.ms` (the numbers are unchanged), and
  `derivative` returns a concentration per time. Inward current stays
  positive and rectified, and a pool without channels decays as if the
  current were zero. Through a `Cell`, a bare or wrongly dimensioned
  `alpha` or `beta` now fails at `init_state()` with an error naming the
  field. No working call changes: the class could not run before.
- The `staggered` and `dhs_voltage` solvers no longer accumulate float32
  rounding with the size of the membrane voltage. The DHS step solved for
  the new voltage itself, and eliminating onto a sealed-end boundary row
  cancels nearly equal terms, so every step carried a relative error of
  about 1e-6 times |V|. On a passive soma, float32 drifted about 1e-2 mV
  from float64 within 1 ms at any dt; it now stays within 2e-5 mV. The
  step now solves for the voltage increment. float64 results change only
  at roundoff (relative 1e-12 or less over 50 ms) and are closer to the
  exact implicit-Euler solution.


## Version 0.2.0

This release redesigns heterogeneous network construction, simulation and
recording, and integrates the CV-space runtime with trainable channels and
optimization workflows. Parameter learning is unified across channels, ions,
synapses and networks, and runtime geometry can be trained. The repository
layout and workflow namespaces are reorganized. A package-wide cleanup fixes a
series of latent bugs, removes dead API and speeds up morphology and
visualization work. `Cell` always carries a population axis, and its
visualization methods move to `braincell.vis.plot_cell_topology`.

BrainCell now requires Python ≥ 3.12, JAX ≥ 0.9 and BrainPy ≥ 2.9.0.

### Breaking Changes

- **`Cell` no longer exposes visualization methods.**
  `Cell.vis_topology`, `Cell.vis_node`, `Cell.vis_cv`, and
  `Cell.vis_branch` are removed with no deprecation shim, and so are the
  same four names on the `MultiCompartment` alias. One free function in
  `braincell.vis` replaces all four, with the topology level chosen by
  `level=`:

  ```python
  # before
  cell.vis_node(value="V", cmap="viridis")
  cell.vis_cv(region=region)
  cell.vis_branch(region=region)
  cell.vis_topology(level=lvl, region=region)

  # after
  import braincell.vis as vis
  vis.plot_cell_topology(cell, value="V", cmap="viridis")   # level="node" is the default
  vis.plot_cell_topology(cell, level="cv", region=region)
  vis.plot_cell_topology(cell, level="branch", region=region)
  vis.plot_cell_topology(cell, level=lvl, region=region)
  ```

  This puts `Cell` in line with every other model object: `Morphology`
  and `Branch` already delegate to `braincell.vis` rather than
  implementing plots themselves.

  Two smaller shifts come with it. **`show=` is gone** — no function in
  `braincell.vis` takes it, so call `matplotlib.pyplot.show()` yourself
  (in Jupyter the inline backend still draws at the end of the cell, so
  `show=False` call sites simply drop the argument). And
  **`level="branch"` now reports every unsupported argument at once**
  (`... does not support: locset, value, cmap.`) instead of raising on
  the first one it finds.

- **`Cell` now always carries a population axis.** `pop_size` defaults to
  `1` and an explicitly empty `pop_size=()` raises `ValueError`. Runtime
  state is therefore always at least two-dimensional: `Cell.V` is shaped
  `pop_size + (n_cv,)` and point-space arrays `pop_size + (n_point,)`.
  Code that relied on the old rank-0 default sees one extra leading axis
  of length 1; `reshape(-1)` or indexing `[0]` recovers the previous view.
- `DiffEqState` is now a marker mixin rather than a concrete state class.
  `DiffEqState(value)` raises `TypeError`. The two concrete classes are
  `DiffEqSingleState` (over `brainstate.HiddenState`, used by
  `SingleCompartment`) and `DiffEqGroupState` (over
  `brainstate.HiddenGroupState`, used by `Cell`). Every
  `isinstance(x, DiffEqState)` check keeps working unchanged.
- `braincell.diffeq_state` is renamed to `braincell.state`. No alias is
  provided.

  ```python
  # before
  self.m = braincell.DiffEqState(init)

  # after (preferred — picks the right class for the host)
  self.m = braincell.state(init)

  # after (explicit)
  self.m = braincell.DiffEqSingleState(init)
  ```

### New Features

- **`braincell.MultiCompartment` is a new alias of `braincell.Cell`.** The
  same class object under a name that spells out what it models, which
  reads better next to `braincell.SingleCompartment`. Nothing about `Cell`
  changes; both names remain supported.
- **Hidden-state classes now match their host model.** Every hidden state
  owned by a `Cell` is a `brainstate.HiddenGroupState` — `Cell.V` is the
  new `braincell.DiffEqGroupState` — so the trailing compartment/point
  axis is exposed as a group of independently traced hidden states, which
  is what eligibility-trace learning requires. `SingleCompartment` has no
  spatial axis and uses `DiffEqSingleState`, which is itself a
  `brainstate.HiddenState`. `DiffEqGroupState` derives from `DiffEqState`,
  so every solver selects it unchanged.
- **`state` / `hidden_state` / `state_grouping`** are exported for
  custom mechanisms. Channel, ion, and synapse code is shared by both host
  models, so the right class cannot be chosen at the creation site.
  `DiffEqState(...)` now raises `TypeError`; write `state(...)` in a
  custom `init_state` instead, and the right class is chosen per host.

- **Three packages were added in this release and were missing from these
  notes** (recorded afterwards from the history: #154, #155, #156, #157).
  - **`braincell.trainable`** (#154) maps latent parameters to physical
    ones for gradient-based learning: `parameter`, `parameterized` and
    `scale` declare a `ParameterSource`; `ParameterSet`, `ParameterBinding`
    and `TrainableManager` bind them to channel, ion, synapse and network
    parameters.
  - **`braincell.reduction`** (#155) holds interchangeable lightweight
    execution models for detailed cells: the `ReductionModel` interface
    with its context, input and output types, and three toy accumulators
    (`EventAccumulatorReduction`, `PayloadAccumulatorReduction`,
    `SynapticKernelAccumulatorReduction`).
  - **`braincell.experimental`** (#156, #157) is explicitly unstable.
    `experimental.optim` provides BPTT/RTRL gradient engines
    (`RolloutGradientEngine`, `TrajectoryGradientEngine`,
    `build_rollout_value_and_grad`, `build_trajectory_value_and_grad`),
    including gradients through trainable runtime geometry; the package
    also holds the scheduled-event cell and delivery helpers. Its API may
    change in any release.

The remaining entries in this section harden the declarative channel
template layer that `braincell.channel` is built on. The catalogue's
numeric output is unchanged: each either adds a check that shipped
channels already pass, or moves code they already shared.

- **`OhmicHH`** — an `HH` subclass supplying the ohmic current
  `g_max * conductance_factor(...) * (E - V)`. 63 channels that restated
  it verbatim now inherit it; a channel driven by a fixed `self.E`
  overrides `reversal_potential()` instead. GHK-flux and
  permeability-scaled channels keep inheriting `HH` directly.
- **`Gate.time_unit`** — the unit a bare `f_*_tau` / `f_*_alpha` /
  `f_*_beta` return is read in, `u.ms` by default. Rate methods may now
  return properly united quantities instead.
- **`Gate.clip` / `Markov.clip_states`** — explicit, per-gate control of
  whether a state is projected into `[0, 1]` at the point of use.
  `Gate.clip` is off by default (NEURON does not clip HH gates);
  `clip_states` is on, preserving existing `Markov` behaviour.
- **String references in gate metadata** — `Gate(q10="q10")` resolves
  the named attribute off the instance, replacing 75
  `lambda self: self.q10` closures that were unpicklable and opaque to
  tooling. The callable form still works.
- `Gate`, `Transition`, `HH`, `OhmicHH`, `Markov` and `ghk_flux` are
  exported from `braincell.channel`; they previously had to be imported
  from the private `braincell.channel._base`.

### Changes & Improvements

- Require `brainpy>=2.9.0`, where `brainpy.state` is part of BrainPy itself.

- **Definition-time validation.** `HH` and `Markov` resolve and check
  `gates` / `pairs` in `__init_subclass__`. A mistyped gate name, a
  duplicate, a gate defining neither or both rate forms, a transition
  naming a missing rate method, a `dependent_state` outside the state
  set, and a Markov class with fewer than two states are now errors when
  the class is created rather than at `reset_state()` or not at all.
- **Unit checking on the derivative.** Gate and transition rates are
  dimension-dispatched, and every state derivative is asserted to be an
  inverse time before it reaches the integrator. A dimensioned `phi`
  used to yield a `mV / ms` derivative silently.
- HH gate methods receive only the ion arguments their signature
  declares, matching what `Markov` already did. This removed 178 lines
  of forwarding boilerplate from `potassium_sodium.py`.

### Bug Fixes

- Cell topology plotting (now `braincell.vis.plot_cell_topology`) and
  node-local runtime inspection (`cell.runtime_nodes[i].ions[...]`)
  rejected any field with a population axis, so they failed for every
  `pop_size` beyond the old rank-0 default. They now collapse a
  single-member population and otherwise raise naming the field and
  `pop_size`.
- Runtime CV inspection (`cell.runtime_cvs[i].ions[...]`) blamed a
  plotting function for its own errors: a population-axis refusal read
  `Cell.vis_cv(...) addresses a single morphology`, on a code path with
  no plotting in it. Messages now name the entry point that was actually
  called. The dependency message from the shared topology renderer no
  longer hardcodes `plot_point_topology(...)` either.
- Ion baseline broadcasting in `_sync_runtime_ion` ignored the population
  axis, disagreeing with the sibling code path.
- Dense channel construction dropped the population axis when a channel's
  parameters were all scalars, producing a rank-1 gate state that failed
  as a `jit`/`scan` carry mismatch.
- A kinetic-ion species declared with a scalar initializer (for example
  `CdpCR_MA2020_GrC.pumpca`) started rank-0 and was silently reshaped
  mid-simulation by the conservation write-back; it is now allocated at
  full shape.
- `set_module_as` assigned the public module path to `__name__` instead of
  `__module__`, so every function it decorated reported itself as being
  called `"braincell.quad"` while still advertising the private module it
  is defined in. Both are now correct: `braincell.quad.rk4_step.__name__`
  is `'rk4_step'` and its `__module__` is `'braincell.quad'`. This also
  changes `IntegratorEntry.module` from the private defining module to the
  public export path, and makes `dhs_voltage_step` consistent with the
  other integrators.
- `init_state()` refused nothing: a gate or Markov state whose name
  matched a constructor parameter silently replaced that parameter with
  a `DiffEqState`. It now raises, naming the collision.

### Deprecations

- `Markov` subclasses that do not declare `dependent_state` emit a
  `DeprecationWarning`. The implicit fallback — "the last state
  discovered while scanning `pairs`" — makes a reordering of `pairs`
  silently change which state is eliminated, and will be removed. All
  shipped channels now declare it explicitly.


## Version 0.1.0

This is a landmark release. BrainCell evolves from single-compartment
Hodgkin–Huxley modeling into a complete **multi-compartment, morphologically
detailed** neuron simulation framework in JAX. It introduces a morphology layer,
a control-volume discretization engine with pluggable policies, a compute
runtime, morphology IO (SWC / ASC / NeuroMorpho.Org), a declarative
mechanism system, location/region selection filters, and a 2D/3D visualization
stack.

### New Features

- **Multi-Compartment Morphological Modeling** (#69)
  - New `Cell` declaration frontend and frozen `RunnableCell` runtime for
    simulating branched morphologies.
  - High-level `rcell.run(dt=, duration=)` driver returning a structured
    `RunResult`.

- **Morphology Layer** (#68, #69)
  - Immutable `Branch` with typed subclasses (`Soma`, `Dendrite`, `Axon`,
    `BasalDendrite`, `ApicalDendrite`, `CustomBranch`).
  - Mutable `Morphology` tree with whole-morphology metric snapshots
    (`MorphoMetric`).

- **Control-Volume Discretization** (#72, #74, #88)
  - Pure-functional CV layer with composable policies: `CVPerBranch`,
    `DLambda`, `MaxCVLen`.
  - Paint/place rule machinery for mapping mechanisms onto morphology by region.

- **Compute Runtime** (#74, #88)
  - Execution-graph lowering (`NodeTree`), scheduling, runtime-state
    installation, and channel–ion binding resolution built on top of the
    discretization layer.

- **Declarative Mechanism System** (`braincell.mech`) (#69)
  - Hashable, order-insensitive `Density` / `Point` declarations and a
    mechanism registry (`@register_channel`, `@register_ion`,
    `@register_synapse`).
  - Point mechanisms: current/sine/function clamps, probes, synapses, and
    gap junctions.

- **Morphology IO** (`braincell.io`) (#68, #69)
  - Readers for **SWC** and **ASC**.
  - Correction: these notes originally also listed a **NeuroML2** reader.
    `NeuroMlReader` is only a stub whose `read()` raises
    `NotImplementedError`, in 0.1.0 and in 0.2.0; no NeuroML2 reading is
    available.
  - Three-tier **NeuroMorpho.Org** client with on-disk caching and a
    `braincell-neuromorpho` CLI.

- **Location/Region Filters** (`braincell.filter`) (#69)
  - Locset and region selection expressions (`ForkPoints`, `Terminals`,
    `UniformSamples`, `SubtreeRegion`, `BranchRangeFilter`, …) with selection
    caching.
  - Ordered, duplicate-preserving locsets with stable set algebra, explicit
    `unique()`, and read-only columnar `LocsetMask` storage. `BranchPoints`
    remains a compatibility alias for `ForkPoints`.

- **Visualization Stack** (`braincell.vis`) (#80, #82, #102)
  - 2D (matplotlib) and 3D (PyVista, Plotly) backends with a unified backend
    chooser.
  - 2D tree-layout engine, morphometry plots (dendrogram, Sholl, topology,
    branch-order histogram), trace panels, movies, and morphology/value
    comparators.

- **Cerebellum Dynamics** (#93)
  - Additional ion/channel dynamics and a Purkinje-cell MA2024 comparison
    scaffold.

### Breaking Changes

- **Removed the direct external-current injection path** for multi-compartment
  cells. The `Cell.update(I_ext)` path is gone; inject external current with
  placed point clamps instead — `CurrentClamp(...)`, `SineClamp`, or
  `FunctionClamp`.
- **Renamed the discretization package** `_cv` → `_discretization`, and
  `PointTree` → `NodeTree`, for clearer terminology (#88).

### Changes & Improvements

- Restructured the single-compartment module and import surface (#71).
- Refreshed the channel/ion public API and added deprecation aliases for the
  previous channel names (#97).
- Hardened multi-compartment dtype boundaries and runtime caching, including
  mixed-ion runtime fixes (#73, #82).
- Rendered the PyVista HTML backend as a static iframe for reliable notebook
  and docs embedding (#102).

### Bug Fixes

- Used `default_factory` for `brainunit.Quantity` dataclass field defaults to
  avoid shared mutable defaults (#92).

### Removed

- Dropped the `diffrax` dependency from the `quad` integrator stack (#94).

### Documentation

- Rebuilt the documentation around a layered, Arbor-inspired architecture;
  consolidated tutorials into runnable notebooks and expanded single-compartment
  examples (#101, #103, #104, #105).
- Added a top-level Numerical Integration tutorial and documented previously
  missing public APIs (#98, #99).
- Self-hosted the documentation at <https://brainx.chaobrain.com/braincell/>.

### Packaging & Tooling

- Marked the package as typed per **PEP 561** (added `py.typed`), so downstream
  type checkers consume BrainCell's inline annotations.
- Numerous CI workflow and dependency updates: deploy docs on release, GitHub
  Actions version bumps, and `brainx-sphinx-header` upgrades.

## Version 0.0.8

This version was tagged and published to PyPI on 2026-03-11 (tag `v0.0.8`) but
shipped without a changelog section. The entries below were written afterwards
from the diff between `v0.0.7` and `v0.0.8`.

### Breaking Changes

- BrainCell requires Python >= 3.11 (it was >= 3.10), and the Python 3.10
  classifier is dropped.
- The minimum `brainevent` is 0.0.7 (it was 0.0.2).

### Changes & Improvements

- The `cpu`, `cuda12`, `cuda13` and `tpu` extras now install only the matching
  `jax[...]` build; the BrainX packages come from the core dependencies.
- Removed the unused numba, warp and pallas kernel generators and the
  `comp_triang_call*` / `comp_backsub_call*` helpers from the voltage solver
  (about 340 lines; internal, not exported).
- Bumped `actions/upload-artifact` and `actions/download-artifact` in the
  Publish workflow.


## Version 0.0.7

This release focuses on structural refactoring to improve codebase organization, specifically grouping morphology and integrator components into dedicated sub-packages (`braincell.morph` and `braincell.quad`).

### Refactoring & Code Organization

- **Morphology Sub-package** (`braincell.morph`)
  - Moved and renamed morphology-related modules into `braincell/morph/`:
    - `_morphology.py` -> `morph/_morphology.py`
    - `_morphology_branch_tree.py` -> `morph/_branch_tree.py`
    - `_morphology_from_asc.py` -> `morph/_from_asc.py`
    - `_morphology_from_swc.py` -> `morph/_from_swc.py`
    - `_morphology_utils.py` -> `morph/_utils.py`

- **Integrator Sub-package** (`braincell.quad`)
  - Moved all integrator and solver modules into `braincell/quad/`:
    - `_integrator*.py` files moved to `braincell/quad/`.
  - This improves the clarity of the top-level namespace.

### Documentation

- **Structure Updates**
  - Updated API documentation to reflect the new module structure.
  - Simplified `index.rst` and reorganized API reference pages.
  - Updated copyright to reflect membership in the BrainX Ecosystem.

### CI/CD

- **Workflow Updates**
  - Bumped versions for `actions/checkout`, `upload-artifact`, and `download-artifact`.

## Version 0.0.6

This release focuses on major dependency updates, code modernization, and extensive refactoring to improve compatibility with the latest BrainPy ecosystem.

### Breaking Changes

- **Dependency Version Updates**
  - Updated `brainstate` from `>=0.1.0` to `>=0.2.0`
  - Updated `brainpy` from `>=3.0.0` to `>=2.7.0`
  - These updates may require users to upgrade their BrainPy ecosystem packages

### Refactoring & Code Improvements

- **Core Architecture Simplification** (2acd212)
  - Refactored `HHTypedNeuron` to use `brainpy` directly for better integration
  - Simplified `_base.py` with significant code reduction (221 insertions, 282 deletions)
  - Removed deprecated `_integrator_diffrax.py` module (29 lines removed)
  - Streamlined integrator implementations in `_integrator_runge_kutta.py`
  - Cleaned up `_single_compartment.py` and integration protocol

- **Parameter Initialization Migration** (fa71171, a79c306, 18b053c, 77a11ac)
  - Migrated parameter initialization from `brainstate.nn` to `braintools` across the entire codebase
  - Updated parameter initialization in ion channels (calcium, potassium, sodium, hyperpolarization-activated)
  - Refactored parameter initialization in synapse models (markov)
  - Updated HTC and EINet classes to use `braintools`
  - Updated all example scripts and notebooks to use `braintools` for parameter initialization

- **API Migration** (e84351a, bf50e6e)
  - Migrated from `brainstate.nn` to `brainpy.state` and `braintools`
  - Fixed `_base` errors in brainpy integration
  - Updated `CurrentProj` references across the codebase

### Documentation

- **Updated Documentation** (#54, 2acd212)
  - Updated braincell logo image
  - Refreshed tutorial notebooks (cell, channel, ion tutorials in both English and Chinese)
  - Updated advanced tutorial examples (sc02-sc05 notebooks)
  - Revised quickstart concepts documentation
  - Updated all documentation to reflect API changes and new parameter initialization patterns

### Examples

- **Example Updates**
  - Updated all example scripts to use new APIs:
    - `SC01_fitting_a_hh_neuron.py`
    - `SC03_COBA_HH_2007_braincell.py`
    - `SC05_thalamus_single_compartment_neurons.py`
    - `SC06_unified_thalamus_model.py`
    - `SC07_Straital_beta_oscillation_2011.py`
    - `MC11_simple_dendrite_model.py`
    - `MC13_golgi_model/` simulations

### CI/CD

- **Publishing Workflow Enhancement** (2acd212)
  - Updated `.github/workflows/Publish.yml` with improved configuration

### Code Statistics

- Overall changes: 48 files changed, 1,307 insertions(+), 1,408 deletions(-)
- Net reduction of ~100 lines while improving code quality and maintainability

## Version 0.0.5

This release brings significant performance improvements, new integration methods, enhanced morphology support, expanded documentation, and modernized packaging infrastructure.

### New Features

- **Pallas Kernel Acceleration** (#51)
  - Added Pallas kernel support for voltage solver to accelerate multi-compartment simulations
  - Introduced optimized triangular matrix computation with GPU/CPU backend support
  - Added debug kernels for Pallas backend testing

- **Backward Euler Solver** (#49)
  - Added backward Euler integration method for improved numerical stability
  - Enhanced integration infrastructure with new solver options

- **Morphology Enhancements** (#41, #46, #51)
  - Added support for immutable sections
  - Implemented DHS (Diagonal Hines Solver) support
  - Added lazy loading of networkx for better performance
  - Improved morphology branch tree handling and documentation
  - Enhanced ASC/SWC file support for morphology loading

### Performance Improvements

- **Sodium Channel Integration** (da6697f, 7f91bbe, 7c218f1)
  - Refactored sodium integration from backward Euler to RK4 solver for better accuracy
  - Updated population size handling in simulations
  - Optimized voltage solver performance

- **Integration System Refactoring** (#47)
  - Refactored integrators to get time from `brainstate.environ` for better consistency
  - Streamlined solver logic and improved code structure

### Documentation

- **Expanded Chinese Documentation** (#45)
  - Added comprehensive Chinese language documentation
  - Included advanced tutorial examples and API references

- **New Documentation Structure** (#40, #42)
  - Added quickstart guides, tutorials, and advanced tutorials
  - Reorganized documentation for better navigation
  - Enhanced code documentation and type hints (#44)

### Infrastructure & Dependencies

- **Packaging Modernization**
  - Migrated from `setup.py` to modern `pyproject.toml`-only configuration
  - Updated license format to SPDX identifier (`Apache-2.0`)
  - Improved package metadata and dependency specifications

- **Dependencies**
  - Added `brainpy>=3.0.0` as core dependency
  - Added `braintools>=0.1.0` for enhanced tooling
  - Updated CI/CD configurations for Python 3.13 support

- **CI/CD Updates**
  - Added Python 3.13 support (#50, #48)
  - Updated GitHub Actions: setup-python from 5 to 6, checkout from 4 to 5

### Code Quality

- **Refactoring & Improvements** (#44)
  - Improved external current registration and error handling
  - Enhanced type hints across the codebase
  - Better code organization and readability

### Examples & Testing

- Added linear solver test notebooks
- Enhanced Golgi model simulation examples
- Updated example scripts for better demonstration of features

## Version 0.0.4

Previous release with core functionality.

## Version 0.0.1

The first release of the project.
