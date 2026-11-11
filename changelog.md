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

- **`Morphology.reattach(target, *, parent, parent_x=1.0, child_x=None)`
  moves a subtree within a tree** (MORPH-2e). `target` and all its
  descendants are cut from their parent and attached to `parent`; names,
  geometry and every attachment inside the moved subtree are kept, and
  `child_x=None` keeps the end `target` attached by. A branch's parent never
  changes, so the move is a removal plus an inserted copy made as one edit:
  the moved branches get new views (the views held before detach) and come
  last in default order. Cell inputs held by branch index are therefore
  rejected at the Cell's next rebuild, as after `remove`, and location
  expressions follow the move. Moving the root, or moving a branch under
  itself or one of its descendants (a cycle), raises `ValueError`. To insert
  a branch between two others, `attach` it and then `reattach` the lower part
  under it. 3-D points do not move. `reattach` is now a reserved branch
  name: a checkpoint that holds a branch named `reattach` no longer loads. No
  branch in this repository uses that name. The morphology tutorial has a new
  section, "Editing a tree", that walks through `clone`, `remove`,
  `replace`, `graft` and `reattach`.

- **`Morphology.graft(other, *, parent, parent_x=1.0, child_x=0.0,
  rename=None)` attaches a copy of another tree** (MORPH-2d). `other`'s root
  becomes a child of `parent`, attached as `attach` would attach it, and its
  other branches follow with their own attachment points; the receiving tree
  keeps its root. To graft part of a tree, pass `other.subtree(name)`; to
  make `other` the host, call `other.graft(tree, ...)`. `other` is copied,
  never changed or taken over: the new branches share its immutable `Branch`
  geometry, it may be frozen, and it may be the receiving tree itself. A
  name that already exists raises `ValueError` listing every clash, before
  anything is written; `rename=` resolves it, as a mapping (`{"d1": "d1b"}`)
  or a function (`lambda name: f"{name}_copy"`). Existing branches keep
  their default indices, so Cell inputs held by branch index stay valid, and
  the auto-naming counters do not change. 3-D continuity at the joint is not
  checked; `to_swc` reports a gap. Grafting the 681-branch CA1 reconstruction
  onto a one-branch tree takes about 1 ms. `graft` is now a reserved branch
  name: a checkpoint that holds a branch named `graft` no longer loads. No
  branch in this repository uses that name.

- **`Morphology.replace(target, new_branch)` swaps one branch's geometry in
  place** (MORPH-2c). The new `Branch` may have another length, number of
  segments or type. The branch keeps its name, its view (`replace` returns the
  same object), its parent and its attachment point, and its children stay
  attached at the same end (`parent_x` is a normalized position), so on tree
  `soma -> dend -> d1, d2` a longer `dend` moves the path distance of `d1`
  and `d2` with it. No default index changes, so Cell inputs held by branch
  index stay valid; a new type can change the type order. No other branch's
  3-D points move: when the new geometry ends elsewhere, the cable model is
  unaffected, but `to_swc` raises `ValueError` naming the first child that no
  longer connects. A `MorphoBranch` as `new_branch` raises `ValueError` (pass
  `view.branch`), any other non-`Branch` raises `TypeError`, and a non-soma
  type for a branch with children at `parent_x=0.5` raises `ValueError`
  listing them. Assigning to an existing name (`morpho.soma.dend = branch`)
  still raises. `replace` is now a reserved branch name: a checkpoint that
  holds a branch named `replace` no longer loads. No branch in this
  repository uses that name.

- **`Morphology.remove(target)` removes a branch and all its descendants**
  (MORPH-2b). It returns the removed part as a new tree, equal to what
  `subtree(target)` returned before the call, so it can be inspected or
  attached elsewhere. `target` is a branch name or a view of a branch in the
  tree. The root cannot be removed (`ValueError`; copy the part to keep with
  `subtree(name)`), and a frozen tree raises `RuntimeError`; a call that
  raises changes nothing. Views of the removed branches are detached: their
  name and geometry stay readable, while `index`, `parent`, `children` and
  attaching to them raise `RuntimeError`. Branches created after the removed
  ones move down in default order, so a Cell input held by branch index (a
  resolved `LocsetMask` given to `place`, a `CVPerBranchList` policy, a
  `VoltageCrossingSource` at a `LocsetMask`) raises at the Cell's next
  rebuild; a location expression such as `at("axon", 0.5)` is evaluated again
  and follows the edit. The removed names are free again: a branch attached
  under one of them is a new branch, and a held view of the old one stays
  detached. The auto-naming counters do not move back. `del
  morpho.soma.dend` still raises `AttributeError`, and now says to call
  `Morphology.remove("dend")`. `remove` is now a reserved branch name: a
  checkpoint that holds a branch named `remove` no longer loads. No branch in
  this repository uses that name.

- **A lumped cell has the views, trainables, synapses and `Network` of a
  cable cell** (CELL-24). On `Cell.point(...)` / `cv_policy=Lumped()`,
  everything that raised `NotImplementedError` now works and gives what the
  same declarations give on the one-CV twin
  `Cell(Morphology.from_soma(...), cv_policy=CVPerBranch())` (float64,
  `rk4`: voltages within 1e-9 mV, the same spike events, first-order
  gradients within 1e-8 relative):
  - views: `cell.channels`, `cell.ions`, `cell.synapses`, `cell.geometry`,
    `cell[...]`, `cell.on(...)`, `cell.loc(...)`, `cell.soma`, `cell.cv[...]`,
    `cell.branch[...]`, read and `set`, including ion initializers
    (`Ci_initializer`) and per-member values; a view drops the twin's one-CV
    axis;
  - `trainable(...)` on density and geometry fields and on the spike
    threshold, with gradients, under `brainstate.transform.grad` and the
    experimental rollout engine (which keeps its `"rollout"` schedule);
  - every point mechanism on the one site: synapses (also on a population
    view, `cell[1:].place(syn)`, and synapses that read an ion, such as
    `GabaA_JE2011`), the deprecated probes and `Junction`; the block solvers
    `exp_euler`, `backward_euler` and `crank_nicolson` reject point synapses
    at `init_state`, as on a cable cell;
  - every `braincell.observe` recording, `membrane_current()` included, and
    `add_reduction` / `use_model`;
  - `Network.add_population`: a lumped cell is a spike source
    (`event_outputs["spike"]`, `VoltageCrossingSource`) and a synapse target,
    also next to cable populations.

  A kinetic ion (e.g. `Cldif_JE2011`) advances before the solver on a lumped
  cell, as on a `SingleCompartment`, and after it on a cable cell, so its
  recorded concentration leads the twin's by one step. `cell.runtime` on a
  lumped cell is the one-CV layout state its mechanisms run on; `n_point` and
  `layouts` still raise `RuntimeError`.

- **`Morphology.clone()` and `Morphology.subtree(target)`** (MORPH-2a).
  `tree.clone()` returns an independent copy that compares equal to `tree`
  and has the same auto-naming counters (`naming_state()`).
  `tree.subtree("dend")` returns a new tree rooted at `dend` with all its
  descendants; names and attachment points inside it are unchanged, and
  `dend`'s own attachment to its parent is dropped. `target` is a branch
  name or a view of a branch in the tree. Both copies share the immutable
  `Branch` geometry and have their own views and `revision`, so editing one
  tree leaves the other unchanged. Both only read the source, which may be
  frozen; a copy is never frozen. Because the geometry is shared, `clone()`
  is about 15 times faster than `copy.deepcopy`: 1.5 ms against 23 ms on a
  681-branch reconstruction. `clone` and `subtree` are now reserved branch
  names, like every public `Morphology` attribute:
  `attach(child_name="clone")` raises `ValueError`, and a checkpoint that
  holds a branch of either name no longer loads. No branch in this
  repository uses either name.

- **External input for lumped cells: `update(I_ext)` and brainpy projections**
  (CELL-23). A lumped cell (`Cell.point(...)`, `cv_policy=Lumped()`) can be
  driven from a user loop and by `brainpy.state` projections, with the same
  numbers as the equivalent `SingleCompartment` (float64, 800 steps, `rk4`,
  `euler`, `ralston2`, `exp_euler`: at most 3e-12 mV apart).
  - `cell.update(I_ext)` takes a current (`0.1 * u.nA`, divided by the
    membrane area) or a current density (`10 * u.uA / u.cm**2`), broadcast
    to `(batch,) + pop_size`. It is added to the step's clamp current and
    held over every stage of the solver step; `cell(I_ext)` does the same
    after the `before_updates`. `update()` without input is unchanged.
  - `AlignPostProj` (with `COBA` or another output) and `CurrentProj` reach
    the cell through `add_current_input`; the output may be a current (a
    conductance in nS) or a density (mS/cm^2), and is evaluated at every
    stage with the stage voltage, as on a `SingleCompartment`. A lumped cell
    used to ignore these inputs silently. They also apply under `run()` and
    a directly called solver step.
  - Errors at `update`: `UnitMismatchError` for any other unit (or a bare
    number), naming the input for a registered one; `ValueError` for a
    shape that does not broadcast. A cable cell raises `TypeError` for
    `update(I_ext)` and points to `CurrentClamp` / `FunctionClamp`, as it
    never took an input.

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
    `place` raises `ValueError`.
  - `Lumped()` on a morphology with more than one branch raises `ValueError`
    when the cell is built or the policy assigned.
  - Initial gates are at steady state for `V_init` (as on cable cells;
    `SingleCompartment.init_state()` leaves them at 0). `V_init=None` starts
    at the resting potential.
  - Views, trainable parameters, synapses, `add_reduction` and
    `Network.add_population` work on a lumped cell (CELL-24, above).
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

- **Chloride ions** (MECH-30). `braincell.ion` gains the `Chloride` family
  (family key `"cl"`, valence -1, default `Ci` 5 mM and `Co` 133.5 mM),
  `ChlorideFixed` (`E = -87.5 mV`), `ChlorideInitNernst` (36 °C) and
  `Cldif_JE2011`, the four-shell radial chloride pool of Jedlicka et al.
  (2011), ModelDB 148253 (`cldifus`), driven by the chloride current (an
  inward current lowers `Ci`); its relaxation to `cli0` is explicit, as
  NMODL's `<<` flux is. It matches NEURON on one segment; the `.mod`'s
  diffusion along the cable is not ported. `cl` is not a placeholder ion: a
  chloride channel or synapse needs a declared chloride ion.

- **Chloride leak channel `Cl_Leak`** (MECH-31). `braincell.channel.Cl_Leak`
  is a passive chloride conductance, `I = g_max (E_Cl - V)` with `g_max`
  0.1 mS/cm² by default; `E_Cl` comes from the attached chloride ion and the
  current is that ion's, so it drives `Cldif_JE2011`. A tonic GABA_A current
  with bicarbonate share `P` is `Cl_Leak(g_max=(1 - P) g_t)` plus
  `IL(g_max=P g_t, E=E_HCO3)`. It matches NEURON (a `clleak` testing
  mechanism) with a fixed-concentration chloride ion and with `Cldif_JE2011`
  under current clamp.

- **GABA_A synapse that reads and writes chloride, `GabaA_JE2011`** (MECH-32).
  `braincell.synapse.GabaA_JE2011` (Jedlicka et al. 2011, `gabaA_Cl.mod`) is
  an `Exp2Syn`-shaped conductance split between chloride (`1 - P`) and
  bicarbonate (`P`, default 0.18); its `E_Cl` comes from the Cell's chloride
  ion on the synapse's CV, and its chloride current goes into that ion, so a
  `Cldif_JE2011` pool accumulates chloride and `E_GABA` rises with activity.
  It matches NEURON (`gaba` + `cldifus`) on one segment. New, optional
  `Synapse` API: a class attribute `ion_type` (default `None`) and a method
  `ion_current(V_post, ion_info)` (default `None`); a synapse with
  `ion_type` set implements `current(V_post, ion_info)`. Synapses without
  `ion_type` (`ExpSyn`, `Exp2Syn`, user subclasses) are unchanged. A Cell
  with such a synapse writing into a dynamic ion raises `ValueError` in
  `init_state` when `cache_ion_total_current=False`.

- **Dynamic sodium and potassium, `SodiumDetailed`, `PotassiumDetailed`, and
  the sodium leak `Na_Leak`** (MECH-35). The two ions make `Ci` a state,
  `dCi/dt = I/(zFd) + beta (C_rest - Ci)` with `I` the family's inward
  current and `E` from Nernst; the influx is not rectified, so an outward
  current lowers `Ci`. Defaults: `d` 1 µm, `beta` 0 (pure accumulation), and
  the family concentrations (Na 10 / 140 mM, K 54.4 / 2.5 mM); for the whole
  compartment volume pass `d=lambda ctx: ctx.diam_mid / 4`.
  `braincell.channel.Na_Leak` is `I = g_max (E_Na - V)` (`g_max` 0.005
  mS/cm²) with `E_Na` from the attached sodium ion, like `K_Leak`. A
  Canavier (1999) soma with the pump off matches NEURON (`nadifl.mod` and
  `leak.mod`, ModelDB 6763).

- **Dynamic extracellular potassium, `PotassiumDetailedFH`** (MECH-37). Like
  `PotassiumDetailed`, with `Co` a state too: the Frankenhaeuser-Hodgkin
  space, a thin layer of finite volume outside the membrane of each
  compartment, `dCo/dt = -I/(zF d_o) + beta_o (Co_rest - Co)`, so `E_K`
  follows both concentrations. Parameters `d`, `d_o`, `beta`, `beta_o`,
  `C_rest`, `Co_rest`, `Ci_initializer` and `Co_initializer` (no parameter is
  named `Co`, which is the state); record it with
  `observe.ion(name).state("Co")`. The space is not shared between
  compartments or cells. Botta et al.'s (2010) `K_conc.mod` (ModelDB 127021)
  is this model with `d = d_o` twice the `.mod` file's `d` (it writes the
  influx as `ik / (2 FARADAY d)`); against NEURON the pools differ by
  `e_V = 4.7e-5 mV` over 200 ms, from NEURON's implicit Euler update.

- **Na⁺/K⁺ pump `braincell.channel.NaKPump_Ca1999`** (MECH-36), after
  Canavier (1999) `pump.mod`: `I_p = imax / (1 + (km / [Na⁺]i)^n)`, net
  outward, three sodium out and two potassium in. The sodium ion receives
  `-3 I_p`, the potassium ion `+2 I_p` (inward positive), so a
  `SodiumDetailed` pool falls and a `PotassiumDetailed` pool rises; with fixed
  ions the pump adds the membrane current `-I_p` only. Defaults are the
  `.mod`'s (`imax` 0.04 mA/cm², `km` 10 mM, `n` 1.5; the published soma uses
  `imax` 0.012). It needs both a sodium and a potassium ion where it is
  painted. A Canavier soma with the pump matches NEURON (`nadifl.mod`,
  `pump.mod` and `leak.mod`, ModelDB 6763).

### Changes & Improvements

- **A Cell declaration whose branch name is gone says which declaration it
  is** (MORPH-2b). A location expression names its branch, such as
  `at("dend", 0.5)` given to `Cell.place` or to `VoltageCrossingSource`, and
  is evaluated again at every rebuild. Once `dend` was removed, the next grid
  read raised a bare `KeyError: 'dend'`. It still raises `KeyError`, now
  reading `Cell.place(...) at <file>:<line> locates its mechanisms on branch
  'dend', which the morphology (revision N) does not have: ...` (or
  `VoltageCrossingSource location ... names branch 'dend' ...`), with what to
  do next. A misspelt name at `place` time gets the same message. Attaching a
  new branch named `dend` makes the expression follow it.

- **Deprecation warnings name their releases and point at the calling line**
  (API-7). Each braincell deprecation now reads `<name> is deprecated since
  braincell <since> and will be removed in braincell <removed_in>. Use
  <replacement> instead.` and is attributed to the line that used the
  deprecated API, also through a subclass `__init__`, brainstate's node
  metaclass or a dataclass, so a script or notebook shows it by default.
  braincell's own uses no longer warn: an unnamed probe placed on a Cell
  warned a second time, from `dataclasses.py`, when the Cell named it, and now
  warns once. Warning categories are unchanged; silence them all with
  `warnings.filterwarnings("ignore", message=".* is deprecated since
  braincell", category=DeprecationWarning)`.

- **A lumped Cell warns at `init_state` when its explicit solver meets a
  stiff HH gate** (CELL-29). An explicit Runge-Kutta solver (the default
  `ralston2` included) integrates the gates with `V`, and is stable on a
  gate only while `dt` times the gate's rate stays below about 2. The m gate
  of `Cav1p2_MA2020_GoC` relaxes at 2.27e6 /ms at -65 mV, so a lumped Cell
  with it returned NaN at every `dt` from 0.025 down to 1e-6 ms without a
  word. `init_state` now scans each HH gate's rate between -100 and 60 mV
  and, where `0.025 ms * rate > 2`, raises a `UserWarning` naming the
  channel, the gate, its largest rate and the `dt` the solver would need,
  and suggesting `solver="exp_euler"` (second order on HH cells, about 20
  times the cost per step) or `"ind_exp_euler"` (first order, the cost of
  `ralston2`). In the catalogue this flags `Cav1p2_*`, `Cav1p3_*`, `HCN1_*`
  and `Na_ZH2019_IO`. The default solver and all results are unchanged.

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
  The `Kca1p1_*` docstrings add the same guidance with their own numbers
  (MECH-23a): about 2 % at `dt = 0.025 ms` on `backward_euler`, 0.6 % with
  four substeps, exact on `exp_euler`. So do the eight sodium Markov
  channels (MECH-23c): `Nav1p6_*` about 8 %, `Nav1p1_*` 7 %,
  `Nav_MA2020_GrC` 14 % and `NaFHF_MA2020_GrC` 20 % on `backward_euler`,
  2-5 % with four substeps, exact on `exp_euler`.

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

- The `AttributeError` for the removed `braincell.quad.rk2_step` no longer
  names it twice. It now reads: "braincell.quad.rk2_step was removed; it was
  Ralston's second-order method; use braincell.quad.ralston2_step (same
  numbers), or heun2_step for Heun's method."

- A `Cell.run` that continues an earlier `run` or `update()` no longer
  delivers the events on its first step twice (CELL-34). Each step prepares
  the synaptic input of the next boundary, and `run` prepared its first
  boundary again on top of it. Splitting a run at an event time doubled that
  event: an ExpSyn (tau 2 ms) driven by `NetStim(start=1 ms, number=1)` at
  0.005 uS had g = 0.006065 uS at 2 ms after two 1 ms runs, instead of the
  0.003033 uS of one 2 ms run. Bound synapse inputs and reduced cells were
  affected in the same way; every solver was. `run` now prepares its first
  boundary only when no step has run since `init_state` or `reset_state`, and a
  split run matches one run to 1e-12. One run is bit-identical to before.
  `Network.run` was not affected.

- The first `Cell.update()` after `init_state()` or `reset_state()` now
  delivers the events at the start time, as `Cell.run` does (CELL-35). `run`
  prepares the synaptic input of its first boundary when no step has, but
  `update()` only prepared the boundary after its step, so an `update()` loop
  never delivered an event at t = 0: one 600 um x 0.5 um CV with an ExpSyn
  (tau 2 ms) driven by `NetStim(start=0 ms, number=1)` at 0.005 uS stayed at
  g = 0 and -65 mV, where `run` gives g = 0.001839 uS and -56.89 mV at 2 ms.
  Bound synapse inputs missed their first boundary in the same way, and so
  did the synapses of a lumped Cell (`Cell.point`).
  `update()` now prepares that boundary under the same condition as `run`, so
  an `update()` loop matches `run` to 1e-12 and neither delivers an event
  twice when one continues the other. Loops whose events all fall after the
  first step are bit-identical to before.

- A population-view `place` (`cell[[0]].place(...)`, or
  `cell[[0]].loc(mask).place(...)`) that fails now leaves the Cell as it was,
  as `Cell.place` and per-member placement already did. Before, the failed
  rule stayed, and every later grid read (`n_cv`, `init_state`) raised the
  same error again. A `LocsetMask` or `LocsetBatch` that names a branch the
  morphology does not have now raises an `IndexError` naming that branch and
  the valid range, from `place` and `cell.loc`. Before it was a bare
  `IndexError: tuple index out of range` from deep in the rebuild.

- Cell inputs given as resolved branch indices no longer land silently on
  other branches after an edit shifts those indices (MORPH-16). There are
  three such inputs:
  - a `LocsetMask` or `LocsetBatch` passed to `Cell.place` (directly, per
    member, or through `cell.loc(mask)` or a population view);
  - a `CVPerBranchList` policy, alone or inside a `CompositeByTypePolicy`;
  - a `VoltageCrossingSource` whose `location` is a `LocsetMask`.

  Each records the morphology revision it was declared at. Removing or moving
  a branch moves later default indices down. After that the next
  rebuild raises `RuntimeError`, and the message names the declaration. The
  rebuild is the first grid read, `init_state`, or the detector's next
  `cv_id`. Before, on a soma with `dend`, `axon` and `apical` (indices 1, 2
  and 3), removing `dend` left a synapse placed with
  `at("axon", 0.5).evaluate(m)` on `apical`. It also pointed the detector at
  apical's CV, and `CVPerBranchList((1, 2, 1, 1))` handed dend's two CVs to
  the axon once a fourth branch was attached. Appending branches, and
  replacing or transforming a branch in place, keep every existing index, so
  these declarations stay valid. A location expression such as
  `at("axon", 0.5)` is evaluated again at each rebuild and follows any edit.
  Unedited morphologies behave as before. `Morphology.remove` and
  `Morphology.reattach` are the public edits that move them.

- A current input registered with `Cell.add_current_input` after a
  `Cell.run` is now included by the next `run` (CELL-23). `run` reuses its
  compiled loop for the same `dt`, step count and recordings, and the loop
  had traced the inputs present at the first run: on a one-CV cable cell
  (`IL` 0.3 mS/cm^2, E -65 mV) a 10 uA/cm^2 input added between two 1 ms
  runs left V at -65 mV. Registering an input now drops the compiled loops.

- Clamps and synapses on a branch end or a branch point now act on the
  membrane under every solver other than `"staggered"` (CELL-2a). The
  explicit Runge-Kutta solvers (`"rk4"`, `"ralston2"`, `"dopri5"`, ...),
  `"exp_euler"`, `"ind_exp_euler"`, `"backward_euler"` and `"crank_nicolson"`
  eliminated these points (they have no membrane capacitance) as if nothing
  were placed there, so the inputs were lost: one 600 um dendrite in 3 CVs with
  a 0.05 nA clamp and an ExpSyn at x = 0 stayed at -65 mV, where the cable
  equation gives -35.73 mV at the first CV after 5 ms. `Cell.compute_derivative()`
  now solves each such point's row at every evaluation (three Newton steps) and
  adds the current it passes to the neighbouring CVs; the explicit solvers
  converge at their order (rk4 error 3.6e-9 mV at dt = 0.025 ms), current is
  conserved, and gradients reach the input (d sum V / d g was 0). Models with
  nothing placed on a branch end or branch point give bitwise the same results
  as before. The solved voltage at those points is also what mechanisms,
  events and records read under these solvers: a synapse current recorded with
  `observe.synapse(...).current()` at a branch end was evaluated at 0 mV
  (0 nA for an ExpSyn with `e = 0 mV`) and now uses the point's voltage, and
  the deprecated `StateProbe` at a point carrying a clamp or synapse returns
  that voltage instead of its CV's (as it already did under `"staggered"`). Network and direct event
  delivery to a synapse at a branch end also passes that voltage instead of
  0 mV, under every solver; the built-in synapses do not read it.

- A synapse on a branch end or a branch point now reads, in its
  `compute_derivative`, the voltage each Runge-Kutta stage solves for that
  point (CELL-2b). The solvers built on `Cell.compute_derivative()` (the
  explicit Runge-Kutta solvers, `"ind_exp_euler"`, ...) handed it the voltage
  solved at the step start, so a synapse whose dynamics depend on the voltage
  lost the solver's order there. An ExpSyn whose decay time grows as
  exp((V + 65 mV) / 20 mV), at x = 0 of a 600 um dendrite in 3 CVs, was off
  by 1.3e-2 mV after 5 ms with `"rk4"` at dt = 0.025 ms (first order); it is
  now off by 6.2e-9 mV (fourth order), and `"staggered"` converges to the same
  values. Each stage solves the point once and hands the same voltage to the
  cable equation and to the synapse. The built-in synapses do not read the
  voltage in `compute_derivative`: their results, and those of models with
  nothing on a branch end or branch point, are bitwise unchanged.

- A solver step called directly on a `Cell` now samples the current clamps
  as `update()` does (CELL-2c): once per main step, at `t + dt / 2`, with every
  Runge-Kutta stage reading that value. Before, `braincell.quad.rk4_step(cell)`
  (and the other explicit Runge-Kutta steps) re-sampled the clamps at each
  stage time on a cable Cell, so a clamp switching on at 0.02 ms already
  raised the middle CV by 0.164 mV in the first 0.025 ms step, where
  `update()` keeps it at -65 mV; on a lumped Cell every solver step ignored
  the clamps (V stayed at -65 mV instead of -64.90 mV). `Cell.pre_integral()`
  now prepares the clamp current when the step was not prepared by
  `update()`, `run()` or `Network.run`; those paths are unchanged.

- A `Morphology.attach` without `child_name` that fails validation no longer
  consumes an auto-name suffix (MORPH-11). Before, on a soma-dend tree,
  `m.attach(parent="dend", child_branch=axon, parent_x=0.5)` raised
  `ValueError` but still moved `naming_state()` from `{}` to `{'axon': 1}`, so
  the next auto-named axon was `axon_1` instead of `axon_0`. Every edit now checks
  all its arguments and picks the name before writing anything, so a failed
  `attach` (or `morpho.soma.dend = branch`, or `morpho.soma[0.5].dend =
  branch`) leaves the tree, `revision` and `naming_state()` unchanged, and a
  successful one advances `revision` by exactly one. Assigning a private
  attribute of a branch view (`m.dend._branch = new_branch`) now raises
  `AttributeError`: it used to swap the geometry without advancing
  `revision`, so `total_length`, `x_range`, `to_vis_data()`, a
  `SelectionCache` and a Cell's discretization kept returning results for the
  old geometry.

- Every solver name a cable `Cell` accepts now runs, or is rejected at
  `init_state` (CELL-19). Before, `solver="exp_euler"`, `"ind_exp_euler"`,
  `"backward_euler"` and `"crank_nicolson"` raised `ValueError: State
  RuntimeParameterState(...) is not in the state list.` in `run` on every
  cable Cell (the steps rejected the geometry they read, which the Cell keeps
  outside its state graph), and the explicit Runge-Kutta solvers (`"ralston2"`,
  `"rk4"`, ...) raised `ValueError: State ... must be written by
  compute_derivative().` once the Cell had a Markov channel or a kinetic ion.
  Now a generic solver integrates `V` with the dependent mechanism and synapse
  states, and the Markov channels and kinetic ions then advance at the new
  voltage through their subsolver, in the order `"staggered"` uses
  (`ion_channel_update_order`). On a one-CV soma with `Nav_MA2020_GrC` or a
  kinetic pool, `V` at 1 ms is within 2.3e-4 mV of `"staggered"` at
  `dt = 0.005 ms` (float64) and the gap shrinks 5-fold at `dt = 0.001 ms`.
  `"exp_euler"`, `"backward_euler"` and `"crank_nicolson"` solve all states of
  a cell as one block, which point-synapse states (no CV axis) do not fit: with
  a point synapse they now raise `ValueError` at `init_state` naming the solver
  and the synapse class, where `Network.run` raised `TypeError: Cannot
  concatenate arrays with different numbers of dimensions` or `ValueError:
  Cannot stack arrays ...`. A user callable still owns the whole step.
- A kinetic ion (e.g. `CdpStC_MA2020_GoC`) now runs on a lumped Cell under its
  default `"ralston2"` and every other explicit solver, and a Markov channel on
  a `SingleCompartment` under its default (CELL-19); both raised `ValueError:
  State ... must be written by compute_derivative()`. The explicit steps of
  `braincell.quad` now read a `DiffEqState` of an `IndependentIntegration`
  submodule as a constant of the step instead of rejecting it. The dependent
  channels of a kinetic ion (`CaHVA_MA2020_GoC` on the pool) are now
  integrated by the cell's solver on a `SingleCompartment`, a lumped Cell and
  a cable Cell under a generic solver: no hook stepped them, and the block
  solvers failed with `UnexpectedTracerError`.

- `NetStim(number=10**9)`, NEURON's idiom for an unlimited train, no longer
  exhausts memory (CELL-30). `NetStim` generated all `size * max(number)`
  event times at construction, and every run compiled the whole matrix into
  its loop: about 120 bytes per event, so a 200 ms single-synapse run took
  1.9 GB at `number=10**7` and tens of GB at `10**9`. A schedule above
  `2**20` entries is now generated on demand: `Cell.run` and `Network.run`
  generate events only as far as the end of the run, and a later run extends
  them. The same run at `number=10**9` peaks at 0.85 GB, as at `number=8`. The
  events are unchanged bit for bit, noisy ones included (the random numbers
  of the full draw are reproduced for the columns needed), and smaller
  schedules are generated as before. `NetStim.event_times`, `NetStim.events`,
  `event_count` outside a run, `Cell.run` under `jax.jit` and
  `Network.prepare_run` still generate every event.

- A lumped Cell (`cv_policy=Lumped()`, `Cell.point`) now checks the units of
  declared Channel and Ion parameters at `init_state`, with the same exception
  type and message as a cable Cell (MECH-41): the mechanism, the parameter, the
  expected unit, the value and the `paint` call site. Before, the lumped build
  passed the values to the runtime classes unchecked, so
  `Ion("SodiumDetailed", d=1.0)`, `Ion("SodiumFixed", E=50.0)` or
  `Channel("IL", g_max=0.1 * u.mV)` failed on the first step of `run` that
  used the value, with a `UnitMismatchError` that named no parameter. A
  braintools initializer for a Channel parameter is still accepted on a lumped
  Cell.

- A lumped Cell (`cv_policy=Lumped()`, `Cell.point`) with a declared
  `Ci_initializer` on a `DynamicNernstIon` (`CalciumDetailed`,
  `CalciumFirstOrder`, `CdpHVA_SU2015_DCN`, `CdpLVA_SU2015_DCN`) now runs
  (MECH-38). The declared value reached the ion as a bare scalar, `Ci` started
  with shape `()` instead of `pop_size`, and `run` raised `TypeError: scan body
  function carry input and carry output must have equal types` (or `ValueError:
  Cannot stack arrays` under `exp_euler`). A `DynamicNernstIon` now broadcasts
  its `Ci` initializer to the state shape on `init_state` and `reset_state`, as
  kinetic pools already did for their species, so the same declaration also
  works on a directly built `SingleCompartment` (`CalciumDetailed(1,
  Ci_initializer=2.4e-4 * u.mM)`). Results equal the undeclared default
  bit for bit.

- A `TypeError` raised inside a custom ion solver that accepts
  `excluded_paths` now propagates (MECH-27). A kinetic ion called its solver
  with `excluded_paths` and, on a `TypeError` naming that keyword, called it
  again without, so such an error (a unit error, say) was swallowed and the
  step ran twice. Whether the solver takes the keyword is now read from its
  signature (a parameter of that name or `**kwargs`); a solver whose
  signature cannot be read is called without it. The registered solvers are
  dispatched as before: `backward_euler` and `ind_exp_euler` get
  `excluded_paths`, the others do not.

- The `K_HH1952` docstring said its defaults do not reproduce Hodgkin and
  Huxley's 6.3 degC kinetics; they do (MECH-19b). `temp` and `temp_ref` both
  default to 36 degC, so the Q10 factor is 1 and the gate runs the published
  rates. With `temp_ref = 6.3 degC` the class matches NEURON's built-in `hh`
  at 36 degC to 1e-15 under voltage clamp, now part of the NEURON comparison.

- Five calcium channels no longer return NaN when the voltage sits exactly on
  a removable 0/0 point of their source expressions (MECH-23a): `CaHT_Re1993`
  at -27 mV (the `p` gate's rate, so its state, derivative and current),
  `CaHVA_SU2015_DCN` at -8.9 mV (`m` derivative) and at 0 mV (current),
  `CaLVA_SU2015_DCN` at 0 mV (current), and `Cav1p2_MA2020_GoC` and
  `Cav1p2_MA2025_BC` at -8.124 mV (`m` derivative). A voltage clamp at 0 mV
  or a ramp through it gave NaN currents for the two DCN channels. The rates
  and the DCN GHK drive now use `exprel`, which equals the source expression
  to 1e-13 relative at least 0.01 mV away from those voltages and takes the
  limit at them.

- `K_Kv_test` no longer returns NaN at `V - V_sh = v12` (25 mV by default),
  where both quotients of its `n` time constant are 0/0 (MECH-23b). Its `n`
  derivative was NaN there, so a voltage clamp at, or a ramp through, 25 mV
  gave NaN states and currents. The rate now uses `exprel`, which equals the
  previous expression to 1e-13 relative at least 0.01 mV away from `v12` and
  takes the limit `tau_n = 1 / (q (Ra + Rb))` there.

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

- The README, the documentation home page and the File Formats pages no
  longer say braincell reads NeuroML2 (API-8). `braincell.io.NeuroMlReader`
  has been a placeholder since 0.1.0: `read()` raises `NotImplementedError`
  for every file. The docs now call NeuroML2 import planned, and the
  NeuroML2 page no longer shows a loading example that cannot run. The error
  message, which read `Parse NeuroML2 into an editable Morpho tree.`, now
  says NeuroML2 import is not supported yet and points to
  `Morphology.from_swc` and `Morphology.from_asc`; the exception type is
  unchanged.

### Deprecations

- **`braincell.mech.StateProbe`, `braincell.mech.MechanismProbe`,
  `braincell.mech.CurrentProbe` and `braincell.mech.ProbeMechanism`** (API-7)
  are deprecated since 0.3; removal planned for 0.4. Use
  `Cell.record(..., braincell.observe.*)`, e.g.
  `cell.record("v", braincell.observe.state("V"))` for a `StateProbe`. They
  have warned since 0.2.0 without naming a release; the warning now does.
- **`braincell.Cell.bind_synapse_input()`** (API-7) is deprecated since 0.3;
  removal planned for 0.4. Use an `EventSource` with `braincell.connect()`.
  It has warned since 0.2.0 without naming a release; the warning now does.
- **`braincell.filter.density.exponential()` and
  `braincell.filter.density.gaussian()`** (API-7) are deprecated since 0.3;
  removal planned for 0.4. Use a callable built from `braincell.filter.metric`:
  `density.exponential("branch_x", 0.2)` is
  `lambda context: np.exp(metric.branch_x(context) / 0.2)`. They have warned
  since 0.2.0 without naming a release; the warning now does.

- **Constructing an invalid mechanism declaration** (MECH-26).
  `bc.mech.Channel`, `bc.mech.Ion` and `bc.mech.Synapse` now check literal
  parameters (quantities, numbers, arrays) and parameter names when the class
  is known at construction (a class object, or an already registered name),
  and emit a `FutureWarning` at the declaring line, e.g.
  `Channel 'leak' (IL) parameter 'g_max' has units incompatible with mS / cm^2;
  got 0.1 mV. This will raise ValueError at declaration from braincell 0.4.`
  An unknown name warns with a close match, e.g. `Channel 'leak' (IL) got an
  unexpected parameter 'gmax'. Did you mean 'g_max'?`. `init_state()` still
  raises the same exception as before, so only code that builds such a
  declaration and never initializes it is affected; from 0.4 the declaration
  itself will raise. Callables, braintools initializers and names not yet
  registered are still checked at `init_state()` only.
- **Directly constructing a Channel or Ion with wrongly united parameters**
  (MECH-28). A `bc.Channel` or `bc.Ion` built outside a Cell (for example in a
  `SingleCompartment`) now compares the units of its `parameter_info()`
  parameters once its outermost `__init__` returns, and emits a
  `FutureWarning` at the constructing line, e.g. `Channel 'IL' parameter
  'g_max' requires a quantity compatible with mS / cm^2; got 0.1 (no unit).
  This will raise TypeError at construction from braincell 0.4.` Such an
  object already failed at its first `update` with a `UnitMismatchError` that
  named neither class nor parameter; the object is built as before. Only units
  are compared, so values traced under `jax.jit` or `jax.grad` are checked too;
  `*_initializer` parameters, callables and `None` are skipped. Subclass
  `__init__` signatures are unchanged. Synapses already check at construction.


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

### Errata

Added after the release (API-7): changes 0.2.0 shipped without an entry, and
one correction.

- **25 names were removed from `__all__` without a deprecation:**
  - `braincell.network` (18), replaced by the network redesign with no
    one-to-one successor: `Connection`, `ConnectionBlock`, `ContactMethod`,
    `ContactTable`, `EdgeMethod`, `EdgeSet`, `NetworkRunResult`, `Projection`,
    `ProjectionContactContext`, `ProjectionEdgeContext`, `all_pairs`,
    `by_post`, `dense`, `explicit_contacts`, `lower_connections`, `pairs`,
    `per_edge`, `probability`;
  - `braincell.quad` (6 step functions, no replacement): `cn_exp_euler_step`,
    `cn_rk4_step`, `exp_exp_euler_step`, `implicit_exp_euler_step`,
    `implicit_rk4_step`, `splitting_step`;
  - `braincell.mech.NetStim`, a `Point` declaration in 0.1.0; the
    `braincell.NetStim` of 0.2.0 is an `EventSource`, a different type.
- **The 19 channel aliases deprecated in 0.1.0 were removed.** Use the name
  without the leading `I`: `INa_HH1952`, `INa_Ba2002`, `INa_TM1991`,
  `IK_HH1952`, `IK_TM1991`, `IK_Leak`, `IKDR_Ba2002`, `IKNI_Ya1989`,
  `IKA1_HM1992`, `IKA2_HM1992`, `IKK2A_HM1992`, `IKK2B_HM1992`,
  `ICaN_IS2008`, `ICaL_IS2008`, `ICaT_HM1992`, `ICaT_HP1992`, `ICaHT_HM1992`,
  `ICaHT_Re1993` and `IAHP_De1994` (e.g. `braincell.channel.Na_HH1952`).
- **Correction to Deprecations:** a `Markov` subclass that does not declare
  `dependent_state` does not warn; 0.2.0 raises `ValueError` when the class is
  created.


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
