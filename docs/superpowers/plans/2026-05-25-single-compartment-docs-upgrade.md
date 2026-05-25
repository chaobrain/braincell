# Single-Compartment Docs Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unfold the tutorial toctrees on the docs landing page, delete the broken `advanced_tutorial/` directory, and fix + rename + expand the single-compartment example notebooks.

**Architecture:** Pure documentation work under `docs/`. Three independent parts: (1) `index.rst` toctree changes, (2) directory deletion, (3) example notebooks — renames, channel-name/bug fixes, and five new notebooks. Notebooks are validated by executing them with `nbconvert` (myst-nb has `jupyter_execute_notebooks = "off"`, so committed outputs are what render); the final Sphinx build validates toctrees.

**Tech Stack:** Sphinx + `myst_nb`, Jupyter notebooks (`.ipynb`), `braincell` / `brainstate` / `brainunit` / `braintools`, matplotlib.

---

## Conventions used by every notebook task

**Repo root (worktree):** `/mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem`
`braincell` is **not** pip-installed — it resolves via the repo root on `sys.path`. All notebook execution MUST set `PYTHONPATH` to the repo root.

**Execute a notebook in place (fills + saves outputs):**
```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  docs/single_compartment/examples/<file>.ipynb
```
Expected: exit code 0, prints `[NbConvertApp] Writing ... <file>.ipynb`. Any cell exception fails the run.

**Canonical channel-name map** (from `braincell/channel/__init__.py` `_DEPRECATED_ALIASES`; canonical = drop leading `I`). Deprecated names still work but emit `DeprecationWarning`:

| deprecated | canonical |
|---|---|
| `INa_HH1952` | `Na_HH1952` |
| `INa_Ba2002` | `Na_Ba2002` |
| `INa_TM1991` | `Na_TM1991` |
| `IK_HH1952` | `K_HH1952` |
| `IK_TM1991` | `K_TM1991` |
| `IK_Leak` | `K_Leak` |
| `IKDR_Ba2002` | `KDR_Ba2002` |
| `ICaL_IS2008` | `CaL_IS2008` |
| `ICaN_IS2008` | `CaN_IS2008` |
| `ICaT_HM1992` | `CaT_HM1992` |
| `ICaT_HP1992` | `CaT_HP1992` |
| `ICaHT_HM1992` | `CaHT_HM1992` |
| `IAHP_De1994` | `AHP_De1994` |

**Two hard breakages (NOT simple aliases) — must be fixed wherever they appear:**
- `Ih_HM1992` → **`HCN_HM1992`** (the `I`-name was removed entirely; raises `AttributeError`). Constructor: `HCN_HM1992(size, g_max=..., E=...)`.
- `KDR_Ba2002` / `Na_Ba2002` **no longer accept `phi=`** (replaced by `temp`/`q10`). Delete any `phi=0.25` argument passed to these channels.

**Current-injection convention:** the default `SingleCompartment.update(I)` expects a **current density** (e.g. `u.uA / u.cm**2` or `u.nA / u.cm**2`), not an absolute current. Validated working densities are baked into the new-notebook tasks below.

---

## Part 1 — Landing page

### Task 1: Unfold toctrees, drop Advanced Tutorials, fix quickstart channels in `index.rst`

**Files:**
- Modify: `docs/index.rst`

- [ ] **Step 1: Fix the broken channel names in the quickstart code block**

In `docs/index.rst`, the `HTC` quickstart (lines ~64–81) uses deprecated/broken names. Apply these exact replacements (each appears once):

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
sed -i \
 -e 's/channel\.INa_Ba2002/channel.Na_Ba2002/' \
 -e 's/channel\.IK_Leak/channel.K_Leak/' \
 -e 's/channel\.IKDR_Ba2002/channel.KDR_Ba2002/' \
 -e 's/channel\.ICaL_IS2008/channel.CaL_IS2008/' \
 -e 's/channel\.ICaN_IS2008/channel.CaN_IS2008/' \
 -e 's/channel\.ICaT_HM1992/channel.CaT_HM1992/' \
 -e 's/channel\.ICaHT_HM1992/channel.CaHT_HM1992/' \
 -e 's/channel\.IAHP_De1994/channel.AHP_De1994/' \
 -e 's/channel\.Ih_HM1992/channel.HCN_HM1992/' \
 docs/index.rst
```

- [ ] **Step 2: Remove `phi=0.25` from the `KDR_Ba2002` line**

Edit `docs/index.rst`, change the IDR line:
```
           self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV, phi=0.25))
```
to:
```
           self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV))
```

- [ ] **Step 3: Verify the quickstart snippet now imports/constructs cleanly**

Run (extracts and executes just the class definition logic):
```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD python3 - <<'PY'
import braincell, brainstate, braintools, brainunit as u
class HTC(braincell.SingleCompartment):
    def __init__(self, size, solver='ind_exp_euler'):
        super().__init__(size, V_initializer=braintools.init.Constant(-65.*u.mV), V_th=20.*u.mV, solver=solver)
        self.na=braincell.ion.SodiumFixed(size,E=50.*u.mV); self.na.add(INa=braincell.channel.Na_Ba2002(size,V_sh=-30*u.mV))
        self.k=braincell.ion.PotassiumFixed(size,E=-90.*u.mV)
        self.k.add(IKL=braincell.channel.K_Leak(size,g_max=0.01*(u.mS/u.cm**2)))
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size,V_sh=-30.*u.mV))
        self.ca=braincell.ion.CalciumDetailed(size,C_rest=5e-5*u.mM,tau=10.*u.ms,d=0.5*u.um)
        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size,g_max=0.5*(u.mS/u.cm**2)))
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size,g_max=0.5*(u.mS/u.cm**2)))
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size,g_max=2.1*(u.mS/u.cm**2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size,g_max=3.0*(u.mS/u.cm**2)))
        self.kca=braincell.MixIons(self.k,self.ca); self.kca.add(IAHP=braincell.channel.AHP_De1994(size,g_max=0.3*(u.mS/u.cm**2)))
        self.Ih=braincell.channel.HCN_HM1992(size,g_max=0.01*(u.mS/u.cm**2),E=-43*u.mV)
        self.IL=braincell.channel.IL(size,g_max=0.0075*(u.mS/u.cm**2),E=-70*u.mV)
HTC(1); print("quickstart OK")
PY
```
Expected: `quickstart OK` with no `AttributeError`/`TypeError`/`DeprecationWarning`.

- [ ] **Step 4: Unfold the tutorial toctrees and delete the Advanced Tutorials block**

In `docs/index.rst`, the toctree section currently reads:
```rst
.. toctree::
   :maxdepth: 2
   :caption: Single-Compartment Modeling
   :hidden:

   single_compartment/index

.. toctree::
   :maxdepth: 2
   :caption: Multi-Compartment Modeling
   :hidden:

   multi_compartment/index


.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   advanced_tutorial/index


.. toctree::
   :maxdepth: 2
   :caption: Numerical Integration
   :hidden:

   integration/index
```
Replace that entire block with (drop the three `:hidden:` lines for the modeling + integration sections; remove the Advanced Tutorials toctree entirely):
```rst
.. toctree::
   :maxdepth: 2
   :caption: Single-Compartment Modeling

   single_compartment/index

.. toctree::
   :maxdepth: 2
   :caption: Multi-Compartment Modeling

   multi_compartment/index

.. toctree::
   :maxdepth: 2
   :caption: Numerical Integration

   integration/index
```
Leave the **API Documentation** toctree (further down) unchanged, including its `:hidden:`.

- [ ] **Step 5: Confirm no lingering references**

Run:
```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
grep -n "advanced_tutorial\|phi=0.25\|Ih_HM1992\|channel.I" docs/index.rst || echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 6: Commit**

```bash
git add docs/index.rst
git commit -m "docs: unfold tutorial toctrees, drop advanced_tutorial, fix quickstart channels"
```

---

## Part 2 — Delete advanced_tutorial

### Task 2: Delete the `docs/advanced_tutorial/` directory

**Files:**
- Delete: `docs/advanced_tutorial/` (contains `index.rst`, `differential_equation.ipynb`, `rationale.ipynb`)

- [ ] **Step 1: Confirm nothing else references it**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
grep -rn "advanced_tutorial" docs/ --include=*.rst --include=*.ipynb || echo "NO REFERENCES"
```
Expected: `NO REFERENCES` (Task 1 already removed the only one).

- [ ] **Step 2: Remove the directory**

```bash
git rm -r docs/advanced_tutorial/
```

- [ ] **Step 3: Verify gone**

```bash
test ! -d docs/advanced_tutorial && echo "DELETED"
```
Expected: `DELETED`.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: delete orphaned advanced_tutorial directory"
```

---

## Part 3a — Fix + rename existing examples

> All four tasks operate in `docs/single_compartment/examples/`. Renames use `git mv`; channel fixes use `sed` on the `.ipynb` (token replacement is safe in notebook JSON). Each task ends by executing the renamed notebook.

### Task 3: `sc04.ipynb` → `integration_methods.ipynb`

**Files:**
- Rename: `docs/single_compartment/examples/sc04.ipynb` → `integration_methods.ipynb`

- [ ] **Step 1: Rename**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
git mv docs/single_compartment/examples/sc04.ipynb docs/single_compartment/examples/integration_methods.ipynb
```

- [ ] **Step 2: Fix channel names**

```bash
sed -i -e 's/INa_HH1952/Na_HH1952/g' -e 's/IK_HH1952/K_HH1952/g' \
  docs/single_compartment/examples/integration_methods.ipynb
```

- [ ] **Step 3: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600 \
  docs/single_compartment/examples/integration_methods.ipynb
```
Expected: exit 0, `Writing ... integration_methods.ipynb`.

- [ ] **Step 4: Confirm no deprecated names remain**

```bash
grep -o "channel\.I[A-Za-z0-9_]*" docs/single_compartment/examples/integration_methods.ipynb || echo "CLEAN"
```
Expected: `CLEAN` (the only `channel.I*` legitimately remaining would be `channel.IL`, which is canonical — if `IL` shows, that's fine; verify only `Na_*`/`K_*` replaced).

- [ ] **Step 5: Commit**

```bash
git add docs/single_compartment/examples/integration_methods.ipynb
git commit -m "docs: rename sc04 to integration_methods, fix HH channel names"
```

### Task 4: `sc02.ipynb` → `calcium_channel_gating.ipynb`

**Files:**
- Rename: `docs/single_compartment/examples/sc02.ipynb` → `calcium_channel_gating.ipynb`

- [ ] **Step 1: Rename**

```bash
git mv docs/single_compartment/examples/sc02.ipynb docs/single_compartment/examples/calcium_channel_gating.ipynb
```

- [ ] **Step 2: Fix channel names**

```bash
sed -i -e 's/ICaT_HP1992/CaT_HP1992/g' -e 's/ICaHT_HM1992/CaHT_HM1992/g' \
  docs/single_compartment/examples/calcium_channel_gating.ipynb
```

- [ ] **Step 3: Fix the title cell (currently a `raw` cell) and drop the empty markdown cell**

The first cell is a `raw` cell holding the H1 title (won't render as a heading) and the second is an empty markdown cell. Convert the title to a markdown heading and delete the empty cell:
```bash
PYTHONPATH=$PWD python3 - <<'PY'
import nbformat as nbf
p="docs/single_compartment/examples/calcium_channel_gating.ipynb"
nb=nbf.read(p, as_version=4)
# Convert leading raw title cell -> markdown
if nb.cells and nb.cells[0].cell_type=="raw":
    nb.cells[0]=nbf.v4.new_markdown_cell(nb.cells[0].source)
# Drop empty markdown cells
nb.cells=[c for c in nb.cells if not (c.cell_type=="markdown" and not c.source.strip())]
nbf.write(nb,p); print("title fixed, empty cells removed")
PY
```
Expected: `title fixed, empty cells removed`.

- [ ] **Step 4: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600 \
  docs/single_compartment/examples/calcium_channel_gating.ipynb
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add docs/single_compartment/examples/calcium_channel_gating.ipynb
git commit -m "docs: rename sc02 to calcium_channel_gating, fix Ca channel names and title cell"
```

### Task 5: `sc05.ipynb` → `thalamic_neurons.ipynb`

**Files:**
- Rename: `docs/single_compartment/examples/sc05.ipynb` → `thalamic_neurons.ipynb`

- [ ] **Step 1: Rename**

```bash
git mv docs/single_compartment/examples/sc05.ipynb docs/single_compartment/examples/thalamic_neurons.ipynb
```

- [ ] **Step 2: Fix all channel names + the two hard breakages**

```bash
F=docs/single_compartment/examples/thalamic_neurons.ipynb
sed -i \
 -e 's/INa_Ba2002/Na_Ba2002/g' \
 -e 's/IKDR_Ba2002/KDR_Ba2002/g' \
 -e 's/IK_Leak/K_Leak/g' \
 -e 's/ICaL_IS2008/CaL_IS2008/g' \
 -e 's/ICaN_IS2008/CaN_IS2008/g' \
 -e 's/ICaT_HM1992/CaT_HM1992/g' \
 -e 's/ICaHT_HM1992/CaHT_HM1992/g' \
 -e 's/IAHP_De1994/AHP_De1994/g' \
 -e 's/Ih_HM1992/HCN_HM1992/g' \
 "$F"
```

- [ ] **Step 3: Remove the `phi=0.25` argument (appears in multiple neuron classes)**

The `KDR_Ba2002` calls pass `phi=0.25`, which the canonical class rejects. Remove it (handles the `, phi=0.25` form used in the notebook):
```bash
F=docs/single_compartment/examples/thalamic_neurons.ipynb
sed -i -e 's/, *phi=0\.25//g' -e 's/,phi=0\.25//g' "$F"
```

- [ ] **Step 4: Replace the Chinese inline comment with English**

The import cell contains `import time  # 用于记录模拟耗时`. Replace:
```bash
sed -i 's/# 用于记录模拟耗时/# for timing the simulation/' docs/single_compartment/examples/thalamic_neurons.ipynb
```

- [ ] **Step 5: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 \
  docs/single_compartment/examples/thalamic_neurons.ipynb
```
Expected: exit 0. (If a cell still references `phi` or `Ih_HM1992`, the run errors — re-check Steps 2–3.)

- [ ] **Step 6: Confirm clean**

```bash
grep -o "phi=0.25\|Ih_HM1992" docs/single_compartment/examples/thalamic_neurons.ipynb || echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 7: Commit**

```bash
git add docs/single_compartment/examples/thalamic_neurons.ipynb
git commit -m "docs: rename sc05 to thalamic_neurons, fix channels (Ih->HCN, drop phi), english comment"
```

### Task 6: `sc03.ipynb` → `ei_network.ipynb`

**Files:**
- Rename: `docs/single_compartment/examples/sc03.ipynb` → `ei_network.ipynb`

Note: this example builds a network and legitimately depends on `brainpy` (2.7.8, installed) for `brainpy.state.AlignPostProj/Expon/COBA` — keep the `brainpy` import. Only fix the channel names, the `add_elem`→`add` inconsistency, and the trailing stray comment.

- [ ] **Step 1: Rename**

```bash
git mv docs/single_compartment/examples/sc03.ipynb docs/single_compartment/examples/ei_network.ipynb
```

- [ ] **Step 2: Fix channel names and method name**

```bash
F=docs/single_compartment/examples/ei_network.ipynb
sed -i -e 's/INa_TM1991/Na_TM1991/g' -e 's/IK_TM1991/K_TM1991/g' \
       -e 's/\.add_elem(/.add(/g' "$F"
```

- [ ] **Step 3: Remove the stray trailing comment**

Delete the line that reads (a leftover apology about Jupyter interruption):
```
# In this example, visualization cannot be completed after interruption in Jupyter, so temporarily keep the simulation results!
```
Use Edit to remove that exact comment line from the final code cell (leave the surrounding plotting code intact).

- [ ] **Step 4: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1200 \
  docs/single_compartment/examples/ei_network.ipynb
```
Expected: exit 0. (4000-neuron, 100 ms sim on CPU — may take a minute. If it errors on units, the existing absolute-unit `g_max`/`C` setup needs the error addressed; report the traceback before changing the science.)

- [ ] **Step 5: Commit**

```bash
git add docs/single_compartment/examples/ei_network.ipynb
git commit -m "docs: rename sc03 to ei_network, fix channel names, standardize .add(), drop stray comment"
```

---

## Part 3b — New example notebooks

> Each new notebook is built by an `nbformat` script (full content below), then executed in place to capture outputs. All densities/durations below are validated to run and to exhibit the target phenomenon on CPU.

### Task 7: New `hh_neuron_basics.ipynb`

**Files:**
- Create: `docs/single_compartment/examples/hh_neuron_basics.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD python3 - <<'PY'
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.cells=[
nbf.v4.new_markdown_cell(
"# Your First Hodgkin–Huxley Neuron\n\n"
"This example builds a single Hodgkin–Huxley (HH) neuron, injects a constant "
"current, and plots the resulting membrane-potential trace and spikes. It is "
"the simplest end-to-end workflow in `braincell`: define a cell, initialize "
"its state, step it through time, and read out `V` and `spike`."),
nbf.v4.new_code_cell(
"import brainstate\n"
"import brainunit as u\n"
"import matplotlib.pyplot as plt\n"
"import braincell"),
nbf.v4.new_markdown_cell(
"## Define the neuron\n\n"
"A `SingleCompartment` subclass holds ion channels grouped by ion. Here we use "
"the classic HH sodium and potassium channels plus a passive leak. `V_th` only "
"sets the threshold used to emit `spike` events for plotting; it does not alter "
"the dynamics."),
nbf.v4.new_code_cell(
"class HH(braincell.SingleCompartment):\n"
"    def __init__(self, size, solver='exp_euler'):\n"
"        super().__init__(size, V_th=20. * u.mV, solver=solver)\n"
"        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)\n"
"        self.na.add(INa=braincell.channel.Na_HH1952(size))\n"
"        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)\n"
"        self.k.add(IK=braincell.channel.K_HH1952(size))\n"
"        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,\n"
"                                       g_max=0.03 * (u.mS / u.cm ** 2))"),
nbf.v4.new_markdown_cell(
"## Run the simulation\n\n"
"We inject a constant current **density** of `5 uA/cm^2`. `update` advances the "
"cell one `dt`; we record both the membrane potential and the spike flag at "
"each step with `brainstate.transform.for_loop`."),
nbf.v4.new_code_cell(
"neuron = HH(1)\n"
"neuron.init_state()\n\n"
"I = 5. * u.uA / u.cm ** 2\n\n"
"def step(t):\n"
"    with brainstate.environ.context(t=t):\n"
"        neuron.update(I)\n"
"    return neuron.V.value, neuron.spike.value\n\n"
"with brainstate.environ.context(dt=0.01 * u.ms):\n"
"    times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())\n"
"    vs, spikes = brainstate.transform.for_loop(step, times)\n\n"
"print('number of spikes:', int(u.math.sum(spikes)))"),
nbf.v4.new_markdown_cell(
"## Plot the membrane potential\n\n"
"The trace shows the characteristic train of action potentials driven by the "
"constant input."),
nbf.v4.new_code_cell(
"plt.figure(figsize=(8, 3))\n"
"plt.plot(times / u.ms, u.math.squeeze(vs) / u.mV, linewidth=1.2)\n"
"plt.xlabel('Time (ms)')\n"
"plt.ylabel('Membrane potential (mV)')\n"
"plt.title('HH neuron under 5 uA/cm^2 constant current')\n"
"plt.tight_layout()\n"
"plt.show()"),
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
nbf.write(nb,"docs/single_compartment/examples/hh_neuron_basics.ipynb")
print("built hh_neuron_basics.ipynb")
PY
```
Expected: `built hh_neuron_basics.ipynb`.

- [ ] **Step 2: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600 \
  docs/single_compartment/examples/hh_neuron_basics.ipynb
```
Expected: exit 0; the `number of spikes:` output is `> 0`.

- [ ] **Step 3: Commit**

```bash
git add docs/single_compartment/examples/hh_neuron_basics.ipynb
git commit -m "docs: add hh_neuron_basics example"
```

### Task 8: New `fi_curve.ipynb`

**Files:**
- Create: `docs/single_compartment/examples/fi_curve.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD python3 - <<'PY'
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.cells=[
nbf.v4.new_markdown_cell(
"# The Frequency–Current (F–I) Curve\n\n"
"A neuron's **F–I curve** relates the amplitude of a sustained input current to "
"its steady-state firing rate. It is one of the most basic characterizations of "
"a neuron's excitability. Here we drive a population of identical HH neurons, "
"each with a different constant current, and measure the firing rate of each."),
nbf.v4.new_code_cell(
"import brainstate\n"
"import brainunit as u\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import braincell"),
nbf.v4.new_markdown_cell(
"## A population of HH neurons\n\n"
"`braincell` cells are vectorized: building `HH(N)` creates `N` independent "
"neurons that we can drive with an `N`-vector of currents in a single pass."),
nbf.v4.new_code_cell(
"class HH(braincell.SingleCompartment):\n"
"    def __init__(self, size, solver='exp_euler'):\n"
"        super().__init__(size, V_th=20. * u.mV, solver=solver)\n"
"        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)\n"
"        self.na.add(INa=braincell.channel.Na_HH1952(size))\n"
"        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)\n"
"        self.k.add(IK=braincell.channel.K_HH1952(size))\n"
"        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,\n"
"                                       g_max=0.03 * (u.mS / u.cm ** 2))"),
nbf.v4.new_markdown_cell(
"## Sweep the input current\n\n"
"We sweep 11 current densities from 0 to 20 uA/cm^2. We simulate 600 ms and "
"**discard the first 100 ms** as a warm-up so onset transients do not inflate "
"the rate, then count spikes over the remaining 500 ms."),
nbf.v4.new_code_cell(
"n_levels = 11\n"
"amplitudes = np.linspace(0., 20., n_levels)        # uA/cm^2\n"
"I = amplitudes * (u.uA / u.cm ** 2)\n\n"
"net = HH(n_levels)\n"
"net.init_state()\n\n"
"warmup = 100. * u.ms\n"
"total = 600. * u.ms\n\n"
"def step(t):\n"
"    with brainstate.environ.context(t=t):\n"
"        net.update(I)\n"
"    return t, net.spike.value\n\n"
"with brainstate.environ.context(dt=0.01 * u.ms):\n"
"    times = u.math.arange(0. * u.ms, total, brainstate.environ.get_dt())\n"
"    ts, spikes = brainstate.transform.for_loop(step, times)\n\n"
"mask = (ts >= warmup)\n"
"counts = np.asarray(u.math.sum(spikes[mask], axis=0))\n"
"rate = counts / float((total - warmup) / u.second)   # Hz\n"
"print('spike counts:', counts.astype(int).tolist())"),
nbf.v4.new_markdown_cell(
"## Plot the F–I curve\n\n"
"Firing rate rises monotonically with input current — the signature of a "
"Type-I/Type-II excitable membrane."),
nbf.v4.new_code_cell(
"plt.figure(figsize=(5, 4))\n"
"plt.plot(amplitudes, rate, 'o-')\n"
"plt.xlabel('Input current density (uA/cm$^2$)')\n"
"plt.ylabel('Firing rate (Hz)')\n"
"plt.title('F–I curve of an HH neuron')\n"
"plt.tight_layout()\n"
"plt.show()"),
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
nbf.write(nb,"docs/single_compartment/examples/fi_curve.ipynb")
print("built fi_curve.ipynb")
PY
```
Expected: `built fi_curve.ipynb`.

- [ ] **Step 2: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 \
  docs/single_compartment/examples/fi_curve.ipynb
```
Expected: exit 0. Acceptance: printed `spike counts` are **non-decreasing** and `max > min` (a rising curve). If the curve is flat, widen the sweep upper bound to 40 uA/cm^2 and re-run.

- [ ] **Step 3: Commit**

```bash
git add docs/single_compartment/examples/fi_curve.ipynb
git commit -m "docs: add fi_curve example"
```

### Task 9: New `channel_ablation.ipynb`

**Files:**
- Create: `docs/single_compartment/examples/channel_ablation.ipynb`

- [ ] **Step 1: Build the notebook**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD python3 - <<'PY'
import nbformat as nbf
nb=nbf.v4.new_notebook()
nb.cells=[
nbf.v4.new_markdown_cell(
"# Channel Ablation: What the Potassium Current Does\n\n"
"To build intuition for what each conductance contributes, we can **ablate** a "
"channel — set its maximal conductance to zero — and watch how the dynamics "
"change. Here we compare an intact HH neuron against one whose delayed-rectifier "
"potassium current has been removed. Potassium repolarizes the membrane after a "
"spike, so removing it should abolish normal, repetitive spiking."),
nbf.v4.new_code_cell(
"import brainstate\n"
"import brainunit as u\n"
"import matplotlib.pyplot as plt\n"
"import braincell"),
nbf.v4.new_markdown_cell(
"## A neuron with a tunable potassium conductance\n\n"
"We expose the potassium maximal conductance `gK` as a constructor argument so "
"we can instantiate an intact cell and an ablated cell (`gK = 0`) from the same "
"class."),
nbf.v4.new_code_cell(
"class HH(braincell.SingleCompartment):\n"
"    def __init__(self, size, gK=36. * (u.mS / u.cm ** 2), solver='exp_euler'):\n"
"        super().__init__(size, V_th=20. * u.mV, solver=solver)\n"
"        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)\n"
"        self.na.add(INa=braincell.channel.Na_HH1952(size))\n"
"        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)\n"
"        self.k.add(IK=braincell.channel.K_HH1952(size, g_max=gK))\n"
"        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,\n"
"                                       g_max=0.03 * (u.mS / u.cm ** 2))"),
nbf.v4.new_markdown_cell(
"## Simulate intact vs. ablated\n\n"
"Both cells receive the same 5 uA/cm^2 current. The intact cell uses the default "
"potassium conductance; the ablated cell sets `gK = 0`."),
nbf.v4.new_code_cell(
"intact = HH(1)\n"
"ablated = HH(1, gK=0. * (u.mS / u.cm ** 2))\n"
"intact.init_state()\n"
"ablated.init_state()\n\n"
"I = 5. * u.uA / u.cm ** 2\n\n"
"def step(t):\n"
"    with brainstate.environ.context(t=t):\n"
"        intact.update(I)\n"
"        ablated.update(I)\n"
"    return intact.V.value, ablated.V.value\n\n"
"with brainstate.environ.context(dt=0.01 * u.ms):\n"
"    times = u.math.arange(0. * u.ms, 80. * u.ms, brainstate.environ.get_dt())\n"
"    v_intact, v_ablated = brainstate.transform.for_loop(step, times)"),
nbf.v4.new_markdown_cell(
"## Compare the traces\n\n"
"The intact neuron fires a clean spike train; without potassium the membrane "
"cannot repolarize and gets stuck in a depolarized state (depolarization block)."),
nbf.v4.new_code_cell(
"plt.figure(figsize=(8, 3))\n"
"plt.plot(times / u.ms, u.math.squeeze(v_intact) / u.mV, label='intact')\n"
"plt.plot(times / u.ms, u.math.squeeze(v_ablated) / u.mV, label='no $I_K$', linestyle='--')\n"
"plt.xlabel('Time (ms)')\n"
"plt.ylabel('Membrane potential (mV)')\n"
"plt.title('Effect of ablating the potassium current')\n"
"plt.legend()\n"
"plt.tight_layout()\n"
"plt.show()"),
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
nbf.write(nb,"docs/single_compartment/examples/channel_ablation.ipynb")
print("built channel_ablation.ipynb")
PY
```
Expected: `built channel_ablation.ipynb`.

- [ ] **Step 2: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600 \
  docs/single_compartment/examples/channel_ablation.ipynb
```
Expected: exit 0. (If `K_HH1952` rejects `g_max=`, inspect its signature with `python3 -c "import braincell,inspect;print(inspect.signature(braincell.channel.K_HH1952.__init__))"` and use the correct conductance kwarg name; this channel is HH-family and exposes `g_max`.)

- [ ] **Step 3: Commit**

```bash
git add docs/single_compartment/examples/channel_ablation.ipynb
git commit -m "docs: add channel_ablation example"
```

### Task 10: New `spike_frequency_adaptation.ipynb`

**Files:**
- Create: `docs/single_compartment/examples/spike_frequency_adaptation.ipynb`

This reuses the validated thalamic-relay-cell channel set (canonical names, no `phi`). Under sustained current its calcium-activated potassium (AHP) current builds up, lengthening successive inter-spike intervals — spike-frequency adaptation.

- [ ] **Step 1: Build the notebook**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD python3 - <<'PY'
import nbformat as nbf
CLASS = (
"class AdaptingCell(braincell.SingleCompartment):\n"
"    \"\"\"A thalamic-relay-style cell with Ca-activated K (AHP) current.\"\"\"\n"
"    def __init__(self, size, solver='ind_exp_euler'):\n"
"        super().__init__(size, V_initializer=braintools.init.Constant(-65. * u.mV),\n"
"                         V_th=20. * u.mV, solver=solver)\n"
"        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)\n"
"        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30 * u.mV))\n"
"        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)\n"
"        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))\n"
"        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV))\n"
"        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM,\n"
"                                                tau=10. * u.ms, d=0.5 * u.um)\n"
"        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))\n"
"        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))\n"
"        self.kca = braincell.MixIons(self.k, self.ca)\n"
"        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2)))\n"
"        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)"
)
nb=nbf.v4.new_notebook()
nb.cells=[
nbf.v4.new_markdown_cell(
"# Spike-Frequency Adaptation\n\n"
"Many neurons fire rapidly at the onset of a sustained stimulus and then slow "
"down. This **spike-frequency adaptation** is commonly produced by a "
"calcium-activated potassium current (an AHP current): each spike admits "
"calcium, calcium gradually activates the AHP conductance, and the growing "
"outward current lengthens successive inter-spike intervals (ISIs)."),
nbf.v4.new_code_cell(
"import brainstate\n"
"import braintools\n"
"import brainunit as u\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import braincell"),
nbf.v4.new_markdown_cell(
"## A cell with an AHP current\n\n"
"We combine sodium and delayed-rectifier potassium for spiking, an L- and "
"T-type calcium current as the calcium source, and a calcium-activated "
"potassium current (`AHP_De1994`) created on a `MixIons` group of K and Ca."),
nbf.v4.new_code_cell(CLASS),
nbf.v4.new_markdown_cell(
"## Drive with a sustained current\n\n"
"A constant 2 uA/cm^2 injection evokes repetitive firing. We record the spike "
"times to measure how the ISIs evolve."),
nbf.v4.new_code_cell(
"cell = AdaptingCell(1)\n"
"cell.init_state()\n\n"
"I = 2. * u.uA / u.cm ** 2\n\n"
"def step(t):\n"
"    with brainstate.environ.context(t=t):\n"
"        cell.update(I)\n"
"    return cell.V.value, cell.spike.value\n\n"
"with brainstate.environ.context(dt=0.01 * u.ms):\n"
"    times = u.math.arange(0. * u.ms, 500. * u.ms, brainstate.environ.get_dt())\n"
"    vs, spikes = brainstate.transform.for_loop(step, times)\n\n"
"spike_times = np.asarray(times[u.math.squeeze(spikes) > 0] / u.ms)\n"
"isi = np.diff(spike_times)\n"
"print('number of spikes:', spike_times.size)\n"
"print('first ISIs (ms):', np.round(isi[:6], 2).tolist())"),
nbf.v4.new_markdown_cell(
"## Visualize the adaptation\n\n"
"The voltage trace (top) shows firing that slows over time; the ISI sequence "
"(bottom) grows monotonically — the quantitative signature of adaptation."),
nbf.v4.new_code_cell(
"fig, axs = plt.subplots(2, 1, figsize=(8, 5))\n"
"axs[0].plot(times / u.ms, u.math.squeeze(vs) / u.mV, linewidth=0.8)\n"
"axs[0].set_xlabel('Time (ms)'); axs[0].set_ylabel('V (mV)')\n"
"axs[0].set_title('Membrane potential under sustained drive')\n"
"axs[1].plot(np.arange(1, isi.size + 1), isi, 'o-')\n"
"axs[1].set_xlabel('Inter-spike interval index'); axs[1].set_ylabel('ISI (ms)')\n"
"axs[1].set_title('ISIs lengthen over time (adaptation)')\n"
"plt.tight_layout(); plt.show()"),
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
nbf.write(nb,"docs/single_compartment/examples/spike_frequency_adaptation.ipynb")
print("built spike_frequency_adaptation.ipynb")
PY
```
Expected: `built spike_frequency_adaptation.ipynb`.

- [ ] **Step 2: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 \
  docs/single_compartment/examples/spike_frequency_adaptation.ipynb
```
Expected: exit 0. Acceptance: the printed `first ISIs (ms)` are increasing (validated example: ~`[8.37, 11.47, 14.09, 15.35, 15.72, ...]`). If ISIs do not grow, raise the AHP `g_max` to `0.5 * (u.mS / u.cm ** 2)` and re-run.

- [ ] **Step 3: Commit**

```bash
git add docs/single_compartment/examples/spike_frequency_adaptation.ipynb
git commit -m "docs: add spike_frequency_adaptation example"
```

### Task 11: New `t_current_rebound.ipynb`

**Files:**
- Create: `docs/single_compartment/examples/t_current_rebound.ipynb`

Reuses the same thalamic-style cell. A hyperpolarizing step de-inactivates the low-threshold T-type calcium current; when the step is released, the rebound depolarization triggers a burst of spikes — the classic thalamic rebound burst.

- [ ] **Step 1: Build the notebook**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
PYTHONPATH=$PWD python3 - <<'PY'
import nbformat as nbf
CLASS = (
"class ThalamicCell(braincell.SingleCompartment):\n"
"    \"\"\"Thalamic-relay-style cell with a low-threshold T-type Ca current.\"\"\"\n"
"    def __init__(self, size, solver='ind_exp_euler'):\n"
"        super().__init__(size, V_initializer=braintools.init.Constant(-65. * u.mV),\n"
"                         V_th=20. * u.mV, solver=solver)\n"
"        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)\n"
"        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30 * u.mV))\n"
"        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)\n"
"        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))\n"
"        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV))\n"
"        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM,\n"
"                                                tau=10. * u.ms, d=0.5 * u.um)\n"
"        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))\n"
"        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))\n"
"        self.Ih = braincell.channel.HCN_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)\n"
"        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)"
)
nb=nbf.v4.new_notebook()
nb.cells=[
nbf.v4.new_markdown_cell(
"# Post-Inhibitory Rebound Bursting\n\n"
"Thalamic relay neurons can fire a **rebound burst** after a period of "
"hyperpolarization. The low-threshold T-type calcium current inactivates at "
"rest, but a sustained hyperpolarizing input removes that inactivation. When "
"the hyperpolarization ends, the T-current activates transiently and drives a "
"brief burst of spikes — even though the net input is now zero."),
nbf.v4.new_code_cell(
"import brainstate\n"
"import braintools\n"
"import brainunit as u\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import braincell"),
nbf.v4.new_markdown_cell(
"## A cell with a T-type calcium current\n\n"
"The key ingredient is the low-threshold T-type calcium channel "
"(`CaT_HM1992`); `HCN_HM1992` (the H-current) sets a realistic resting "
"potential, and the high-threshold calcium current shapes the burst."),
nbf.v4.new_code_cell(CLASS),
nbf.v4.new_markdown_cell(
"## Apply a hyperpolarizing step, then release\n\n"
"We inject a hyperpolarizing current density of -2 uA/cm^2 for the first "
"200 ms, then switch to zero. The current is a function of time `t`, selected "
"with `u.math.where`."),
nbf.v4.new_code_cell(
"cell = ThalamicCell(1)\n"
"cell.init_state()\n\n"
"def I_of_t(t):\n"
"    return u.math.where(t < 200. * u.ms,\n"
"                        -2. * u.uA / u.cm ** 2,\n"
"                        0. * u.uA / u.cm ** 2)\n\n"
"def step(t):\n"
"    with brainstate.environ.context(t=t):\n"
"        cell.update(I_of_t(t))\n"
"    return cell.V.value, cell.spike.value\n\n"
"with brainstate.environ.context(dt=0.01 * u.ms):\n"
"    times = u.math.arange(0. * u.ms, 500. * u.ms, brainstate.environ.get_dt())\n"
"    vs, spikes = brainstate.transform.for_loop(step, times)\n\n"
"spike_times = np.asarray(times[u.math.squeeze(spikes) > 0] / u.ms)\n"
"rebound = spike_times[spike_times > 200.]\n"
"print('spikes after release (ms):', np.round(rebound[:10], 1).tolist())"),
nbf.v4.new_markdown_cell(
"## Visualize the rebound burst\n\n"
"During the hyperpolarizing step the cell is silent and below rest; on release "
"it overshoots and fires a short, high-frequency burst before settling."),
nbf.v4.new_code_cell(
"plt.figure(figsize=(8, 3))\n"
"plt.plot(times / u.ms, u.math.squeeze(vs) / u.mV, linewidth=0.8)\n"
"plt.axvline(200., color='k', linestyle=':', label='release')\n"
"plt.xlabel('Time (ms)'); plt.ylabel('Membrane potential (mV)')\n"
"plt.title('Post-inhibitory rebound burst (T-type Ca current)')\n"
"plt.legend(); plt.tight_layout(); plt.show()"),
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
nbf.write(nb,"docs/single_compartment/examples/t_current_rebound.ipynb")
print("built t_current_rebound.ipynb")
PY
```
Expected: `built t_current_rebound.ipynb`.

- [ ] **Step 2: Execute**

```bash
PYTHONPATH=$PWD jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 \
  docs/single_compartment/examples/t_current_rebound.ipynb
```
Expected: exit 0. Acceptance: `spikes after release (ms)` is non-empty with values just after 200 ms (validated example: a burst around ~266–298 ms). If empty, deepen/lengthen the step (e.g. -3 uA/cm^2 for 250 ms) and re-run.

- [ ] **Step 3: Commit**

```bash
git add docs/single_compartment/examples/t_current_rebound.ipynb
git commit -m "docs: add t_current_rebound example"
```

---

## Part 4 — Wire up toctrees and build

### Task 12: Update the example toctrees

**Files:**
- Modify: `docs/single_compartment/examples/index.rst`
- Modify: `docs/single_compartment/index.rst`

- [ ] **Step 1: Rewrite `examples/index.rst`**

Replace the entire contents of `docs/single_compartment/examples/index.rst` with:
```rst
Examples
========

.. toctree::
   :maxdepth: 1

   hh_neuron_basics
   calcium_channel_gating
   fi_curve
   channel_ablation
   spike_frequency_adaptation
   t_current_rebound
   integration_methods
   thalamic_neurons
   ei_network
```

- [ ] **Step 2: Update the Examples toctree inlined in `single_compartment/index.rst`**

In `docs/single_compartment/index.rst`, replace the Examples toctree block:
```rst
.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/sc02
   examples/sc03
   examples/sc04
   examples/sc05
```
with:
```rst
.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/hh_neuron_basics
   examples/calcium_channel_gating
   examples/fi_curve
   examples/channel_ablation
   examples/spike_frequency_adaptation
   examples/t_current_rebound
   examples/integration_methods
   examples/thalamic_neurons
   examples/ei_network
```
(Leave the Quickstart and Tutorials toctrees in that file unchanged.)

- [ ] **Step 3: Verify no stale `sc0` references remain anywhere**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
grep -rn "sc0[2-5]" docs/ --include=*.rst || echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 4: Commit**

```bash
git add docs/single_compartment/index.rst docs/single_compartment/examples/index.rst
git commit -m "docs: update example toctrees with renamed and new notebooks"
```

### Task 13: Build the docs and verify

**Files:** none (build verification only)

- [ ] **Step 1: Build HTML**

myst-nb has `jupyter_execute_notebooks = "off"`, so the build uses committed outputs and does not run `braincell`.
```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem/docs
make html 2>&1 | tee /tmp/docbuild.log | tail -25
```
Expected: ends with `build succeeded` (warnings about pre-existing pages are acceptable; new breakage is not).

- [ ] **Step 2: Check for toctree / reference errors introduced by this work**

```bash
grep -iE "advanced_tutorial|sc0[2-5]|toctree contains ref|nonexisting document" /tmp/docbuild.log || echo "NO RELEVANT WARNINGS"
```
Expected: `NO RELEVANT WARNINGS`.

- [ ] **Step 3: Confirm the new pages built**

```bash
cd /mnt/d/codes/projects/braincell/.claude/worktrees/zesty-riding-gem
ls docs/_build/html/single_compartment/examples/ | \
  grep -E "hh_neuron_basics|fi_curve|channel_ablation|spike_frequency_adaptation|t_current_rebound" \
  && echo "NEW PAGES PRESENT"
```
Expected: the five filenames listed, then `NEW PAGES PRESENT`.

- [ ] **Step 4 (optional): commit any build-config touch-ups** — none expected. No commit if `docs/_build/` is gitignored (it should be).

---

## Self-Review Notes

- **Spec coverage:** Part 1 (unfold + drop advanced_tutorial + quickstart fix) → Task 1; delete directory → Task 2; renames + bug fixes → Tasks 3–6; five new examples → Tasks 7–11; toctree wiring → Task 12; build verification → Task 13. All spec sections covered.
- **`Ih_HM1992` / `phi` breakages:** fixed in `index.rst` (Task 1) and `thalamic_neurons` (Task 5); avoided in all new notebooks (canonical names, no `phi`).
- **Type/name consistency:** every new notebook uses `Na_HH1952`/`K_HH1952`/`IL` (HH cells) or the canonical thalamic set; current always passed as a density (`u.uA/u.cm**2`); spike readout always `cell.spike.value`; recording always via `brainstate.transform.for_loop`.
- **Verification is empirical:** notebook tasks gate on `nbconvert` exit 0 plus a phenomenon-specific acceptance check (rising F–I, growing ISIs, post-release burst), each with a concrete fallback knob.
