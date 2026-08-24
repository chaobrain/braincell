# Consolidated Visualization Tutorial

## Goal

Replace the three overlapping visualization notebooks under
`examples/multi_compartment/vis/` with one executable tutorial. The tutorial
must cover morphology rendering, Cell topology views, scalar overlays, dynamic
values, morphometry, interaction, comparison, and export without teaching
private APIs.

## Structure

The retained notebook is `vis.ipynb`; `vis2d.ipynb` and `vis_old.ipynb` are
removed after their unique material is incorporated. A small synthetic
morphology with complete 3-D point geometry keeps every example self-contained
and fast.

Each feature family has one runnable demonstration. Alternatives that differ
only by parameter choice are summarized in tables so readers can modify the
nearby example instead of stepping through a large gallery. The notebook uses
the full `braincell.vis.plot2d` and `plot3d` APIs for advanced morphology views,
and the Cell convenience APIs for node, CV, and branch topology.

The tutorial sections are:

1. setup, morphology construction, and API entry-point map;
2. static 2-D layouts and shapes;
3. one-off, scoped, and publication styling;
4. scalar values plus Region/Locset overlays;
5. guarded 3-D rendering;
6. node, CV, and branch Cell topology;
7. 2-D animation and guarded GIF export;
8. morphology-aligned traces;
9. dendrogram, topology, Sholl, and branch-order plots;
10. `VisHooks` interaction;
11. morphology/value comparison and publication export;
12. a task-oriented quick reference.

## Interface Rules

- Import only public symbols from `braincell`, `braincell.filter`,
  `braincell.mech`, and `braincell.vis`.
- Do not use the unexported `braincell.vis.compare2d` helper.
- Use `root_layout="type_split"`; do not demonstrate the deprecated legacy
  layout.
- Explain the capability differences among `plot2d`, `plot3d`, `vis_node`,
  `vis_cv`, and `vis_branch` in tables.
- Generate time-varying values with array broadcasting, not a Python model
  loop.
- Keep every physical model value explicitly united with `brainunit`.

## Optional Features

Core Matplotlib examples always execute. PyVista/Plotly 3-D rendering and the
Pillow GIF writer are detected before use. An unavailable optional dependency
prints a concise skip message and does not interrupt the notebook.

Generated tutorial artifacts live in the repository's ignored
`data/vis_outputs/` directory. The animation uses a small frame count and figure
size to keep execution time and notebook output bounded.

## Verification

- Execute the retained notebook from start to finish in the supported BrainCell
  environment with a CPU JAX backend.
- Confirm the executed notebook contains no error outputs.
- Confirm the generated GIF and PDF exist and are non-empty when their writers
  are available.
- Confirm each public feature family above has a runnable example and each
  parameter family has a concise comparison table.
- Confirm only `vis.ipynb` remains in the example directory and unrelated
  worktree changes are preserved.
