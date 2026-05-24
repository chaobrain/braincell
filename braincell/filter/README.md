# Filter TODO

## Spatial callable parameters

We want a future API where spatially varying cable and density parameters can be
declared as functions over a local morphology/CV context, instead of manually
looping over branches and painting one branch at a time.

Example target shape:

```python
def pc24_cm(ctx):
    if ctx.branch_type == "soma":
        return 2.0 * (u.uF / u.cm**2)
    diam_um = ctx.diam_arc_mean.to_decimal(u.um)
    return (11.510294 * np.exp(-1.376463 * diam_um) + 2.120503) * (u.uF / u.cm**2)

cell.paint(
    soma | dend,
    mech.CableProperty(
        resting_potential=LEAK_E_MV * u.mV,
        membrane_capacitance=pc24_cm,
        axial_resistivity=RA_OHM_CM * (u.ohm * u.cm),
    ),
)
```

And for channel parameters:

```python
def nav_by_distance(ctx):
    d_um = ctx.path_distance_from_soma.to_decimal(u.um)
    return np.where(d_um < 100.0, 0.02, 0.005) * (u.siemens / u.cm**2)

cell.paint(dend, mech.Channel("Nav1p6_MA2024_PC", g_max=nav_by_distance))
```

### Notes

- Define a small spatial context object for callable evaluation.
- Useful context fields include `branch_id`, `branch_name`, `branch_type`,
  `cv_id`, `prox`, `dist`, `midpoint`, `length`, `area`, `diam_mid`,
  `diam_arc_mean`, `radius_mid`, `path_distance_to_root`, and
  `path_distance_from_soma`.
- Cable fields need support during discretization because `CableProperty` is
  lowered directly into CV cable values.
- Density parameters can be evaluated during runtime state-buffer allocation.
- Callable return values should be validated for units and shape.
- Do not start this implementation until the callable context semantics are
  finalized, especially the exact meaning of soma-relative distance.
