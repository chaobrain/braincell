# Filter: spatially varying callable parameters

## Spatial callable parameters

Spatially varying cable and sampling-density parameters can be declared as
functions over morphology-backed contexts instead of manually looping over
branches.

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
    d_um = braincell.filter.metric.path_distance_from_soma(ctx).to_decimal(u.um)
    return np.where(d_um < 100.0, 0.02, 0.005) * (u.siemens / u.cm**2)

cell.paint(dend, mech.Channel("Nav1p6_MA2024_PC", g_max=nav_by_distance))
```

For a composable sampling density:

```python
def distal_density(ctx):
    distance = braincell.filter.metric.path_distance_from_soma(ctx)
    radius = braincell.filter.metric.radius(ctx)
    return u.math.exp(distance / (100 * u.um)) * u.math.exp(
        -0.5 * ((radius - 1 * u.um) / (0.25 * u.um)) ** 2
    )
```

### Context and distance contract

- `braincell.filter.metric` exposes the common `branch_x`, `radius`,
  `path_distance_from_soma`, and `position` surface for continuous sampling,
  CV, and Synapse contexts.
- Useful context fields include `branch_id`, `branch_name`, `branch_type`,
  `cv_id`, `prox`, `dist`, `midpoint`, `length`, `area`, `diam_mid`,
  `diam_arc_mean`, `radius_mid`, `path_distance_to_root`, and
  `path_distance_from_soma`.
- `path_distance_from_soma` is continuous in `branch_x` and means shortest tree
  distance to the union of all soma branches. Every point on those branches is
  zero. If no soma exists, every point on the root branch is zero instead.
- `position` is the interpolated morphology-local 3-D position and raises when
  the morphology does not provide full point geometry.
- A sampling density returns a non-negative, finite, dimensionless scalar or an
  array matching the context's `branch_x` shape.
- The string-field `density.exponential(...)` and `density.gaussian(...)`
  helpers are deprecated. `density.spatial_gaussian(...)` remains available.
