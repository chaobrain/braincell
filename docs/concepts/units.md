# Units

In `braincell`, **every physical quantity carries an explicit unit.** This is
not optional decoration — it is enforced. Passing a bare number where a quantity
is expected raises a `TypeError`. Units come from
[`brainunit`](https://github.com/chaobrain/brainunit), which `braincell`
re-exports nothing of; you import it directly as `u`.

```python
import brainunit as u
```

## Why mandatory units?

Biophysical models mix quantities across many orders of magnitude — millivolts,
nanoamps, microfarads per square centimeter. Silent unit mismatches are one of
the most common and hardest-to-find bugs in neural modeling. By requiring units
everywhere, `braincell`:

- catches dimensional errors *immediately*, at construction time;
- performs automatic unit conversion (adding `10 * u.mV` to `0.01 * u.V` just
  works);
- makes model code self-documenting — you can see at a glance that `g_max` is a
  conductance density.

```{important}
The rejection of bare numbers happens in `normalize_param` inside
`braincell._misc`. If you see `TypeError: ... expected a quantity`, you almost
certainly passed a plain `float` or `int` where a united quantity was required.
```

## Creating quantities

Multiply a number (or array) by a unit:

```python
import numpy as np
import jax.numpy as jnp
import brainunit as u

v_rest = -65.0 * u.mV                 # scalar
dt     = 0.1 * u.ms

lengths = np.array([10.5, 20.0]) * u.um   # numpy array
coords  = jnp.zeros((10, 3)) * u.um       # JAX array
radii   = [2.0, 3.0, 4.0] * u.um          # list
```

## Common units

| Category | Quantity | Units |
|----------|----------|-------|
| Electrical | Voltage | `u.V`, `u.mV` |
| Electrical | Current | `u.A`, `u.mA`, `u.uA`, `u.nA`, `u.pA` |
| Electrical | Conductance | `u.S`, `u.mS`, `u.uS`, `u.nS`, `u.pS` |
| Electrical | Resistance | `u.ohm`, `u.kohm`, `u.Mohm` |
| Electrical | Capacitance | `u.F`, `u.uF`, `u.nF`, `u.pF` |
| Space/Time | Length | `u.m`, `u.cm`, `u.mm`, `u.um` |
| Space/Time | Time | `u.s`, `u.ms` |
| Substance/Temp | Concentration | `u.M`, `u.mM` |
| Substance/Temp | Temperature | `u.kelvin`, `u.celsius` |

Prefixes follow SI: `m` (milli), `u` (micro), `n` (nano), `p` (pico),
`k` (kilo), `M` (mega).

## Arithmetic and dimensional analysis

Units propagate through arithmetic and combine into compound dimensions:

```python
# Conductance density × voltage → current density (Ohm's law)
g = 50 * u.nS / u.cm**2
I = g * (-65 * u.mV)

# Geometry: surface area of a cylinder
radius, length = 1.0 * u.um, 10.0 * u.um
area = 2 * np.pi * radius * length     # → u.um**2

# Membrane capacitance density
cm = 1.0 * u.uF / u.cm**2
```

## Extracting raw values

When you need a plain number — for plotting, say — strip the unit explicitly:

```python
from brainunit import get_unit, get_mantissa

a = jnp.array([1, 2, 3]) * u.mV

get_unit(a)        # u.mV
get_mantissa(a)    # array([1, 2, 3])

a.to_decimal(u.V)  # array([0.001, 0.002, 0.003]) — float in target unit
a / u.mV           # array([1., 2., 3.])           — divide the unit out
a.to(u.V)          # 0.001 * u.V ...               — convert, keep unit
```

```{tip}
For plotting, `value.to_decimal(u.ms)` (or whatever unit you want on the axis)
gives you a clean float array. You will see this pattern throughout the
examples.
```

## See also

- {doc}`mechanisms` — every mechanism parameter is a united quantity.
- {doc}`../troubleshooting` — the most common unit-related errors and fixes.
