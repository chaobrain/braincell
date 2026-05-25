# First Steps

This page walks you through your **first complete `braincell` simulation**: a
classic Hodgkin–Huxley neuron that we stimulate with a constant current and
watch spike. It takes only a few lines and introduces the core workflow you
will reuse everywhere.

```{tip}
Every physical quantity in `braincell` **must** carry an explicit unit from
[`brainunit`](https://github.com/chaobrain/brainunit). A bare number like
`10` is rejected — write `10 * u.nA` instead. See {doc}`../concepts/units`.
```

## 1. Define the neuron

A single-compartment neuron is a subclass of
{class}`braincell.SingleCompartment`. Inside `__init__` you attach **ion
species** and the **channels** that carry their currents:

```python
import brainstate
import brainunit as u
import braincell


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='exp_euler'):
        super().__init__(size, solver=solver)

        # Sodium: fixed reversal potential + the HH sodium channel
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        # Potassium: fixed reversal potential + the HH potassium channel
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        # Passive leak
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV,
                                       g_max=0.03 * (u.mS / u.cm ** 2))
```

A few things to notice:

- `size` is the number of independent neurons — `braincell` is vectorized, so
  `HH(100)` simulates 100 neurons at once.
- Channels are *added to* an ion (`self.na.add(...)`) because their current
  depends on that ion's reversal potential. The keyword (`INa`, `IK`) names the
  channel on the cell.
- `solver` selects the numerical integrator by name. See
  {doc}`../integration/index` for the full list.

## 2. Initialize state

`braincell` models are stateful. Allocate and initialize the state (membrane
voltage, gating variables, …) before simulating:

```python
hh = HH(1, solver='ind_exp_euler')
hh.init_state()
```

## 3. Step the simulation

Time stepping is driven by `brainstate`. Each step injects a stimulus current
and advances the model by one time step `dt`; we collect the membrane voltage:

```python
def step(t):
    with brainstate.environ.context(t=t):
        hh.update(10. * u.nA / u.cm ** 2)   # injected current density
    return hh.V.value


with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())
    voltages = brainstate.transform.for_loop(step, times)
```

`brainstate.transform.for_loop` compiles the loop with JAX, so the whole 100 ms
run executes as a single fused kernel.

## 4. Plot the result

```python
import matplotlib.pyplot as plt

plt.plot(times.to_decimal(u.ms), u.math.squeeze(voltages).to_decimal(u.mV))
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.show()
```

You should see a train of action potentials riding on the injected current.

## What you just did

```text
  define cell  ──▶  init_state()  ──▶  for_loop(step, times)  ──▶  plot V
  (ions +                            (advance dt each step,
   channels)                          collect a trace)
```

This is the canonical single-compartment loop. The same `update`/`for_loop`
pattern reappears in every single-compartment example.

## Next steps

- **Understand each piece** — read {doc}`../concepts/cells`,
  {doc}`../concepts/ions_channels`, and {doc}`../concepts/units`.
- **Explore more single-compartment models** — the
  {doc}`../single_compartment/index` tutorials cover calcium dynamics,
  f–I curves, spike-frequency adaptation, and full thalamic models.
- **Add geometry** — when an isopotential point is not enough, move to
  morphological cells in {doc}`../multi_compartment/index`.
- **Choose a solver** — for stiff channel kinetics, see
  {doc}`../integration/index`.
```
