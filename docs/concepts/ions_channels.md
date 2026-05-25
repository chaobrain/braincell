# Ions & Channels

The electrical behavior of a neuron comes from **ion channels** — pores that
let specific **ion species** flow across the membrane. `braincell` models these
as two cooperating layers: ion species ({mod}`braincell.ion`) own a reversal
potential and (optionally) concentration dynamics, and channels
({mod}`braincell.channel`) carry current driven by that reversal potential.

## Ion species

An ion species tracks the reversal potential `E` and, in detailed models, the
intra/extracellular concentration. The families in {mod}`braincell.ion`:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Class family
  - Behavior
* - `*Fixed` (e.g. {class}`~braincell.ion.SodiumFixed`, {class}`~braincell.ion.PotassiumFixed`, {class}`~braincell.ion.CalciumFixed`)
  - constant reversal potential `E` — the simplest, fastest choice.
* - {class}`~braincell.ion.CalciumDetailed`, {class}`~braincell.ion.CalciumFirstOrder`
  - calcium with concentration dynamics (buffering, pumps, diffusion), so `E`
    follows the Nernst equation as concentration changes.
* - `*InitNernst`
  - reversal potential initialized from the Nernst equation.
```

```python
import brainunit as u
import braincell

# constant sodium reversal at +50 mV
na = braincell.ion.SodiumFixed(size, E=50. * u.mV)

# calcium with first-order concentration decay
ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM,
                                   tau=10. * u.ms, d=0.5 * u.um)
```

## Channels

A channel produces a current as a function of voltage (and possibly an ion
concentration). {mod}`braincell.channel` ships a large, literature-derived
library, named by `<current>_<source><year>` convention:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Group
  - Examples
* - Sodium
  - {class}`~braincell.channel.Na_HH1952`, {class}`~braincell.channel.Na_Ba2002`
* - Potassium (delayed rectifier, A-type, M-type, Kv, Kir, …)
  - {class}`~braincell.channel.K_HH1952`, {class}`~braincell.channel.KDR_Ba2002`
* - Calcium (L/N/T/P-type, Cav families)
  - {class}`~braincell.channel.CaL_IS2008`, {class}`~braincell.channel.CaT_HM1992`
* - Potassium–calcium (Ca-activated K)
  - {class}`~braincell.channel.AHP_De1994`, `Kca1p1_*`, `Kca2p2_*`
* - Hyperpolarization-activated (H-current)
  - {class}`~braincell.channel.HCN_HM1992`
* - Leak
  - {class}`~braincell.channel.IL`, {class}`~braincell.channel.LeakageChannel`
```

See the {doc}`../apis/braincell.channel` reference for the complete list.

## How ions and channels connect

Most channels carry a specific ion's current, so they are **added to that ion**.
The ion provides the reversal potential the channel's current depends on:

```python
na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
na.add(INa=braincell.channel.Na_HH1952(size))   # INa uses na.E
```

For a multi-compartment {class}`~braincell.Cell` you express the same
relationship declaratively, naming the channel as a string in a
{class}`~braincell.mech.Channel` (see {doc}`mechanisms`).

## Channels that depend on two ions: `MixIons`

Some channels (calcium-activated potassium currents, for example) depend on
*two* ion species at once — a potassium reversal potential **and** an
intracellular calcium concentration. {class}`braincell.MixIons` bundles the
relevant ions so such a channel can read both:

```python
k  = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
ca = braincell.ion.CalciumDetailed(size, ...)

kca = braincell.MixIons(k, ca)
kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.3 * (u.mS / u.cm**2)))
```

The convenience function {func}`braincell.mix_ions` builds the same combination.

## The registry

Channels and ions self-register at import time via the
`@register_channel` / `@register_ion` / `@register_synapse` decorators in
{mod}`braincell.mech`. That registry is what lets you refer to a channel by
string name (`mech.Channel("Na_Ba2002", ...)`). To add your own, see
{doc}`../developer/extending`.

## See also

- {doc}`mechanisms` — declaring channels/ions on a multi-compartment cell.
- {doc}`../apis/braincell.ion` and {doc}`../apis/braincell.channel` — full lists.
- {doc}`../single_compartment/index` — building neurons from ions and channels.
