# Understanding NMODL from Scratch: NEURON 8.2.x, 9.0 and Arbor through `hh.mod`

This note is not a syntax manual but a study note that "builds the mental model first, then looks at the details". Its goal is to separate the things that are easiest to confuse:

- what a `mod` file actually describes
- why each block exists
- how `PARAMETER`, `ASSIGNED` and `STATE` really differ
- when `INITIAL`, `BREAKPOINT` and `DERIVATIVE` each run at run time
- whether the differences between `NEURON 8.2.x`, `9.0` and `Arbor` are "different documentation wording" or "really different semantics"

## 1. Remember one overall picture first

Read a `mod` file as four layers:

1. External interface
   - what the mechanism is called
   - whether it is a density mechanism or a point mechanism
   - which ions and currents it reads and writes
2. Variable classification
   - which quantities the user sets
   - which quantities are computed directly at run time
   - which quantities are states that must be integrated
3. Initialization
   - what the states are set to when the simulation starts
4. Per-step update
   - in each time step, what is computed first, what is integrated next, and what current is output last

In NMODL the most common pieces are:

- `NEURON` / `UNITS`
- `PARAMETER` / `ASSIGNED` / `STATE`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- the optional `PROCEDURE` / `FUNCTION`
- `NET_RECEIVE`, common in point mechanisms

If you first read these blocks as a "division of responsibilities", they stop looking like a pile of unrelated syntax.

## 2. Start with the simplest case: a leak mechanism without states

Many people start with `hh.mod` and get put off by `m/h/n`, `rates()` and `SOLVE`. In fact the smallest channel can have no `STATE` at all.

```mod
NEURON {
    SUFFIX pas_simple
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

PARAMETER {
    g = 0.001 (S/cm2)
    e = -65 (mV)
}

ASSIGNED {
    v (mV)
    i (mA/cm2)
}

BREAKPOINT {
    i = g * (v - e)
}
```

This example does one thing: it computes the leak current by Ohm's law.

Block by block:

- `NEURON`
  - `SUFFIX pas_simple` declares a density mechanism, named `pas_simple` once inserted into a section
  - `NONSPECIFIC_CURRENT i` declares that the mechanism contributes a non-specific transmembrane current `i`
  - `RANGE g, e` lets every segment have its own `g` and `e`
- `PARAMETER`
  - `g` and `e` are provided or changed by the user
- `ASSIGNED`
  - `v` is the membrane potential, supplied by the simulator
  - `i` is the current computed at the present time; it is not integrated
- `BREAKPOINT`
  - evaluates the present current in every time step

There is no `STATE` here because:

- `g` and `e` are not dynamically evolving states
- `i` is computed directly from `v` at the present time
- no variable "has to be updated over time by a differential equation"

This is the first key to understanding NMODL:

`STATE` does not mean "important variable"; it means "variable advanced by the integrator".

## 3. Step two: why `hh.mod` needs more blocks

The repository's sample file [`hh.mod`](../mod_files/hh.mod) is a simplified Hodgkin-Huxley mechanism. Compared with the leak, it adds only one core complication:

`m`, `h` and `n` are not constants but gating variables that change over time.

### 3.1 The full structure first

```mod
TITLE Simplified Hodgkin-Huxley example for AST and IR extraction

NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    RANGE gnabar, gkbar, gl, el
    GLOBAL minf, hinf, ninf, mtau, htau, ntau
    NONSPECIFIC_CURRENT il
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gnabar = 0.12 (S/cm2)
    gkbar = 0.036 (S/cm2)
    gl = 0.0003 (S/cm2)
    el = -54.3 (mV)
}

STATE {
    m
    h
    n
}

ASSIGNED {
    v (mV)
    ena (mV)
    ek (mV)
    ina (mA/cm2)
    ik (mA/cm2)
    il (mA/cm2)
    minf
    hinf
    ninf
    mtau (ms)
    htau (ms)
    ntau (ms)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
    n = ninf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h * (v - ena)
    ik = gkbar * n*n*n*n * (v - ek)
    il = gl * (v - el)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    n' = (ninf - n) / ntau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v + 40) / 10))
    hinf = 1 / (1 + exp((v + 62) / 10))
    ninf = 1 / (1 + exp(-(v + 53) / 16))
    mtau = 0.2
    htau = 1
    ntau = 1
}
```

### 3.2 What each block does

#### `NEURON`

This is where the mechanism "declares its interface to the simulator".

- `SUFFIX hh`
  - declares a density mechanism
- `USEION na READ ena WRITE ina`
  - the mechanism reads the sodium reversal potential `ena`
  - and writes the sodium current `ina`
- `USEION k READ ek WRITE ik`
  - reads the potassium reversal potential `ek`
  - writes the potassium current `ik`
- `RANGE gnabar, gkbar, gl, el`
  - these quantities can vary along space
- `GLOBAL minf, hinf, ninf, mtau, htau, ntau`
  - in NEURON these quantities are shared per mechanism
- `NONSPECIFIC_CURRENT il`
  - `il` is not the current of any particular ion

Rule of thumb:

- `SUFFIX`/`POINT_PROCESS` decide the "mechanism type"
- `USEION`/`NONSPECIFIC_CURRENT` decide "how the mechanism is wired to currents and ions"
- `RANGE`/`GLOBAL` decide "whether an exposed variable is stored per segment or shared per mechanism"

#### `UNITS`

This block declares units and helps with unit checking.

In NEURON it mainly serves readability and unit-consistency checks.  
In Arbor the units are parsed, but most conversion semantics are not applied, so in Arbor it is closer to "an explanatory note + agreed units".

#### `PARAMETER`

This holds the "parameters the user sets":

- `gnabar`
- `gkbar`
- `gl`
- `el`

They are the knobs of the model, not states.

A handy test:

- imagine you are running a parameter sweep
- a quantity you would deliberately vary is usually a `PARAMETER` first

#### `STATE`

This holds the "dynamic variables advanced by the integrator".

In HH these are:

- `m`
- `h`
- `n`

Why are they `STATE`?

Because you do not recompute `m/h/n` from an algebraic expression every step; you update them from their old values to new values according to the differential equations:

```text
m' = ...
h' = ...
n' = ...
```

This is the second key:

- a `PARAMETER` is a value you give
- an `ASSIGNED` is a value you compute
- a `STATE` is a value the solver advances

#### `ASSIGNED`

This holds quantities "computed by assignment at run time that need no integration".

In this `hh.mod` they include:

- quantities from the simulator: `v`, `ena`, `ek`
- output currents: `ina`, `ik`, `il`
- intermediate results: `minf`, `hinf`, `ninf`, `mtau`, `htau`, `ntau`

Why are `minf` and `mtau` not `STATE`?

Because they are functions of the present voltage `v`. Knowing the present `v` is enough to compute them directly; they need not be stored as "unknowns advanced by integration over time".

#### `INITIAL`

This block runs once when the simulation is initialized.

It does two things here:

```mod
INITIAL {
    rates(v)
    m = minf
    h = hinf
    n = ninf
}
```

It first computes the steady-state gating values from the initial voltage, then initializes the states to those steady-state values.

This step matters, because without initialization:

- `m/h/n` may default to 0
- the model shows an unreasonable transient at the start

Rule of thumb:

- `INITIAL` is responsible for "putting the system into a reasonable initial state when the simulation starts"

#### `BREAKPOINT`

This is the most easily misunderstood block. Read it first as:

"In every time step, when the simulator needs this mechanism's present current, it runs this block."

```mod
BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h * (v - ena)
    ik = gkbar * n*n*n*n * (v - ek)
    il = gl * (v - el)
}
```

It has two parts:

- `SOLVE states METHOD cnexp`
  - calls the `DERIVATIVE states` block below, advancing `m/h/n` with the `cnexp` method
- three current expressions
  - compute `ina/ik/il` from the present states and voltage

The most common beginner mistake is to treat `BREAKPOINT` as "a main function where any logic goes".  
A safer reading is:

- `BREAKPOINT` is mainly responsible for "the observables at the present time", especially currents
- what actually defines how states change is `DERIVATIVE` or `KINETIC`

#### `DERIVATIVE`

This block writes the differential equations of the states.

```mod
DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    n' = (ninf - n) / ntau
}
```

Here it expresses the classic HH form:

```text
x' = (x_inf(v) - x) / tau_x(v)
```

That is:

- first compute the present target steady state `x_inf` from `v`
- then approach that target with the time constant `tau_x`

So:

- `DERIVATIVE` defines "how the states evolve"
- `SOLVE` decides "which numerical method evolves them"

#### `PROCEDURE rates(...)`

This block is not part of the mechanism's life cycle; it is a "helper subroutine".

It only factors out repeated logic:

- given `v`
- compute `minf/hinf/ninf`
- compute `mtau/htau/ntau`

so that both `INITIAL` and `DERIVATIVE` can reuse it.

So the third key is:

- `INITIAL`, `BREAKPOINT` and `DERIVATIVE` are "scheduling blocks"
- `PROCEDURE`/`FUNCTION` are "helper computation blocks"

## 4. `hh.mod` in one sentence

If you can remember only one sentence, remember this:

`hh.mod` first declares that it produces a sodium current, a potassium current and a leak current; it then initializes the gating variables `m/h/n` as states; and in every step it advances those states from the voltage and computes the three currents from them.

Once you read it this way, the blocks no longer look scattered.

## 5. How to think about the run-time order

For a beginner, this simplified order is enough:

1. `INITIAL` runs at `finitialize()`
2. `BREAKPOINT` runs in every time step
3. the `SOLVE` in `BREAKPOINT` calls the corresponding `DERIVATIVE` or `KINETIC`
4. `NET_RECEIVE` runs when a point mechanism receives an event

More precisely, when and how often NEURON calls `BREAKPOINT` within one time step follows from its numerical algorithm, in particular `fadvance()`, `secondorder` and the conductance derivative. While learning, you need not dig into that implementation detail first; just remember:

- `BREAKPOINT` is not "an ordinary function that runs exactly once"
- do not put side effects that depend on the call count into `BREAKPOINT`
- keep current and conductance expressions "purely computable from the present states and voltage"

## 6. Why a point synapse adds `NET_RECEIVE`

An HH channel has no event input, so it needs no `NET_RECEIVE`.  
A synapse is often "on a spike event, the conductance jumps instantly and then decays exponentially", so it adds an event block.

A minimal exponential synapse can be written as:

```mod
NEURON {
    POINT_PROCESS expsyn_simple
    RANGE tau, e
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    tau = 2 (ms)
    e = 0 (mV)
}

STATE {
    g (uS)
}

ASSIGNED {
    v (mV)
    i (nA)
}

INITIAL {
    g = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g * (v - e)
}

DERIVATIVE state {
    g' = -g / tau
}

NET_RECEIVE(weight) {
    g = g + weight
}
```

How to read it:

- `POINT_PROCESS` means it is not a density mechanism distributed along the cable but a point object placed at one location
- `g` is a state that decays exponentially between events
- on every event, `NET_RECEIVE(weight)` increments `g` instantly

So:

- channel mechanisms typically have `BREAKPOINT + DERIVATIVE`
- event-driven synapses typically have `BREAKPOINT + DERIVATIVE + NET_RECEIVE`

## 7. What actually differs between `NEURON 8.2.x` and `9.0`

If you are confused right now, hold on to the single most important statement:

The basic learning semantics of the core blocks `PARAMETER`, `ASSIGNED`, `STATE`, `INITIAL`, `BREAKPOINT` and `DERIVATIVE` did not change fundamentally between NEURON 8.2.x and 9.0.

What really makes people think "the syntax changed" is mainly the following.

### 7.1 The biggest change: from 9.0 on, NMODL generates C++

From `NEURON >= 9.0` on, a translated MOD file produces C++ instead of the old C.  
The official migration guide also states clearly that:

- the numerical methods themselves did not change because of this
- most ordinary MOD files need no changes
- the files that really tend to break are those containing `VERBATIM`

So if the HH-style teaching example you are reading has no `VERBATIM`, you usually need not treat 8.2.x and 9.0 as two languages.

### 7.2 Documentation location and wording changed, the basic concepts did not

You will see that:

- the 8.2.x documentation lives under `python/modelspec/programmatic/mechanisms`
- the 9.0 documentation moved to `nmodl/language`

Many older documents also carry obvious historical baggage, for example:

- `PARAMETER` and `ASSIGNED` are described as "almost synonymous"
- old mechanism extensions such as `LOCAL`, `POINTER` and `VERBATIM` are discussed more cautiously in the new documentation

These can make it look as if "9.0 changed all block semantics", but they are not the main issue when you learn the basic blocks.

### 7.3 The most practical version check for beginners

Remember it this way:

- if you write an ordinary HH/ohmic/channel/synapse mechanism
  - the same understanding of the learning semantics applies
- if your MOD file uses `VERBATIM`
  - you must also check C++ compatibility for 9.0
- if it uses very old or very tricky constructs
  - check the documentation and the compiler behaviour more carefully

## 8. When Arbor is "the same" and when it is "different"

Arbor is best understood this way:

- for common HH-style mechanisms it is similar enough to NEURON to share one learning framework
- but Arbor is not "a fully compatible stand-in for NEURON"
- it supports a stricter subset and adds a few Arbor-specific extensions

### 8.1 What is nearly the same when learning the basic blocks

At the learning level, these can be read as having roughly the same responsibilities:

- `NEURON { SUFFIX ... }`
- `PARAMETER`
- `ASSIGNED`
- `STATE`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- `POINT_PROCESS`
- `NET_RECEIVE`

That is, if you are trying to understand:

- a leak mechanism
- an HH channel
- an exponential synapse

then the mental models of Arbor and NEURON are essentially the same.

### 8.2 The Arbor differences you do need to remember

#### Units

- NEURON's `UNITS` really takes part in unit checking and the unit database
- Arbor only parses units and ignores most conversion semantics

So:

- in Arbor, `UNITS` is closer to "for human readers + a constraint to write in the agreed units yourself"

#### Scope of special variables

Arbor requires special variables to be more explicit:

- `v`, `diam`, `area` and `celsius` must be placed in `PARAMETER` to be accessible
- in `PROCEDURE` / `FUNCTION`, non-`LOCAL` variables such as `v` must be passed explicitly as arguments

This differs from many older NEURON examples that "use `v` directly inside a procedure".

#### Stricter ion semantics

Arbor constrains `USEION` more explicitly than NEURON:

- `Xi/Xo/eX/iX` are exposed automatically
- they should no longer be placed in `PARAMETER`, `ASSIGNED` or `CONSTANT` by hand
- if a mechanism writes `Xi/Xo`, they should be `STATE` and initialized in `INITIAL`

#### A set of unsupported old features

Arbor's documentation explicitly lists as unsupported:

- `TABLE`
- `VERBATIM`
- `INDEPENDENT`
- `LOCAL` outside blocks
- loops / arrays / pointers
- `derivimplicit`

This is why "an old NEURON model compiles" does not mean "it compiles in Arbor as is".

### 8.3 Arbor also has some extensions NEURON lacks

Arbor additionally supports, for example:

- `POST_EVENT`
- `VOLTAGE_PROCESS`
- `WHITE_NOISE`
- `v_peer`

So Arbor is neither "a cut-down NEURON" nor "an implementation with identical semantics"; it is a dialect whose "common core is similar, corner-case semantics are stricter, and which carries a few extensions of its own".

## 9. A practical three-column table

| Topic | NEURON 8.2.x | NEURON 9.0 | Arbor |
|---|---|---|---|
| Learning semantics of `PARAMETER/ASSIGNED/STATE` | as in traditional tutorials | essentially the same | essentially the same, but stricter for special and ion variables |
| Learning semantics of `INITIAL/BREAKPOINT/DERIVATIVE` | essentially the same | essentially the same | essentially the same |
| `VERBATIM` | available | watch C++ compatibility | not supported |
| `TABLE` | available | available | not supported |
| `POINTER` | available | available, but take more care over compatibility and implementation details | not supported |
| `UNITS` | really part of the unit system | really part of the unit system | parsed, but mostly no conversion |
| Using `v` directly in `PROCEDURE/FUNCTION` | common in old examples | still common | should usually be passed explicitly |
| Constraints after `USEION` exposes ion variables automatically | historical compatibility logic | historical compatibility logic | stricter |
| Basic mechanisms such as `hh` / `pas` / `expsyn` | directly understandable | directly understandable | also directly understandable |

## 10. How you should learn it now

If you feel lost, do not start from "the full set of all blocks"; follow this order instead:

1. Master the minimal leak first
   - why there is no `STATE`
   - why there is only `BREAKPOINT`
2. Then master `hh.mod`
   - why `m/h/n` are `STATE`
   - why `minf/mtau` are `ASSIGNED`
   - why `INITIAL` first sets the states to steady state
   - why `BREAKPOINT` contains `SOLVE`
3. Then look at the exponential synapse
   - why a point mechanism uses `POINT_PROCESS`
   - why `NET_RECEIVE` appears
4. Only then touch the advanced blocks
   - `KINETIC`
   - `POINTER`
   - `TABLE`
   - `VERBATIM`
   - `BEFORE/AFTER`
   - `WATCH`

That is, you do not need to "memorize every block" now; you only need to memorize this minimal table first:

| Block | Read it first as |
|---|---|
| `NEURON` | declares the interface to the simulator |
| `PARAMETER` | parameters given by the user |
| `ASSIGNED` | quantities computed directly |
| `STATE` | states advanced by integration |
| `INITIAL` | runs once at initialization |
| `BREAKPOINT` | computes the present outputs every step, especially currents |
| `DERIVATIVE` | defines how the states change |
| `PROCEDURE/FUNCTION` | helper computation |
| `NET_RECEIVE` | runs when an event arrives |

## 11. One more note on this repository's `hh.mod`

[`hh.mod`](../mod_files/hh.mod) is a simplified HH suited to teaching and to AST/IR extraction.

It does not try to reproduce every historical detail of NEURON's built-in `hh`; it keeps the most essential block relations:

- several ions through `USEION`
- `STATE m/h/n`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- `PROCEDURE rates`

Precisely because it is simplified, it is a better starting point for a first understanding than working through a complex old model directly.

## 12. References

- The repository's sample `hh.mod`:
  - [`examples/convert_mod/nmodl/mod_files/hh.mod`](../mod_files/hh.mod)
- Arbor NMODL rules:
  - <https://docs.arbor-sim.org/en/latest/fileformat/nmodl.html>
- Arbor NMODL tutorial:
  - <https://docs.arbor-sim.org/en/latest/tutorial/nmodl.html>
- NEURON 9.0 NMODL documentation:
  - <https://nrn.readthedocs.io/en/9.0.1/nmodl/language/nmodl.html>
- NEURON 8.2.x NMODL documentation:
  - <https://nrn.readthedocs.io/en/8.2.7/python/modelspec/programmatic/mechanisms/nmodl.html>
- NEURON 9.0 MOD migration guide:
  - <https://nrn.readthedocs.io/en/latest/guide/porting_mechanisms_to_cpp.html>

## 13. The conclusion in one sentence

Do not treat NMODL as "a syntax exercise with many blocks"; treat it as:

"Using a few fixed blocks to state clearly a mechanism's interface, states, initialization and per-step evolution."

Once you have mastered the leak, then `hh.mod`, then `NET_RECEIVE`, most of the confusion goes away on its own.
