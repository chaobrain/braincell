# Discretization

A morphology is *continuous* cable, but a simulation works on a *finite* set of
state variables. **Discretization** is the step that divides the continuous
geometry into discrete **control volumes** (CVs), each treated as a single
isopotential compartment. This concept applies only to multi-compartment
{class}`~braincell.Cell` models.

## Control volumes

A {class}`braincell.CV` is one isopotential chunk of cable. The collection of
CVs for a cell forms a {class}`~braincell.CVTree` whose connectivity mirrors the
morphology. Each CV holds the membrane voltage and the channel states for its
patch of membrane; adjacent CVs exchange current through axial resistance.

```text
   one branch                          discretized into 3 CVs
   ●━━━━━━━━━━━━━━━━━━━━━●     ──▶      ●━━━━●━━━━●━━━━●
                                        CV0  CV1  CV2
```

The finer the discretization (more, shorter CVs), the more accurately voltage
gradients along a dendrite are resolved — at the cost of more state and
computation. Choosing the resolution is the job of a **CV policy**.

## CV policies

A {class}`braincell.CVPolicy` decides how many CVs each branch receives.
`braincell` provides several, which you can mix:

```{list-table}
:header-rows: 1
:widths: 32 68

* - Policy
  - Rule
* - {class}`~braincell.CVPerBranch`
  - a fixed number of CVs per branch (e.g. one).
* - {class}`~braincell.MaxCVLen`
  - cap each CV's physical length; long branches get more CVs.
* - {class}`~braincell.DLambda`
  - the classic **d-λ rule**: size CVs as a fraction of the AC length constant,
    so electrically long cable is discretized more finely.
* - {class}`~braincell.CVPolicyByTypeRule`
  - apply different rules to different branch types (e.g. fine soma, coarse axon).
* - {class}`~braincell.CompositeByTypePolicy`
  - compose several by-type policies into one.
```

```python
import brainunit as u
import braincell

# the d-lambda rule at 100 Hz
policy = braincell.DLambda(frequency=100. * u.Hz)

# or: no CV longer than 20 microns
policy = braincell.MaxCVLen(20. * u.um)
```

```{tip}
The **d-λ rule** ({class}`~braincell.DLambda`) is the standard default in
detailed modeling: it puts compartments where the cable physics needs them and
spares them where it doesn't. Start there unless you have a reason not to.
```

## From CVs to runtime

Once the CVs are fixed, `braincell`'s runtime layer (`braincell._compute`)
builds an execution graph over them — a {class}`~braincell.Node`/
{class}`~braincell.NodeTree` of compute points — and assembles the frozen,
JAX-friendly state that the {doc}`integrator <integration>` advances. You do not
interact with this layer directly; it is what `paint`/`place` + a CV policy
compile into.

## See also

- {doc}`morphology` — the geometry being discretized.
- {doc}`integration` — advancing the discretized state in time.
- {doc}`../apis/braincell` — `CV`, `CVPolicy`, and policy classes.
