# Synapse comparisons

BrainCell / NEURON comparisons of postsynaptic mechanisms, event routing and
external event sources.

- `reference.py` and `reference_test.py`: `ExpSyn` and `Exp2Syn` against
  NEURON's built-in mechanisms on twelve scenarios (single events, trains,
  two events in one step, short and long time constants, excitatory,
  inhibitory, zero and strong driving force; one-segment passive cylinder,
  `dt` 0.025 ms, float64). The two sides differ by rounding only: V within
  1e-9 mV, g within 1e-12 uS, the current within 1e-10 nA. The test needs
  NEURON (built-in mechanisms only, no `nrnivmodl`) and skips without it;
  it is not run by CI yet. Run it with
  `pytest -q validation/neuron/synapse/reference_test.py`.
- `netcon_delay.py` and `netcon_delay_test.py`: an HH soma whose spike
  drives its own `ExpSyn` through a `Network` connection, against NEURON's
  `NetCon` with delays of 0, 1, 2, 3, 4 and 12 steps (`dt` 0.025 ms). Each
  side's V first differs from the unconnected soma in step m + 1 + k after
  its own crossing step m, and the recorded g is 0 at t_{m+1+k} and
  w exp(-dt / tau) one sample later on both sides (within 1e-12 uS; zero
  delay is not compared, BrainCell adds it at the end of the step). Needs
  NEURON and skips without it.
- `exp_syn_compare.ipynb`: plots one `ExpSyn` and one `Exp2Syn` case built
  with `reference.py`.
- `hh_2x2_neuron_compare.ipynb`: synaptic transmission in a small HH network.
- `netstim_heterogeneous_compare.ipynb`: heterogeneous NetStim, Connection
  weight and delay, and ExpSyn, end to end.

The analytic reference of the same scenarios (the conductance on the step
grid, event rounding and multiplicity, zero current at `e = E_leak`) is in
`braincell/synapse/exponential_test.py`; it needs no NEURON.

## Sampling phase

Sample k of a `Network.run` or `Cell.run` trace is the state at t = k dt,
before the events of that step boundary; an event rounded to step s shows
from sample s + 1. NEURON's recorded states (`ExpSyn.g`, `Exp2Syn.A` and `B`)
follow the same rule. NEURON's recorded point-process `i` and `Exp2Syn.g` are
`BREAKPOINT` assignments: they include the boundary's events, lag the states
by one sample, and `i` is outward-positive. Compare the conductance on the
states and the current as `g_nrn (e - V_nrn)`, BrainCell's inward-positive
convention; NEURON's own `i` differs by the whole jump of an `ExpSyn` event
on the step it arrives (0.065 nA for 0.001 uS at -65 mV).

## Event times exactly half a step from two boundaries

An event at t is delivered at the boundary nearest t / dt (delay included).
An exact half-step tie, or a step ratio within four ulp of one, goes to the
later step in BrainCell. NEURON's choice follows its accumulated `t`: with
`dt` 0.025 ms, events at 0.0125, 2.0125, 5.0375, 8.0125 and 12.0125 ms act on
steps 1, 81, 202, 321 and 481 in BrainCell and on steps 0, 81, 201, 320 and
481 in NEURON 9.0.2. The NEURON comparisons therefore avoid exact ties; the
analytic test fixes BrainCell's rule.

## NetStim randomness

NetStim's random numbers are not required to share NEURON's random stream.
`netstim_heterogeneous_compare.ipynb` first generates the event times in
BrainCell and then replays them event by event in NEURON, so it compares event
routing, delay, synaptic dynamics and membrane integration rather than two
random number generators. The NEURON side turns off the rate tables of the
built-in `hh` (`h.usetable_hh = 0`): `hh` interpolates its rates linearly on a
1 mV grid while BrainCell evaluates them exactly; with the tables on the two
sides differ by up to 0.165 mV, without them by 2.8e-7 mV.
