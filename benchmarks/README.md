# Performance benchmarks

Performance workflows measure compilation, steady execution, memory and scaling.
Keep numerical protocols fixed and synchronize device execution before recording time.
Accuracy studies live in [validation](../validation/README.md).

- [Geometry initialization](performance/geometry_initialization/README.md): initialization,
  first JIT and fixed-geometry population execution timing.
- [Gradient scaling](performance/optim_gradient_scaling/README.md): BPTT/RTRL
  state, parameter, time, batch and seed sweeps.
- [Parameter fitting](performance/parameter_fitting/README.md): historical capacity and training-throughput evidence.
- [Simulator comparison](performance/simulator_compare/README.md): BrainCell,
  NEURON and Jaxley under the same model and stimulus.
- [Sampling](performance/sampling/README.md): connection and continuous-region sampling.
- [Solvers](performance/solvers/README.md): isolated DHS solver measurements.
- [Synapse events](performance/synapse_events/README.md): NetStim scaling,
  schedule lookup, synapse aggregation and CPU/GPU timing.
- [Profiling](profiling/README.md): construction phases, XPlane and device traces.

```bash
python -m benchmarks.performance.optim_gradient_scaling.benchmark --help
python -m benchmarks.profiling.profile_simulation --help
```

GPU and simulator requirements are documented with each workflow.

## Before running a benchmark

**List every count the run will execute and have the list confirmed before starting it.** Script defaults, the commands in a README and the protocol of an earlier run are not an authorization. This holds whether the measurement is started from a script, a notebook or pytest; a "trial run" is a run.

Read the driver first: count its first executions, inner loops and extra validation runs, not only the `rounds`, `warmup` and `repeat` arguments. Expand the list by device, configuration and method; identical combinations may be merged if their number is stated.

| Item | What to state |
| --- | --- |
| Workload | Device, model size, simulated duration, dt and steps per full simulation |
| Methods | Every method and baseline; whether a shared baseline is measured again in each candidate's process |
| Rounds and processes | Independent rounds, processes per round, total processes, serial or parallel |
| First execution and warm-up | First (compiling) executions and extra warm-ups per method; whether the first execution counts as a warm-up |
| Timed repeats | Timed runs per method per process, what the timer covers, and how the device is synchronized |
| Extra runs | Trajectory or reset checks, profiling, diagnostics and trial runs; a reset call and the full simulation after it are listed separately |
| Totals | Method results, timed runs, and model executions including first runs, warm-ups and checks |
| Time and failures | Expected wall time, total budget, per-process timeout, and whether retries are included (their scope and limit) |

Keep these counts apart; never write "run once":

- **round**: one independent repetition of the whole configuration set, possibly several processes;
- **process**: one program start, which may measure several methods;
- **method result**: one method's timings on one device, configuration, round and process;
- **full simulation**: the model advanced from its initial state for the planned number of steps; first runs, warm-ups and checks can each be one;
- **step**: one dt advance.

For example, a method with one first execution, W extra warm-ups, R timed runs and V checking simulations executes `1 + W + R + V` full simulations per process (fewer if the first execution is one of the W). The example explains the arithmetic; it authorizes no counts.

Choose the counts by purpose. A rough comparison of methods that differ a lot needs one independent round. Several rounds are for close comparisons or a formal evaluation, when asked for, and their number is still confirmed. One measurement supports a preliminary comparison, not a claim of stability across processes.

Once the list is confirmed:

1. Pass the confirmed counts as explicit arguments and check that the script's inner runs match the list.
2. Run an unchanged, already confirmed protocol without asking again. Adding a device, configuration, method, round, warm-up, repeat or check needs a new confirmation.
3. On failure, keep the logs and finished results. Retry only within the confirmed retries; otherwise explain the failure and propose the scope, count and budget of a rerun.
4. Stop at the confirmed timeout or budget and save what is unfinished.
5. Report the counts actually executed and any deviation from the list, with first runs, warm-ups, steady timings and checks reported separately.

Reading source, inspecting existing results and static checks run no measurement and need no confirmation.

## README, results and artifacts

| Location | Holds |
| --- | --- |
| The experiment's `README.md` | The public entry point: the goal, the question compared, the scripts and their roles, dependencies, parameters and how to run them. A new user can understand and run the experiment without existing artifacts. |
| The experiment's `results/` | Git-tracked, reviewed measurement summaries: the authoritative record of conditions and data. |
| The experiment's `artifacts/` | Ignored raw JSON, logs, traces, source snapshots and generated reports. |

**Keep READMEs environment-neutral.** Relative script names, repository paths, generic output locations and parameter placeholders are fine; personal absolute paths, a specific interpreter or Conda environment, hardware models, machine configuration and GPU indices are not. Device choice is a parameter. The commands, environment and counts of a particular run belong in its results summary.

**A results summary reads on its own.** It records the date, CPU/GPU models and relevant hardware, software versions, the measured code version (and any uncommitted changes), precision, model size, algorithm configuration, what the timer covers, the rounds, warm-ups and repeats actually run, the main data, correctness checks, failures or exclusions, and limitations. Conclusions are limited to the conditions measured; theory and measurement are kept apart, and a local gain is not written as a general performance guarantee.

Raw artifacts may exist only on the runner's machine. The summary states whether they are available and may add a run identifier, relative directory or source hash; missing historical details are marked as not recorded, not reconstructed. Key data and evidence never live only in a local file. A figure needed to understand the summary is tracked in `results/`.

Report scripts write derived reports into the input artifacts directory by default and never overwrite a maintained README or results summary; tracked summaries are updated after review. Raw samples, failure records and historical conditions are kept; tidying a document does not reinterpret them as a new measurement.

When a method is superseded, its benchmark may stop being maintained, be archived, or be removed in a clearly scoped cleanup. Kept historical results state the version measured, the scope and the superseded status, and link the successor. Historical data is not rewritten for a new implementation, and an old script is not promised to run on the current version.
