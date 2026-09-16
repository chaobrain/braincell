# Geometry initialization results

- [CPU/GPU array preparation](arrays-cpu-gpu.md): measured initialization and first-use costs, including the failed CPU baseline and its repaired run.
- [128-CV population comparison](population-128.md): population sizes 1/10/100/1000, fixed shared geometry and warmed forward execution.

The reports preserve their historical environment, source identifiers, protocols
and limitations. `arrays_cpu_gpu/` and `population_128/` retain reviewed tables,
figures and checks beside those reports. Raw JSON/NPZ inputs remain in the
workflow's ignored `artifacts/`. Report generators default to an input-adjacent
output directory there; explicit output paths are used for reviewed material.
These historical measurements do not describe nonlinear-pattern or dynamic
geometry/RTRL training performance.
