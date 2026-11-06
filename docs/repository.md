# Repository Organization

This page describes the responsibilities of top-level directories and provides their entry points.

- [braincell/](https://github.com/chaobrain/braincell/tree/main/braincell): library implementation and co-located tests, including experimental modules.
- [examples/](https://github.com/chaobrain/braincell/tree/main/examples): usage examples, tutorials, and task workflows.
- [validation/](https://github.com/chaobrain/braincell/blob/main/validation/README.md): numerical accuracy and correctness validation.
- [benchmarks/](https://github.com/chaobrain/braincell/blob/main/benchmarks/README.md): performance, resource usage, and scaling measurements. Full measurement reports are tracked in each experiment's `results/`; raw outputs belong in ignored `artifacts/` directories. Design documents cite measurement evidence in the relevant proposal or architecture document.
- [data/](https://github.com/chaobrain/braincell/blob/main/data/README.md): shared input data and reference materials.
- [docs/](https://github.com/chaobrain/braincell/tree/main/docs): user documentation, design documents, contributor guides, and historical records.

Update this page when top-level directory responsibilities change. Each directory's own documentation maintains its internal structure, interfaces, commands, and data details.
