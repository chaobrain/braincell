# Package marker for the cable neuron-compare suite.
#
# This exists so that `cable/tests/` imports as `cable.tests` rather than as a
# bare top-level `tests` package. `channel_no_conc/tests/` is also a package
# named `tests`; without a distinguishing parent the two collide in
# `sys.modules` and pytest aborts collection with "import file mismatch" for
# the basenames they share (`_helpers.py`, `test_dispatch.py`,
# `test_experiment_schema.py`, `test_workflow_api.py`).
