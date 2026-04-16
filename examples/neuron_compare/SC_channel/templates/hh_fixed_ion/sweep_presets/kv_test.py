"""Sweep preset for the ``kv_test`` channel."""

channel_id = "kv_test"

description = "Baseline sweep preset for the minimal Kv validation case."

base_case_overrides = {
    "simulation": {
        "v_init_mV": -65.0,
        "temperature_celsius": 25.0,
    },
    "stimulus": {
        "delay_ms": 0.0,
        "dur_ms": 100.0,
        "amp_nA": 0.01,
    },
    "channel_overrides": {
        "g_max_S_cm2": 0.0,
        "v12_mV": 25.0,
        "q": 9.0,
    },
}

sweep_axes = {
    "simulation.v_init_mV": [-80.0, -65.0, -50.0],
    "stimulus.amp_nA": [-0.02, 0.0, 0.02],
    "channel_overrides.g_max_S_cm2": [0.0, 0.001, 0.01],
}

notes = [
    "This preset is intended as the first sweepable HH + fixed-ion example.",
]
