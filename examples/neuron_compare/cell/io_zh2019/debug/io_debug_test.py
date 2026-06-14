from __future__ import annotations

import unittest

import brainunit as u
import numpy as np
from neuron import h

from braincell import mech
from braincell.filter import at

from ..io_braincell import IO as FormalBrainCellIO
from ..io_neuron import IO as FormalNeuronIO
from ..parameters import load_io19_params as load_formal_io19_params
from .io_braincell_debug import IO as BrainCellIO
from .io_neuron_debug import IO as NeuronIO
from .io_parameters import IOConfig, IOToggles, load_io19_params


class IOZH2019DebugBuildTest(unittest.TestCase):
    def test_formal_templates_build_single_soma(self) -> None:
        params = load_formal_io19_params()
        neuron_io = FormalNeuronIO(params=params).build()
        braincell_io = FormalBrainCellIO(params=params).build()
        try:
            self.assertEqual(len(neuron_io.sections), 1)
            self.assertIsNotNone(neuron_io.root_soma)
            self.assertEqual(len(braincell_io.morph.branches), 1)
            mechanism_names = {density.instance_name for cv in braincell_io.cell.cvs for density in cv.density_mech}
            self.assertIn("IL_soma", mechanism_names)
            self.assertIn("Na_soma", mechanism_names)
            self.assertIn("Kdr_soma", mechanism_names)
            self.assertIn("Ca_soma", mechanism_names)
            self.assertIn("HCN_soma", mechanism_names)
        finally:
            neuron_io.cleanup()

    def test_neuron_and_braincell_build_leak_only_single_soma(self) -> None:
        config = IOConfig(toggles=IOToggles(leak=True, na=False, kdr=False, ca=False, hcn=False))
        neuron_io = NeuronIO(config=config).build()
        braincell_io = BrainCellIO(config=config).build()
        try:
            for summary in (neuron_io.summary(), braincell_io.summary()):
                self.assertTrue(summary["manual_soma"])
                self.assertEqual(summary["branch_counts"], {"n_soma": 1, "n_total": 1})
                self.assertEqual(summary["compartment_counts"], {"n_total_nseg": 1})
                self.assertEqual(summary["enabled_mechanisms"], {"soma": ["leak"]})
        finally:
            neuron_io.cleanup()

    def test_braincell_all_channels_builds_expected_mechanisms(self) -> None:
        cell = BrainCellIO().build()
        summary = cell.summary()
        self.assertEqual(summary["enabled_mechanisms"], {"soma": ["leak", "na", "kdr", "ca", "hcn"]})
        mechanism_names = {density.instance_name for cv in cell.cell.cvs for density in cv.density_mech}
        self.assertIn("Na_soma", mechanism_names)
        self.assertIn("Kdr_soma", mechanism_names)
        self.assertIn("Ca_soma", mechanism_names)
        self.assertIn("HCN_soma", mechanism_names)

    def test_neuron_all_channels_builds_expected_mechanisms(self) -> None:
        cell = NeuronIO().build()
        try:
            mechanisms = {mech.name() for seg in cell.root_soma for mech in seg}
            self.assertIn("Na_ZH19_IO", mechanisms)
            self.assertIn("Kdr_ZH19_IO", mechanisms)
            self.assertIn("Ca_ZH19_IO", mechanisms)
            self.assertIn("HCN_ZH19_IO", mechanisms)
        finally:
            cell.cleanup()

    def test_short_leak_only_voltage_trace_is_finite(self) -> None:
        params = load_io19_params()
        config = IOConfig(toggles=IOToggles(leak=True, na=False, kdr=False, ca=False, hcn=False))
        neuron_io = NeuronIO(params=params, config=config).build()
        braincell_io = BrainCellIO(params=params, config=config).build()
        try:
            nrn_probes = neuron_io.attach_voltage_probes(all_compartments=True, soma=True)
            bc_probes = braincell_io.attach_voltage_probes(all_compartments=True, soma=True)

            stim = h.IClamp(neuron_io.root_soma(0.5))
            stim.delay = 1.0
            stim.dur = 2.0
            stim.amp = 0.01
            h.cvode_active(0)
            h.dt = 0.05
            h.steps_per_ms = 20.0
            h.celsius = config.temperature_celsius
            h.v_init = config.v_init_mV
            h.finitialize(h.v_init)
            h.tstop = 5.0
            h.run()

            braincell_io.cell.place(
                at("soma", 0.5),
                mech.CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.01 * u.nA),
            )
            braincell_io.cell.init_state()
            braincell_io.cell.reset_state()
            bc_run = braincell_io.cell.run(dt=0.05 * u.ms, duration=5.0 * u.ms)

            nrn_v = neuron_io.collect_voltage_results(nrn_probes)
            bc_v = braincell_io.collect_voltage_results(bc_probes, bc_run)
            self.assertGreater(nrn_v["soma_voltage_mV"].size, 0)
            self.assertGreater(bc_v["soma_voltage_mV"].size, 0)
            self.assertTrue(np.all(np.isfinite(nrn_v["soma_voltage_mV"])))
            self.assertTrue(np.all(np.isfinite(bc_v["soma_voltage_mV"])))
            self.assertEqual(nrn_v["compartment_voltage_mV"].shape[1], 1)
            self.assertEqual(bc_v["compartment_voltage_mV"].shape[1], 1)
        finally:
            neuron_io.cleanup()


if __name__ == "__main__":
    unittest.main()
