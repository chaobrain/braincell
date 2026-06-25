import unittest

import brainunit as u
import numpy as np
from braincell.mech import CableProperty

from examples.neuron_compare.cell.pc_ma2024.parameters import (
    DEFAULT_MORPH_PATH,
    load_pc24_params,
)
from examples.neuron_compare.cell.pc_ma2024.pc_braincell import PC, pc24_dend_cm


class PCBrainCellTest(unittest.TestCase):
    def test_build_accepts_population_shape_and_name(self) -> None:
        pc = PC(
            DEFAULT_MORPH_PATH,
            params=load_pc24_params(),
            pop_size=(2,),
            name="pc_pop",
        ).build()

        self.assertEqual(pc.cell.pop_size, (2,))
        self.assertEqual(pc.cell.name, "pc_pop")

    def test_callable_cable_paint_matches_pc_capacitance_rule(self) -> None:
        params = load_pc24_params()
        pc = PC(DEFAULT_MORPH_PATH, params=params).build()
        rules = pc.cell._paint_rules
        self.assertLess(len(rules), len(pc.morph.branches))
        self.assertEqual(sum(isinstance(rule.mechanism, CableProperty) for rule in rules), 2)

        cvs = pc.cell.cvs
        for cv in cvs:
            cm = float(np.asarray(cv.cm.to_decimal(u.uF / u.cm ** 2), dtype=float))
            if cv.branch_type == "soma":
                expected = params.cable.soma_cm_uF_cm2
            else:
                diam_um = float(np.asarray(cv.diam_arc_mean.to_decimal(u.um), dtype=float))
                expected = pc24_dend_cm(diam_um, params.cable)
            self.assertAlmostEqual(cm, expected, places=12)


if __name__ == "__main__":
    unittest.main()
