import unittest

from examples.neuron_compare.cell.pc_ma2024.parameters import (
    DEFAULT_MORPH_PATH,
    load_pc24_params,
)
from examples.neuron_compare.cell.pc_ma2024.pc_braincell import PC


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


if __name__ == "__main__":
    unittest.main()
