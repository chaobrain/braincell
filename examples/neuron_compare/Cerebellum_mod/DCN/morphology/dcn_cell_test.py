from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dcn_cell import build_dcn_cell
from dcn_native import resolve_source_hoc


class DcnCellTemplatePaintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.source_hoc = resolve_source_hoc()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.build = build_dcn_cell()
        cls.cell = cls.build.cell
        cls.native = cls.build.native

    def test_cell_uses_native_morphology(self) -> None:
        self.assertEqual(self.cell.morpho.n_branches, 517)
        self.assertEqual(self.native.morpho.root.name, "soma")

    def test_expected_channel_classes_are_painted(self) -> None:
        classes = Counter(
            getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__)
            for rule in self.cell.paint_rules
        )

        self.assertGreaterEqual(classes["CableProperty"], 2)
        self.assertEqual(classes["IL"], 6)
        self.assertEqual(classes["SodiumFixed"], 1)
        self.assertEqual(classes["PotassiumFixed"], 1)
        self.assertEqual(classes["CdpHVA_SU2015_DCN"], 1)
        self.assertEqual(classes["CdpLVA_SU2015_DCN"], 1)
        self.assertEqual(classes["NaF_SU2015_DCN"], 4)
        self.assertEqual(classes["NaP_SU2015_DCN"], 1)
        self.assertEqual(classes["fKdr_SU2015_DCN"], 4)
        self.assertEqual(classes["sKdr_SU2015_DCN"], 4)
        self.assertEqual(classes["SK_SU2015_DCN"], 3)
        self.assertEqual(classes["HCN_SU2015_DCN"], 3)
        self.assertEqual(classes["CaLVA_SU2015_DCN"], 3)
        self.assertEqual(classes["CaHVA_SU2015_DCN"], 3)

    def test_axnode_region_has_no_active_channel_set(self) -> None:
        axnode_mask = self.native.morpho.select(self.native.region("axNode"))
        self.assertEqual(len(axnode_mask.intervals), 20)

        axnode_rules = [
            rule
            for rule in self.cell.paint_rules
            if len(self.native.morpho.select(rule.region).intervals) == 20
            and all(self.native.morpho.branch(index=index).name.startswith("axNode__") for index, _, _ in self.native.morpho.select(rule.region).intervals)
        ]
        classes = {
            getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__)
            for rule in axnode_rules
        }
        self.assertEqual(classes, {"CableProperty", "IL"})


if __name__ == "__main__":
    unittest.main()
