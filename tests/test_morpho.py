from __future__ import annotations

import unittest

from ._support import np, u

from braincell import Branch, Morpho, MorphoBranch


class MorphoTest(unittest.TestCase):
    def test_tree_topology_queries_and_branch_views(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        dend = Branch.lengths_shared(
            lengths=np.array([100.0]) * u.um,
            radii=np.array([2.0, 1.0]) * u.um,
            type="basal_dendrite",
        )
        axon = Branch.lengths_shared(lengths=[40.0], radii=[0.8, 0.4], type="axon")

        tree = Morpho.from_root(soma, name="soma")
        dend_view = tree.soma.attach("dend", dend, parent_x=0.9)
        axon_view = tree.attach(parent=tree.soma, child="axon", branch=axon, parent_x=0.2, child_x=1.0)
        tree.soma.extra = Branch.lengths_shared(lengths=[30.0], radii=[1.0, 0.6], type="apical_dendrite")

        self.assertIsInstance(tree.soma, MorphoBranch)
        self.assertIsNone(tree.soma.parent)
        self.assertEqual(dend_view.parent.name, "soma")
        self.assertEqual(dend_view.parent_x, 0.9)
        self.assertEqual(axon_view.child_x, 1.0)
        self.assertEqual(tree.branch_by_index(1).name, "basal_dendrite_0")
        self.assertEqual(tree.branch_by_name("axon_0").parent.name, "soma")
        self.assertEqual(tree.soma.dend.name, "basal_dendrite_0")
        self.assertEqual(tree.soma.axon.name, "axon_0")
        self.assertEqual(tree.path_to_root(2), (0, 2))
        self.assertEqual(len(tree.branches), 4)
        self.assertEqual(len(tree.connections), 3)
        self.assertEqual(tree.connections[0].parent_x, 0.9)
        self.assertEqual(tree.connections[0].child_x, 0.0)
        self.assertEqual(tree.connections[1].child_x, 1.0)
        self.assertEqual(tree.soma.total_length.to_decimal(u.um), 20.0)
        self.assertEqual(tree.soma.radius_proximal.to_decimal(u.um), 10.0)
        self.assertEqual(tree.soma.radius_distal.to_decimal(u.um), 10.0)
        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "├── basal_dendrite_0",
                    "├── axon_0",
                    "└── apical_dendrite_0",
                )
            ),
        )

    def test_attach_by_name_and_attachment_point(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        dend = Branch.lengths_shared(lengths=[60.0], radii=[2.0, 1.0], type="basal_dendrite")
        tree = Morpho.from_root(soma, name="soma")
        branch = tree.soma[0.3, 0.4].attach("dend", dend)

        self.assertEqual(branch.parent_x, 0.3)
        self.assertEqual(branch.child_x, 0.4)
        self.assertEqual(tree.connections[0].child_x, 0.4)

    def test_topo_renders_nested_tree(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        dend = Branch.lengths_shared(lengths=[60.0], radii=[2.0, 1.0], type="basal_dendrite")
        tuft = Branch.lengths_shared(lengths=[30.0], radii=[1.0, 0.6], type="apical_dendrite")
        axon = Branch.lengths_shared(lengths=[40.0], radii=[0.8, 0.4], type="axon")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.dend = dend
        tree.soma.dend.tuft = tuft
        tree.soma.axon = axon

        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "├── basal_dendrite_0",
                    "│   └── apical_dendrite_0",
                    "└── axon_0",
                )
            ),
        )

    def test_auto_names_skip_existing_explicit_suffixes_without_jumping_ahead(self) -> None:
        soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        dend_named = Branch.lengths_shared(
            lengths=[60.0],
            radii=[2.0, 1.0],
            name="dend_20",
            type="dend",
        )
        dend_auto = Branch.lengths_shared(lengths=[60.0], radii=[2.0, 1.0], type="dend")

        tree = Morpho.from_root(soma, name="soma")
        tree.soma.named = dend_named
        first_auto = tree.soma.attach("first", dend_auto)
        second_auto = tree.soma.attach(
            "second",
            Branch.lengths_shared(lengths=[60.0], radii=[2.0, 1.0], type="dend"),
        )

        self.assertEqual(tree.soma.named.name, "dend_20")
        self.assertEqual(first_auto.name, "dend_0")
        self.assertEqual(second_auto.name, "dend_1")

    def test_root_can_opt_into_type_based_auto_naming(self) -> None:
        axon = Branch.lengths_shared(lengths=[40.0], radii=[0.8, 0.4], type="axon")

        tree = Morpho.from_root(axon, name=None)

        self.assertEqual(tree.root.name, "axon_0")
        self.assertEqual(tree.topo(), "axon_0")

    def test_foreign_missing_and_reserved_children_are_rejected(self) -> None:
        soma0 = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
        soma1 = Branch.lengths_shared(lengths=[18.0], radii=[9.0, 9.0], type="soma")
        dend = Branch.lengths_shared(lengths=[60.0], radii=[2.0, 1.0], type="basal_dendrite")

        tree0 = Morpho.from_root(soma0, name="soma")
        tree1 = Morpho.from_root(soma1, name="other")
        tree0.soma.dend = dend

        with self.assertRaises(ValueError):
            tree1.other.foreign = tree0.soma.dend
        with self.assertRaises(KeyError):
            tree0.attach(parent="missing", child="axon", branch=dend)
        with self.assertRaises(ValueError):
            tree0.attach(parent=tree1.other, child="foreign", branch=dend)
        with self.assertRaises(ValueError):
            tree0.soma.total_length = Branch.lengths_shared(lengths=[10.0], radii=[1.0, 1.0], type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.select = Branch.lengths_shared(lengths=[10.0], radii=[1.0, 1.0], type="axon")
        with self.assertRaises(ValueError):
            tree0.soma.other = Branch.lengths_shared(
                lengths=[60.0],
                radii=[2.0, 1.0],
                name="basal_dendrite_0",
                type="basal_dendrite",
            )
