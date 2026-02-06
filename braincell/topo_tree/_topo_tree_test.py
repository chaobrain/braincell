import pytest
import numpy as np

from ._topo_tree import Taper, Branch, Tree


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def demo_tree() -> Tree:
    """
    Build a small tree:
        soma
        ├── L
        │   └── L1
        └── R
    """
    tree = Tree(verbose=True)

    soma_branch = Branch.from_length([5.0], [1.0], [0.9])

    L_branch = Branch.from_points(
        points=[(0, 0, 0), (0, 0, 5), (0, 0, 12)],
        radius=[1.0, 0.8, 0.6],
    )

    R_branch = Branch.from_points(
        points=[(0, 0, 0, 1.0), (0, 0, 5, 0.7)],
    )

    L1_branch = Branch.from_tapers([Taper.from_length(4.0, 0.8, 0.6)])

    tree.attach(soma_branch, label="soma")
    tree.attach(L_branch, parent="soma", label="L")
    tree.attach(R_branch, parent="soma", label="R")
    tree.attach(L1_branch, parent="L", label="L1")

    return tree


# -------------------------
# Taper tests
# -------------------------

def test_taper_from_length_default_r1():
    t = Taper.from_length(10, 2.0)
    assert t.L == 10.0
    assert t.r0 == 2.0
    assert t.r1 == 2.0


def test_taper_from_length_with_r1():
    t = Taper.from_length(10, 2.0, 1.0)
    assert t.r1 == 1.0


def test_taper_from_points_vec3():
    t = Taper.from_points((0, 0, 0), (0, 0, 10), r0=1.0, r1=0.5)
    assert t.L == pytest.approx(10.0)
    assert t.p0 == (0, 0, 0)
    assert t.p1 == (0, 0, 10)
    assert t.r0 == pytest.approx(1.0)
    assert t.r1 == pytest.approx(0.5)


def test_taper_from_points_vec4():
    t = Taper.from_points((0, 0, 0, 1.0), (0, 0, 10, 0.5))
    assert t.L == pytest.approx(10.0)
    assert t.p0 == (0, 0, 0)
    assert t.p1 == (0, 0, 10)
    assert t.r0 == pytest.approx(1.0)
    assert t.r1 == pytest.approx(0.5)


def test_taper_from_points_vec3_missing_r0_should_fail():
    with pytest.raises(ValueError):
        _ = Taper.from_points((0, 0, 0), (0, 0, 1), r0=None)


# -------------------------
# Branch tests
# -------------------------

def test_branch_empty():
    b = Branch.empty()
    assert len(b.tapers) == 0
    assert b.total_length() == 0.0


def test_branch_from_length():
    b = Branch.from_length(
        lengths=[5.0, 4.0, 3.0],
        r0s=[1.0, 0.8, 0.6],
        r1s=[0.8, 0.6, 0.5],
    )
    assert len(b.tapers) == 3
    assert b.total_length() == pytest.approx(12.0)


def test_branch_from_length_default_r1s():
    b = Branch.from_length(
        lengths=[5.0, 4.0],
        r0s=[1.0, 0.8],
        r1s=None,
    )
    assert len(b.tapers) == 2
    assert b.tapers[0].r1 == pytest.approx(1.0)
    assert b.tapers[1].r1 == pytest.approx(0.8)


def test_branch_from_length_mismatch_should_fail():
    with pytest.raises(ValueError):
        _ = Branch.from_length(lengths=[1.0], r0s=[1.0, 2.0], r1s=[1.0])


def test_branch_from_points_vec3_radius():
    b = Branch.from_points(
        points=[(0, 0, 0), (0, 0, 5), (0, 0, 12)],
        radius=[1.0, 0.8, 0.6],
    )
    assert len(b.tapers) == 2
    assert b.tapers[0].L == pytest.approx(5.0)
    assert b.tapers[1].L == pytest.approx(7.0)
    assert b.total_length() == pytest.approx(12.0)


def test_branch_from_points_vec3_missing_radius_should_fail():
    with pytest.raises(ValueError):
        _ = Branch.from_points(points=[(0, 0, 0), (0, 0, 1)], radius=None)


def test_branch_from_points_vec3_radius_mismatch_should_fail():
    with pytest.raises(ValueError):
        _ = Branch.from_points(
            points=[(0, 0, 0), (0, 0, 1), (0, 0, 2)],
            radius=[1.0, 0.8],  # mismatch
        )


def test_branch_from_points_vec4():
    b = Branch.from_points(
        points=[(0, 0, 0, 1.0), (0, 0, 5, 0.8), (0, 0, 12, 0.6)]
    )
    assert len(b.tapers) == 2
    assert b.total_length() == pytest.approx(12.0)


def test_branch_from_tapers():
    t1 = Taper.from_length(1.0, 1.0)
    t2 = Taper.from_length(2.0, 0.8, 0.5)
    b = Branch.from_tapers([t1, t2])
    assert len(b.tapers) == 2
    assert b.total_length() == pytest.approx(3.0)


def test_branch_post_init_type_check_should_fail():
    # This should fail: Branch("c1") means tapers="c1" (str), not list[Taper]
    with pytest.raises(TypeError):
        _ = Branch("c1")  # type: ignore


# -------------------------
# Tree tests
# -------------------------

def test_tree_topo_str_contains_labels(demo_tree: Tree):
    topo = demo_tree.topo_str()
    assert "soma" in topo
    assert "L" in topo
    assert "R" in topo
    assert "L1" in topo


def test_tree_print_verbose_on(demo_tree: Tree):
    demo_tree.verbose = True
    s = str(demo_tree)
    assert "Tree Topology:" in s
    assert "Branches:" in s
    assert "Branch(n_tapers=" in s


def test_tree_print_verbose_off(demo_tree: Tree):
    demo_tree.verbose = False
    s = str(demo_tree)
    assert "Tree Topology:" in s
    assert "Branches:" not in s


def test_tree_attach_unknown_parent_should_fail():
    tree = Tree()
    with pytest.raises(KeyError):
        tree.attach(Branch.empty(), parent="no_such_parent", label="X")


def test_tree_remap_ids_inplace(demo_tree: Tree):
    demo_tree.remap_ids({0: 100, 1: 101, 2: 102, 3: 103}, inplace=True)
    assert 100 in demo_tree.branches
    assert demo_tree.label2id["soma"] == 100


def test_tree_reorder_ids(demo_tree: Tree):
    demo_tree.remap_ids({0: 10, 1: 20, 2: 30, 3: 40}, inplace=True)
    demo_tree.reorder_ids(inplace=True)
    assert sorted(demo_tree.branches.keys()) == [0, 1, 2, 3]


def test_tree_delete_subtree_by_label(demo_tree: Tree):
    # delete L subtree (L and L1)
    demo_tree.delete_subtree("L", inplace=True, reorder=False)

    assert "L" not in demo_tree.label2id
    assert "L1" not in demo_tree.label2id

    assert "soma" in demo_tree.label2id
    assert "R" in demo_tree.label2id


def test_tree_merge_tree(demo_tree: Tree):
    other = Tree()
    other.attach(Branch.empty(), label="soma2")
    other.attach(Branch.empty(), parent="soma2", label="A")
    other.attach(Branch.empty(), parent="A", label="A1")

    demo_tree.merge_tree(other, parent="soma", parent_x=0.8, child_x=0.0, inplace=True, reorder=True)

    assert "soma2" in demo_tree.label2id
    assert "A" in demo_tree.label2id
    assert "A1" in demo_tree.label2id

    topo = demo_tree.topo_str()
    assert "soma2" in topo
    assert "A" in topo
    assert "A1" in topo


def test_tree_merge_tree_multiple_roots_should_fail(demo_tree: Tree):
    other = Tree()
    other.attach(Branch.empty(), label="root1")
    other.attach(Branch.empty(), label="root2")  # second root

    with pytest.raises(ValueError):
        demo_tree.merge_tree(other, parent="soma")


def test_tree_merge_tree_no_root_should_fail(demo_tree: Tree):
    other = Tree()
    # Inject a node without creating root edge (-1, root)
    other.branches[0] = Branch.empty()

    with pytest.raises(ValueError):
        demo_tree.merge_tree(other, parent="soma")
