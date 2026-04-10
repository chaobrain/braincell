# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import brainunit as u
import numpy as np

from braincell import (
    ApicalDendrite,
    Axon,
    BasalDendrite,
    Branch,
    CustomBranch,
    Dendrite,
    Morpho,
    Soma,
)
from braincell.io import (
    CheckpointError,
    CheckpointVersionError,
    load_branch,
    load_morpho,
    save_branch,
    save_morpho,
)
from braincell.io import checkpoint as checkpoint_module


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "develop_doc" / "morpho_files"


def _make_lengths_branch(*, type: str = "dendrite") -> Branch:
    return Branch.from_lengths(
        lengths=[10.0, 15.0, 20.0] * u.um,
        radii=[3.0, 2.5, 2.0, 1.5] * u.um,
        type=type,
    )


def _make_points_branch(*, type: str = "axon") -> Branch:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 20.0, 0.0],
            [10.0, 20.0, 30.0],
        ]
    ) * u.um
    return Branch.from_points(points=pts, radii=[2.0, 1.5, 1.0, 0.5] * u.um, type=type)


def _make_complex_morpho() -> Morpho:
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    apical = Branch.from_points(
        points=[[20.0, 0.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="apical_dendrite",
    )
    basal_a = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [-30.0, -50.0, 10.0]] * u.um,
        radii=[3.0, 1.5] * u.um,
        type="basal_dendrite",
    )
    basal_b = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [40.0, -40.0, -10.0], [60.0, -80.0, -20.0]] * u.um,
        radii=[2.5, 1.5, 0.8] * u.um,
        type="basal_dendrite",
    )
    axon = Branch.from_points(
        points=[[10.0, 0.0, 0.0], [10.0, 0.0, -100.0]] * u.um,
        radii=[1.0, 0.5] * u.um,
        type="axon",
    )

    morpho = Morpho.from_root(soma, name="soma")
    morpho.attach(parent="soma", child_branch=apical, child_name="apical", parent_x=1.0)
    morpho.attach(parent="soma", child_branch=basal_a, child_name="basal_a", parent_x=0.0)
    morpho.attach(parent="soma", child_branch=basal_b, child_name="basal_b", parent_x=0.0)
    # parent_x=0.5 is only allowed on a soma parent — exercise that path.
    morpho.attach(parent="soma", child_branch=axon, child_name="axon_main", parent_x=0.5)
    return morpho


class CheckpointBranchRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_round_trip_lengths_only_branch(self) -> None:
        branch = _make_lengths_branch()
        path = save_branch(branch, self.tmp / "len_only.bcm")
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix, ".bcm")

        loaded = load_branch(path)
        self.assertIsInstance(loaded, Dendrite)
        self.assertEqual(loaded, branch)
        self.assertIsNone(loaded.points_proximal)
        self.assertIsNone(loaded.points_distal)

    def test_round_trip_points_branch(self) -> None:
        branch = _make_points_branch()
        path = save_branch(branch, self.tmp / "points.bcm")
        loaded = load_branch(path)
        self.assertIsInstance(loaded, Axon)
        self.assertEqual(loaded, branch)
        self.assertIsNotNone(loaded.points_proximal)
        self.assertIsNotNone(loaded.points_distal)

    def test_each_subclass_round_trip(self) -> None:
        branches = [
            (Soma, Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")),
            (Dendrite, _make_lengths_branch(type="dendrite")),
            (Axon, _make_lengths_branch(type="axon")),
            (BasalDendrite, _make_lengths_branch(type="basal_dendrite")),
            (ApicalDendrite, _make_lengths_branch(type="apical_dendrite")),
            (CustomBranch, _make_lengths_branch(type="custom")),
        ]
        for cls, branch in branches:
            with self.subTest(branch_type=branch.type):
                path = save_branch(branch, self.tmp / f"{branch.type}.bcm")
                loaded = load_branch(path)
                self.assertIsInstance(loaded, cls)
                self.assertEqual(loaded, branch)
                self.assertEqual(type(loaded), cls)

    def test_branch_method_delegates(self) -> None:
        branch = _make_points_branch()
        path = branch.save_checkpoint(self.tmp / "via_method.bcm")
        loaded = Branch.load_checkpoint(path)
        self.assertEqual(loaded, branch)

    def test_typed_subclass_load_checkpoint_accepts_match(self) -> None:
        branch = _make_lengths_branch(type="dendrite")
        path = branch.save_checkpoint(self.tmp / "dend.bcm")
        loaded = Dendrite.load_checkpoint(path)
        self.assertIsInstance(loaded, Dendrite)
        self.assertEqual(loaded, branch)

    def test_typed_subclass_load_checkpoint_rejects_mismatch(self) -> None:
        branch = _make_lengths_branch(type="dendrite")
        path = branch.save_checkpoint(self.tmp / "dend.bcm")
        with self.assertRaises(TypeError) as cm:
            Soma.load_checkpoint(path)
        self.assertIn("Soma.load_checkpoint", str(cm.exception))
        self.assertIn("'dendrite'", str(cm.exception))

    def test_canonical_jump_segments_preserved(self) -> None:
        # Discontinuous radii at the boundary force _canonicalize_segments to
        # insert a zero-length jump segment. The reload must keep that exact
        # segment count rather than re-canonicalizing.
        branch = Branch.from_lengths(
            lengths=[10.0, 15.0] * u.um,
            radii_proximal=[2.0, 3.0] * u.um,
            radii_distal=[2.0, 3.0] * u.um,
            type="custom",
        )
        # _canonicalize_segments inserts one jump between segment 0 and 1.
        self.assertEqual(branch.n_segments, 3)

        path = branch.save_checkpoint(self.tmp / "jumps.bcm")
        loaded = load_branch(path)
        self.assertEqual(loaded.n_segments, branch.n_segments)
        self.assertEqual(loaded, branch)

    def test_save_branch_rejects_morpho(self) -> None:
        morpho = Morpho.from_root(
            Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"),
            name="soma",
        )
        with self.assertRaises(TypeError) as cm:
            save_branch(morpho, self.tmp / "bad.bcm")  # type: ignore[arg-type]
        self.assertIn("Branch", str(cm.exception))

    def test_path_without_suffix_is_auto_extended(self) -> None:
        branch = _make_lengths_branch()
        path = save_branch(branch, self.tmp / "no_suffix")
        self.assertEqual(path.suffix, ".bcm")
        self.assertTrue(path.exists())

    def test_path_with_custom_suffix_is_preserved(self) -> None:
        branch = _make_lengths_branch()
        path = save_branch(branch, self.tmp / "custom.npz")
        self.assertEqual(path.suffix, ".npz")
        self.assertTrue(path.exists())
        loaded = load_branch(path)
        self.assertEqual(loaded, branch)

    def test_load_branch_on_morpho_file_raises(self) -> None:
        morpho = _make_complex_morpho()
        path = save_morpho(morpho, self.tmp / "morpho.bcm")
        with self.assertRaises(CheckpointError) as cm:
            load_branch(path)
        self.assertIn("load_morpho", str(cm.exception))

    def test_missing_file_raises_filenotfound(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_branch(self.tmp / "does_not_exist.bcm")


class CheckpointMorphoRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_round_trip_simple_morpho(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(
            lengths=[50.0] * u.um, radii=[2.0, 1.0] * u.um, type="dendrite"
        )
        morpho = Morpho.from_root(soma, name="soma")
        morpho.soma.dendrite = dend

        path = save_morpho(morpho, self.tmp / "simple.bcm")
        loaded = load_morpho(path)

        self.assertEqual(loaded, morpho)
        self.assertEqual(loaded.topo(), morpho.topo())
        self.assertEqual(loaded.root.name, "soma")

    def test_round_trip_complex_morpho(self) -> None:
        morpho = _make_complex_morpho()
        path = save_morpho(morpho, self.tmp / "complex.bcm")
        loaded = load_morpho(path)

        self.assertEqual(loaded, morpho)
        self.assertEqual(loaded.topo(), morpho.topo())
        self.assertEqual(loaded.n_branches, morpho.n_branches)
        # parent_x = 0.5 attachment must survive.
        axon = loaded.branch(name="axon_main")
        self.assertEqual(axon.parent_x, 0.5)
        self.assertEqual(axon.parent.name, "soma")
        # parent_x = 0.0 attachments survive.
        basal_a = loaded.branch(name="basal_a")
        self.assertEqual(basal_a.parent_x, 0.0)

    def test_morpho_method_delegates(self) -> None:
        morpho = _make_complex_morpho()
        path = morpho.save_checkpoint(self.tmp / "via_method.bcm")
        loaded = Morpho.load_checkpoint(path)
        self.assertEqual(loaded, morpho)

    def test_type_name_counters_restored(self) -> None:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        morpho = Morpho.from_root(soma, name="soma")
        for _ in range(3):
            morpho.attach(
                parent="soma",
                child_branch=Branch.from_lengths(
                    lengths=[10.0] * u.um, radii=[1.0, 0.5] * u.um, type="dendrite"
                ),
            )
        # Auto-naming has produced dendrite_0, dendrite_1, dendrite_2.
        self.assertEqual(morpho.branch(index=3).name, "dendrite_2")
        saved_counters = dict(morpho._type_name_counters)

        path = save_morpho(morpho, self.tmp / "counters.bcm")
        loaded = load_morpho(path)

        self.assertEqual(loaded._type_name_counters, saved_counters)
        # Next auto-attached dendrite must continue the sequence at _3.
        new_view = loaded.attach(
            parent="soma",
            child_branch=Branch.from_lengths(
                lengths=[5.0] * u.um, radii=[1.0, 0.5] * u.um, type="dendrite"
            ),
        )
        self.assertEqual(new_view.name, "dendrite_3")

    def test_round_trip_swc_fixture(self) -> None:
        if not FIXTURE_DIR.exists():
            self.skipTest(f"fixture directory {FIXTURE_DIR} not present")
        swc_path = FIXTURE_DIR / "CA1.swc"
        if not swc_path.exists():
            self.skipTest(f"fixture {swc_path} not present")

        morpho = Morpho.from_swc(swc_path)
        path = morpho.save_checkpoint(self.tmp / "ca1.bcm")
        loaded = load_morpho(path)
        self.assertEqual(loaded, morpho)
        self.assertEqual(loaded.n_branches, morpho.n_branches)

    def test_round_trip_asc_fixture(self) -> None:
        if not FIXTURE_DIR.exists():
            self.skipTest(f"fixture directory {FIXTURE_DIR} not present")
        for name in ("goc.asc", "pc.asc"):
            asc_path = FIXTURE_DIR / name
            if not asc_path.exists():
                continue
            with self.subTest(fixture=name):
                morpho = Morpho.from_asc(asc_path)
                path = morpho.save_checkpoint(self.tmp / f"{name}.bcm")
                loaded = load_morpho(path)
                self.assertEqual(loaded, morpho)

    def test_save_morpho_rejects_branch(self) -> None:
        with self.assertRaises(TypeError) as cm:
            save_morpho(_make_lengths_branch(), self.tmp / "bad.bcm")  # type: ignore[arg-type]
        self.assertIn("Morpho", str(cm.exception))

    def test_load_morpho_on_branch_file_raises(self) -> None:
        path = save_branch(_make_lengths_branch(), self.tmp / "branch.bcm")
        with self.assertRaises(CheckpointError) as cm:
            load_morpho(path)
        self.assertIn("load_branch", str(cm.exception))

    def test_morpho_path_auto_extension(self) -> None:
        morpho = _make_complex_morpho()
        path = save_morpho(morpho, self.tmp / "no_suffix")
        self.assertEqual(path.suffix, ".bcm")
        self.assertTrue(path.exists())


class CheckpointFormatErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    @staticmethod
    def _write_npz_with_manifest(path: Path, manifest: dict, payload: dict | None = None) -> None:
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        manifest_array = np.frombuffer(manifest_bytes, dtype=np.uint8)
        np.savez(path, manifest=manifest_array, **(payload or {}))

    def test_unrecognized_format_field(self) -> None:
        path = self.tmp / "bogus.npz"
        self._write_npz_with_manifest(
            path,
            {"format": "not.braincell", "version": 1, "kind": "branch"},
        )
        with self.assertRaises(CheckpointError) as cm:
            load_branch(path)
        self.assertIn("unrecognized checkpoint format", str(cm.exception))

    def test_unsupported_future_version(self) -> None:
        path = self.tmp / "future.npz"
        self._write_npz_with_manifest(
            path,
            {
                "format": "braincell.io.checkpoint",
                "version": 999,
                "kind": "branch",
                "branch": {"type": "soma", "n_segments": 1, "has_points": False},
            },
        )
        with self.assertRaises(CheckpointVersionError):
            load_branch(path)

    def test_invalid_version_value(self) -> None:
        path = self.tmp / "negative.npz"
        self._write_npz_with_manifest(
            path,
            {
                "format": "braincell.io.checkpoint",
                "version": -1,
                "kind": "branch",
            },
        )
        with self.assertRaises(CheckpointError) as cm:
            load_branch(path)
        self.assertIn("invalid checkpoint version", str(cm.exception))

    def test_missing_manifest_entry(self) -> None:
        path = self.tmp / "no_manifest.npz"
        np.savez(path, foo=np.zeros(3))
        with self.assertRaises(CheckpointError) as cm:
            load_branch(path)
        self.assertIn("missing 'manifest'", str(cm.exception))

    def test_corrupt_manifest_bytes(self) -> None:
        path = self.tmp / "garbage.npz"
        np.savez(path, manifest=np.frombuffer(b"\x00not-json{", dtype=np.uint8))
        with self.assertRaises(CheckpointError) as cm:
            load_branch(path)
        self.assertIn("corrupt manifest", str(cm.exception))

    def test_corrupt_zip_archive(self) -> None:
        path = self.tmp / "broken.bcm"
        path.write_bytes(b"this is not a zip archive at all")
        with self.assertRaises(CheckpointError) as cm:
            load_branch(path)
        self.assertIn("cannot read checkpoint", str(cm.exception))

    def test_morpho_manifest_missing_branches(self) -> None:
        path = self.tmp / "empty_morpho.npz"
        self._write_npz_with_manifest(
            path,
            {
                "format": "braincell.io.checkpoint",
                "version": 1,
                "kind": "morpho",
                "branches": [],
                "root_name": "soma",
            },
        )
        with self.assertRaises(CheckpointError) as cm:
            load_morpho(path)
        self.assertIn("no branches", str(cm.exception))

    def test_morpho_manifest_root_name_mismatch(self) -> None:
        # Build a real morpho, then rewrite the manifest with a wrong root_name.
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        morpho = Morpho.from_root(soma, name="soma")
        path = save_morpho(morpho, self.tmp / "mismatch.bcm")

        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files if key != "manifest"}
            manifest = json.loads(bytes(np.asarray(data["manifest"]).tobytes()).decode("utf-8"))

        manifest["root_name"] = "wrong"
        new_manifest_bytes = json.dumps(manifest).encode("utf-8")
        # ``np.savez`` auto-appends ``.npz`` to string paths, so write through
        # an open file handle to overwrite the original ``.bcm`` in place.
        with open(path, "wb") as fh:
            np.savez(
                fh,
                manifest=np.frombuffer(new_manifest_bytes, dtype=np.uint8),
                **payload,
            )
        with self.assertRaises(CheckpointError) as cm:
            load_morpho(path)
        self.assertIn("root_name", str(cm.exception))

    def test_morpho_manifest_cycle_detection(self) -> None:
        # Two non-root branches that reference each other as parents.
        manifest = {
            "format": "braincell.io.checkpoint",
            "version": 1,
            "kind": "morpho",
            "root_name": "soma",
            "type_name_counters": {},
            "branches": [
                {
                    "index": 0,
                    "name": "soma",
                    "type": "soma",
                    "parent_name": None,
                    "parent_x": None,
                    "child_x": None,
                    "n_segments": 1,
                    "has_points": False,
                },
                {
                    "index": 1,
                    "name": "a",
                    "type": "dendrite",
                    "parent_name": "b",
                    "parent_x": 1.0,
                    "child_x": 0.0,
                    "n_segments": 1,
                    "has_points": False,
                },
                {
                    "index": 2,
                    "name": "b",
                    "type": "dendrite",
                    "parent_name": "a",
                    "parent_x": 1.0,
                    "child_x": 0.0,
                    "n_segments": 1,
                    "has_points": False,
                },
            ],
        }
        payload = {}
        for idx in (0, 1, 2):
            payload[f"branches/{idx}/lengths"] = np.array([10.0], dtype=np.float64)
            payload[f"branches/{idx}/radii_proximal"] = np.array([1.0], dtype=np.float64)
            payload[f"branches/{idx}/radii_distal"] = np.array([1.0], dtype=np.float64)

        path = self.tmp / "cycle.npz"
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        np.savez(
            path,
            manifest=np.frombuffer(manifest_bytes, dtype=np.uint8),
            **payload,
        )
        with self.assertRaises(CheckpointError) as cm:
            load_morpho(path)
        self.assertIn("not yet been inserted", str(cm.exception))


class CheckpointAtomicWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_failed_write_leaves_no_final_file(self) -> None:
        target = self.tmp / "atomic.bcm"

        def boom(*args, **kwargs):
            raise RuntimeError("simulated mid-write failure")

        with mock.patch.object(checkpoint_module.np, "savez", side_effect=boom):
            with self.assertRaises(RuntimeError):
                save_branch(_make_lengths_branch(), target)

        self.assertFalse(target.exists(), "final checkpoint file should not exist after a failed write")
        # Temp staging file should also be cleaned up.
        self.assertFalse((self.tmp / "atomic.bcm.tmp").exists())

    def test_load_uses_allow_pickle_false(self) -> None:
        target = save_branch(_make_lengths_branch(), self.tmp / "no_pickle.bcm")

        original_load = checkpoint_module.np.load
        seen_kwargs: dict = {}

        def spy_load(*args, **kwargs):
            seen_kwargs.update(kwargs)
            return original_load(*args, **kwargs)

        with mock.patch.object(checkpoint_module.np, "load", side_effect=spy_load):
            load_branch(target)

        self.assertIs(seen_kwargs.get("allow_pickle"), False)


if __name__ == "__main__":
    unittest.main()
