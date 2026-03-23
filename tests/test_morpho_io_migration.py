from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path
import os

import brainunit as u

from braincell import Branch, Morpho, SwcReader


def _write_temp_swc(body: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=".swc")
    os.close(fd)
    path = Path(name)
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


def test_new_morpho_api_supports_attach_and_topology() -> None:
    soma = Branch.lengths_shared(lengths=[20.0], radii=[10.0, 10.0], type="soma")
    dend = Branch.lengths_shared(lengths=[60.0], radii=[2.0, 1.0], type="basal_dendrite")

    tree = Morpho.from_root(soma, name="soma")
    tree.soma.dend = dend

    assert len(tree.branches) == 2
    assert tree.topo() == "soma\n└── basal_dendrite_0"
    assert tree.branch_by_name("basal_dendrite_0").total_length.to_decimal(u.um) == 60.0


def test_new_swc_reader_and_morpho_from_swc_work() -> None:
    path = _write_temp_swc(
        """
        1 1 0 0 0 10 -1
        2 3 0 10 0 2 1
        3 3 0 20 0 1 2
        """
    )

    tree = SwcReader().read(path)
    tree2, report = Morpho.from_swc(path, return_report=True)

    assert tree.root.name == "soma"
    assert len(tree.branches) == 2
    assert tree.branch_by_name("basal_dendrite_0").type == "basal_dendrite"
    assert all(issue.level != "error" for issue in report.issues)
    assert len(tree2.branches) == 2
