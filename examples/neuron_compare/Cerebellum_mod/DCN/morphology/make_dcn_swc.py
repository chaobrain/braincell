from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


SOURCE_HOC_ENV = "DCN_SOURCE_HOC"
DEFAULT_SOURCE_HOC: Path | None = None
DEFAULT_OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Point3D:
    """One source HOC ``pt3d`` endpoint.

    Parameters
    ----------
    x, y, z : float
        Cartesian coordinates in micrometers.
    diam : float
        Endpoint diameter in micrometers.
    """

    x: float
    y: float
    z: float
    diam: float

    @property
    def radius(self) -> float:
        return self.diam / 2.0


@dataclass(frozen=True)
class Section:
    """DCN source section prepared for SWC export.

    Parameters
    ----------
    name : str
        Source HOC section name.
    prox, dist : Point3D
        Proximal and distal geometry endpoints.
    parent : str or None
        Parent source section name.
    parent_x : float or None
        Parent attachment location from the HOC ``connect`` statement.
    source_region : str
        Physiological region from source HOC ``SectionList`` entries.
    depth : int
        Tree depth from the soma.
    swc_type : int
        Exported SWC type code.
    swc_node_id : int
        Distal SWC node id for this section.
    neuron_section_index : int
        Expected NEURON import section index.
    """

    name: str
    prox: Point3D
    dist: Point3D
    parent: str | None
    parent_x: float | None
    source_region: str
    depth: int
    swc_type: int
    swc_node_id: int
    neuron_section_index: int

    @property
    def length(self) -> float:
        return math.dist((self.prox.x, self.prox.y, self.prox.z), (self.dist.x, self.dist.y, self.dist.z))


def main() -> None:
    """Generate DCN SWC, section map, and validation summary.

    Notes
    -----
    The source HOC path must be supplied with ``--source-hoc`` or the
    ``DCN_SOURCE_HOC`` environment variable. No user-specific absolute
    path is embedded in the script.
    """

    parser = argparse.ArgumentParser(description="Generate NEURON-roundtrip DCN SWC from DCN_mor.hoc.")
    parser.add_argument("--source-hoc", type=Path, default=DEFAULT_SOURCE_HOC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-neuron-check", action="store_true")
    args = parser.parse_args()

    source_hoc = _resolve_source_hoc(args.source_hoc)

    hoc_text = source_hoc.read_text(encoding="utf-8", errors="replace")
    created_sections = _parse_created_sections(hoc_text)
    points = _parse_pt3d(hoc_text)
    parents, parent_x = _parse_connects(hoc_text)
    source_regions = _parse_section_lists(hoc_text)
    sections = _build_sections(created_sections, points, parents, parent_x, source_regions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    swc_path = args.out_dir / "DCN.swc"
    map_path = args.out_dir / "DCN_section_map.csv"
    summary_path = args.out_dir / "DCN_morphology_summary.json"

    _write_swc(swc_path, sections, parents)
    _write_section_map(map_path, sections)
    summary = _source_summary(sections)
    if not args.skip_neuron_check:
        summary["neuron_roundtrip"] = _run_neuron_checks(source_hoc, swc_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {swc_path}")
    print(f"wrote {map_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _resolve_source_hoc(source_hoc: Path | None) -> Path:
    raw_path = source_hoc if source_hoc is not None else os.environ.get(SOURCE_HOC_ENV)
    if raw_path is None:
        raise FileNotFoundError(
            f"DCN source HOC is not configured. Pass --source-hoc or set {SOURCE_HOC_ENV}."
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"DCN source HOC does not exist: {path}")
    return path


def _parse_created_sections(text: str) -> list[str]:
    sections: list[str] = []
    for match in re.finditer(r"^create\s+(.+)$", text, re.MULTILINE):
        for part in match.group(1).split(","):
            item = part.strip()
            parsed = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", item)
            if parsed is None:
                raise ValueError(f"Cannot parse create target: {item!r}")
            base, count = parsed.groups()
            if count is None:
                sections.append(base)
            else:
                sections.extend(f"{base}[{index}]" for index in range(int(count)))
    return sections


def _parse_pt3d(text: str) -> dict[str, tuple[Point3D, Point3D]]:
    points: dict[str, tuple[Point3D, Point3D]] = {}
    pattern = r"^(\w+(?:\[\d+\])?)\s*\{pt3dclear\(\)(.*?)\}"
    for match in re.finditer(pattern, text, re.MULTILINE):
        sec_name = match.group(1)
        sec_points: list[Point3D] = []
        for values in re.findall(r"pt3dadd\(([^)]*)\)", match.group(2)):
            x, y, z, diam = (float(value.strip()) for value in values.split(","))
            sec_points.append(Point3D(x=x, y=y, z=z, diam=diam))
        if len(sec_points) != 2:
            raise ValueError(f"{sec_name} has {len(sec_points)} pt3d points; expected exactly 2.")
        points[sec_name] = (sec_points[0], sec_points[1])
    return points


def _parse_connects(text: str) -> tuple[dict[str, str], dict[str, float]]:
    parents: dict[str, str] = {}
    parent_x: dict[str, float] = {}
    pattern = r"^connect\s+(\w+(?:\[\d+\])?)\([^)]*\),\s*(\w+(?:\[\d+\])?)\(([^)]*)\)"
    for child, parent, attach_x in re.findall(pattern, text, re.MULTILINE):
        parents[child] = parent
        parent_x[child] = float(attach_x)
    return parents, parent_x


def _parse_section_lists(text: str) -> dict[str, str]:
    regions = {"soma": "soma"}
    for list_name in ("axHillock", "axIniSeg", "axNode", "proxDend", "distDend"):
        pattern = rf"{list_name}\s*=\s*new SectionList\(\)(.*?)(?=\n\n\w+\s*=\s*new SectionList\(\)|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            raise ValueError(f"Missing SectionList block for {list_name}.")
        for sec_name in re.findall(r"^(\w+(?:\[\d+\])?)\s+" + list_name + r"\.append\(\)", match.group(1), re.MULTILINE):
            regions[sec_name] = list_name
    return regions


def _build_sections(
    created_sections: list[str],
    points: dict[str, tuple[Point3D, Point3D]],
    parents: dict[str, str],
    parent_x: dict[str, float],
    source_regions: dict[str, str],
) -> list[Section]:
    if set(created_sections) != set(points):
        raise ValueError("Created sections and pt3d sections do not match.")
    if set(created_sections) - {"soma"} != set(parents):
        raise ValueError("Connect graph must include exactly every non-soma section.")

    children: dict[str, list[str]] = defaultdict(list)
    order = {section: index for index, section in enumerate(created_sections)}
    for child, parent in parents.items():
        children[parent].append(child)
    for sibling_names in children.values():
        sibling_names.sort(key=lambda name: order[name])

    topo_order: list[str] = []
    depth = {"soma": 0}
    queue = deque(["soma"])
    while queue:
        section = queue.popleft()
        topo_order.append(section)
        for child in children[section]:
            depth[child] = depth[section] + 1
            queue.append(child)
    if len(topo_order) != len(created_sections):
        missing = sorted(set(created_sections) - set(topo_order))
        raise ValueError(f"Connect graph is not a single rooted tree. Missing: {missing[:8]}")

    swc_ids = {"soma": 2}
    next_id = 3
    sections: list[Section] = [
        Section(
            name="soma",
            prox=points["soma"][0],
            dist=points["soma"][1],
            parent=None,
            parent_x=None,
            source_region="soma",
            depth=0,
            swc_type=1,
            swc_node_id=2,
            neuron_section_index=0,
        )
    ]
    for neuron_section_index, section_name in enumerate(topo_order[1:], start=1):
        swc_ids[section_name] = next_id
        swc_type = _alternating_swc_type(section_name, source_regions.get(section_name, "distDend"), depth[section_name])
        prox, dist = points[section_name]
        sections.append(
            Section(
                name=section_name,
                prox=prox,
                dist=dist,
                parent=parents[section_name],
                parent_x=parent_x[section_name],
                source_region=source_regions.get(section_name, "distDend"),
                depth=depth[section_name],
                swc_type=swc_type,
                swc_node_id=next_id,
                neuron_section_index=neuron_section_index,
            )
        )
        next_id += 1
    return sections


def _alternating_swc_type(section_name: str, source_region: str, depth: int) -> int:
    if section_name == "soma":
        return 1
    if source_region in {"axHillock", "axIniSeg", "axNode"} or section_name.startswith(("axHill", "axIS", "axIN")):
        return 2 if depth % 2 == 0 else 6
    return 3 if depth % 2 == 0 else 4


def _write_swc(path: Path, sections: list[Section], parents: dict[str, str]) -> None:
    by_name = {section.name: section for section in sections}
    soma = by_name["soma"]
    lines = [
        "# DCN morphology extracted from source DCN_mor.hoc",
        "# NEURON-roundtrip oriented SWC: type codes alternate to prevent Import3d_SWC_read",
        "# from merging contiguous same-type HOC sections.",
        "# Original section lists are preserved in DCN_section_map.csv, not in this SWC.",
        _swc_row(1, 1, soma.prox, -1),
        _swc_row(soma.swc_node_id, 1, soma.dist, 1),
    ]
    for section in sections[1:]:
        parent_name = parents[section.name]
        # The source HOC electrically connects primary dendrites/axon to soma(0.5),
        # but their pt3d geometry starts at the soma proximal coordinate. SWC cannot
        # express both at once, so this morphology file preserves pt3d lengths.
        parent_node_id = 1 if parent_name == "soma" else by_name[parent_name].swc_node_id
        lines.append(_swc_row(section.swc_node_id, section.swc_type, section.dist, parent_node_id))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _swc_row(node_id: int, type_code: int, point: Point3D, parent_id: int) -> str:
    return (
        f"{node_id} {type_code} "
        f"{point.x:.12g} {point.y:.12g} {point.z:.12g} {point.radius:.12g} {parent_id}"
    )


def _write_section_map(path: Path, sections: list[Section]) -> None:
    fieldnames = [
        "swc_node_id",
        "neuron_section_index",
        "source_section",
        "source_region",
        "parent_section",
        "parent_x",
        "depth",
        "swc_type",
        "prox_x",
        "prox_y",
        "prox_z",
        "prox_diam",
        "dist_x",
        "dist_y",
        "dist_z",
        "dist_diam",
        "length_um",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for section in sections:
            writer.writerow(
                {
                    "swc_node_id": section.swc_node_id,
                    "neuron_section_index": section.neuron_section_index,
                    "source_section": section.name,
                    "source_region": section.source_region,
                    "parent_section": "" if section.parent is None else section.parent,
                    "parent_x": "" if section.parent_x is None else f"{section.parent_x:.12g}",
                    "depth": section.depth,
                    "swc_type": section.swc_type,
                    "prox_x": f"{section.prox.x:.12g}",
                    "prox_y": f"{section.prox.y:.12g}",
                    "prox_z": f"{section.prox.z:.12g}",
                    "prox_diam": f"{section.prox.diam:.12g}",
                    "dist_x": f"{section.dist.x:.12g}",
                    "dist_y": f"{section.dist.y:.12g}",
                    "dist_z": f"{section.dist.z:.12g}",
                    "dist_diam": f"{section.dist.diam:.12g}",
                    "length_um": f"{section.length:.12g}",
                }
            )


def _source_summary(sections: list[Section]) -> dict[str, object]:
    region_counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, int] = defaultdict(int)
    for section in sections:
        region_counts[section.source_region] += 1
        type_counts[str(section.swc_type)] += 1
    return {
        "source": {
            "sections": len(sections),
            "connects": len(sections) - 1,
            "total_pt3d": len(sections) * 2,
            "all_sections_have_two_pt3d": True,
            "total_length_um": sum(section.length for section in sections),
            "source_region_counts": dict(sorted(region_counts.items())),
            "swc_type_counts": dict(sorted(type_counts.items())),
        }
    }


def _run_neuron_checks(source_hoc: Path, swc_path: Path) -> dict[str, object]:
    code = r'''
import json
import sys
from neuron import h

mode, path = sys.argv[1], sys.argv[2]
err = None
if mode == "hoc":
    h.load_file(path)
elif mode == "swc":
    h.load_file("import3d.hoc")
    reader = h.Import3d_SWC_read()
    reader.quiet = 1
    reader.input(path)
    err = int(reader.err)
    gui = h.Import3d_GUI(reader, 0)
    gui.instantiate(None)
else:
    raise ValueError(mode)

sections = list(h.allsec())
summary = {
    "err": err,
    "nsec": len(sections),
    "total_n3d": sum(int(h.n3d(sec=sec)) for sec in sections),
    "n3d_hist": {
        str(count): sum(1 for sec in sections if int(h.n3d(sec=sec)) == count)
        for count in sorted({int(h.n3d(sec=sec)) for sec in sections})
    },
    "total_length_um": sum(float(sec.L) for sec in sections),
    "section_names_first10": [sec.name() for sec in sections[:10]],
}
print(json.dumps(summary, sort_keys=True))
'''
    hoc = _run_neuron_probe(code, "hoc", source_hoc)
    swc = _run_neuron_probe(code, "swc", swc_path)
    return {
        "hoc": hoc,
        "swc": swc,
        "matches": {
            "nsec": hoc["nsec"] == swc["nsec"] == 517,
            "total_n3d": hoc["total_n3d"] == swc["total_n3d"] == 1034,
            "n3d_hist": hoc["n3d_hist"] == swc["n3d_hist"] == {"2": 517},
            "total_length_um": math.isclose(
                float(hoc["total_length_um"]),
                float(swc["total_length_um"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
        },
    }


def _run_neuron_probe(code: str, mode: str, path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", code, mode, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
