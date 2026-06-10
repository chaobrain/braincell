from __future__ import annotations

import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.filter import EmptyRegion, RegionExpr, branch_in

SOURCE_HOC_ENV = "DCN_SOURCE_HOC"
DEFAULT_SOURCE_HOC: Path | None = None

DCN_REGION_NAMES = (
    "soma",
    "axHillock",
    "axIniSeg",
    "axNode",
    "proxDend",
    "distDend",
)


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
class DcnSectionSpec:
    """Parsed DCN source section and its BrainCell branch metadata.

    Parameters
    ----------
    source_name : str
        Original HOC section name.
    branch_name : str
        BrainCell branch name, including the physiological region
        prefix when needed.
    region : str
        DCN physiological region name.
    branch_type : str
        BrainCell branch type.
    parent_source_name : str or None
        Original parent HOC section name.
    parent_branch_name : str or None
        BrainCell branch name for the parent section.
    parent_x : float or None
        Parent attachment location from the HOC ``connect`` statement.
    child_x : float
        Child attachment location from the HOC ``connect`` statement.
    prox, dist : Point3D
        Proximal and distal geometry endpoints.
    depth : int
        Tree depth from the soma.
    source_order : int
        Source HOC declaration order.
    """

    source_name: str
    branch_name: str
    region: str
    branch_type: str
    parent_source_name: str | None
    parent_branch_name: str | None
    parent_x: float | None
    child_x: float
    prox: Point3D
    dist: Point3D
    depth: int
    source_order: int

    @property
    def length_um(self) -> float:
        return math.dist(
            (self.prox.x, self.prox.y, self.prox.z),
            (self.dist.x, self.dist.y, self.dist.z),
        )


@dataclass(frozen=True)
class DcnMorphology:
    """BrainCell morphology plus DCN source-section lookup tables.

    Parameters
    ----------
    morpho : Morphology
        Constructed BrainCell morphology.
    specs : tuple of DcnSectionSpec
        Parsed section metadata in source order.
    branch_name_by_source : dict
        Mapping from HOC section name to BrainCell branch name.
    source_name_by_branch : dict
        Mapping from BrainCell branch name to HOC section name.
    region_by_branch : dict
        Mapping from BrainCell branch name to DCN physiological region.
    """

    morpho: Morphology
    specs: tuple[DcnSectionSpec, ...]
    branch_name_by_source: dict[str, str]
    source_name_by_branch: dict[str, str]
    region_by_branch: dict[str, str]

    @property
    def regions(self) -> dict[str, tuple[str, ...]]:
        return {
            region: tuple(spec.branch_name for spec in self.specs if spec.region == region)
            for region in DCN_REGION_NAMES
        }

    def region(self, name: str) -> RegionExpr:
        return dcn_region(self.morpho, name)


def resolve_source_hoc(source_hoc: str | Path | None = DEFAULT_SOURCE_HOC) -> Path:
    """Resolve the DCN source HOC path.

    Parameters
    ----------
    source_hoc : str or Path or None, optional
        Explicit source HOC path. If omitted, ``DCN_SOURCE_HOC`` is
        used.

    Returns
    -------
    pathlib.Path
        Resolved source HOC path.

    Raises
    ------
    FileNotFoundError
        If no path is provided or the resolved path does not exist.
    """

    raw_path = source_hoc if source_hoc is not None else os.environ.get(SOURCE_HOC_ENV)
    if raw_path is None:
        raise FileNotFoundError(
            f"DCN source HOC is not configured. Pass source_hoc or set {SOURCE_HOC_ENV}."
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"DCN source HOC does not exist: {path}")
    return path


def load_dcn_morphology(source_hoc: str | Path | None = DEFAULT_SOURCE_HOC) -> DcnMorphology:
    """Build the DCN morphology directly from the source HOC file.

    Parameters
    ----------
    source_hoc : str or Path or None, optional
        Source HOC path. If omitted, ``DCN_SOURCE_HOC`` is used.

    Returns
    -------
    DcnMorphology
        Parsed BrainCell morphology and source-section lookup metadata.
    """

    specs = parse_dcn_hoc(source_hoc)
    morpho = _build_morphology(specs)
    return DcnMorphology(
        morpho=morpho,
        specs=tuple(specs),
        branch_name_by_source={spec.source_name: spec.branch_name for spec in specs},
        source_name_by_branch={spec.branch_name: spec.source_name for spec in specs},
        region_by_branch={spec.branch_name: spec.region for spec in specs},
    )


def parse_dcn_hoc(source_hoc: str | Path | None = DEFAULT_SOURCE_HOC) -> tuple[DcnSectionSpec, ...]:
    """Parse source HOC section geometry, topology, and DCN regions.

    Parameters
    ----------
    source_hoc : str or Path or None, optional
        Source HOC path. If omitted, ``DCN_SOURCE_HOC`` is used.

    Returns
    -------
    tuple of DcnSectionSpec
        Parsed sections in source declaration order.
    """

    text = resolve_source_hoc(source_hoc).read_text(encoding="utf-8", errors="replace")
    created_sections = _parse_created_sections(text)
    points = _parse_pt3d(text)
    parents, parent_x = _parse_connects(text)
    source_regions = _parse_section_lists(text)
    return _build_specs(created_sections, points, parents, parent_x, source_regions)


def dcn_region(morpho: Morphology, name: str) -> RegionExpr:
    """Return a BrainCell region selector for one DCN region.

    Parameters
    ----------
    morpho : Morphology
        DCN morphology returned by :func:`load_dcn_morphology`.
    name : str
        Physiological region name.

    Returns
    -------
    RegionExpr
        Region expression selecting all matching branches.

    Raises
    ------
    ValueError
        If ``name`` is not a known DCN region.
    """

    if name not in DCN_REGION_NAMES:
        raise ValueError(f"Unknown DCN region {name!r}; expected one of {DCN_REGION_NAMES!r}.")
    prefix = f"{name}__"
    names = tuple(branch.name for branch in morpho.branches if branch.name == name or branch.name.startswith(prefix))
    if not names:
        return EmptyRegion()
    expr: RegionExpr = branch_in("name", names[0])
    for branch_name in names[1:]:
        expr = expr | branch_in("name", branch_name)
    return expr


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
    for list_name in DCN_REGION_NAMES[1:]:
        pattern = rf"{list_name}\s*=\s*new SectionList\(\)(.*?)(?=\n\n\w+\s*=\s*new SectionList\(\)|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            raise ValueError(f"Missing SectionList block for {list_name}.")
        for sec_name in re.findall(r"^(\w+(?:\[\d+\])?)\s+" + list_name + r"\.append\(\)", match.group(1), re.MULTILINE):
            regions[sec_name] = list_name
    return regions


def _build_specs(
    created_sections: list[str],
    points: dict[str, tuple[Point3D, Point3D]],
    parents: dict[str, str],
    parent_x: dict[str, float],
    source_regions: dict[str, str],
) -> tuple[DcnSectionSpec, ...]:
    if set(created_sections) != set(points):
        raise ValueError("Created sections and pt3d sections do not match.")
    if set(created_sections) - {"soma"} != set(parents):
        raise ValueError("Connect graph must include exactly every non-soma section.")
    if set(created_sections) != set(source_regions):
        missing = sorted(set(created_sections) - set(source_regions))
        raise ValueError(f"Missing DCN region assignments: {missing[:8]}")

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

    branch_name_by_source = {
        section_name: _branch_name(section_name, source_regions[section_name])
        for section_name in topo_order
    }
    specs: list[DcnSectionSpec] = []
    for section_name in topo_order:
        region = source_regions[section_name]
        parent = parents.get(section_name)
        prox, dist = points[section_name]
        specs.append(
            DcnSectionSpec(
                source_name=section_name,
                branch_name=branch_name_by_source[section_name],
                region=region,
                branch_type=_branch_type(region),
                parent_source_name=parent,
                parent_branch_name=None if parent is None else branch_name_by_source[parent],
                parent_x=None if parent is None else parent_x[section_name],
                child_x=0.0,
                prox=prox,
                dist=dist,
                depth=depth[section_name],
                source_order=order[section_name],
            )
        )
    return tuple(specs)


def _branch_name(source_name: str, region: str) -> str:
    if source_name == "soma":
        return "soma"
    match = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", source_name)
    if match is None:
        raise ValueError(f"Cannot convert source section name {source_name!r}.")
    base, index = match.groups()
    suffix = "" if index is None else f"__{index}"
    return f"{region}__{base}{suffix}"


def _branch_type(region: str) -> str:
    if region == "soma":
        return "soma"
    if region in {"axHillock", "axIniSeg", "axNode"}:
        return "axon"
    if region in {"proxDend", "distDend"}:
        return "dendrite"
    raise ValueError(f"Unknown DCN region {region!r}.")


def _build_morphology(specs: tuple[DcnSectionSpec, ...]) -> Morphology:
    if not specs or specs[0].source_name != "soma":
        raise ValueError("DCN specs must start with the soma root.")
    root = specs[0]
    morpho = Morphology.from_root(_branch_from_spec(root), name=root.branch_name)
    for spec in specs[1:]:
        if spec.parent_branch_name is None or spec.parent_x is None:
            raise ValueError(f"Non-root spec {spec.source_name!r} is missing parent information.")
        morpho.attach(
            parent=spec.parent_branch_name,
            child_branch=_branch_from_spec(spec),
            child_name=spec.branch_name,
            parent_x=spec.parent_x,
            child_x=spec.child_x,
        )
    return morpho


def _branch_from_spec(spec: DcnSectionSpec) -> Branch:
    points = np.array(
        [
            [spec.prox.x, spec.prox.y, spec.prox.z],
            [spec.dist.x, spec.dist.y, spec.dist.z],
        ],
        dtype=float,
    )
    radii = np.array([spec.prox.radius, spec.dist.radius], dtype=float)
    return Branch.from_points(
        points=points * u.um,
        radii=radii * u.um,
        type=spec.branch_type,
    )
