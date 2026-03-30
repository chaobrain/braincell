from __future__ import annotations

from dataclasses import dataclass, field
import math
from os import PathLike
from pathlib import Path

import numpy as np

from braincell._units import u
from braincell.morpho.branch import Branch
from braincell.morpho.morpho import Morpho
from .rules import apply_swc_rules, raise_for_swc_errors
from .soma import (
    contour_equivalent_center_radius,
    is_special_three_point_soma,
    row_point,
    row_radius,
)
from .types import (
    MIN_SYNTHETIC_LENGTH_UM,
    SwcReadOptions,
    SwcReport,
    _SwcAttach,
    _SwcBranch,
    _SwcContext,
    _SwcRawRow,
    _SwcRow,
    map_swc_type_code,
)


@dataclass(frozen=True)
class SwcReader:
    options: SwcReadOptions = field(default_factory=SwcReadOptions)

    def read(
        self,
        path: str | PathLike[str],
        *,
        return_report: bool = False,
    ) -> Morpho | tuple[Morpho, SwcReport]:
        context = self._run_pipeline(Path(path), mark_fix_applied=True)
        raise_for_swc_errors(context.report, context.path)
        self._build_graph_index(context)
        if context.root_id is None:
            raise ValueError(f"SWC validation failed for {context.path}: missing root.")
        branches = self._extract_branches(
            context.rows,
            context.nodes,
            context.children,
            context.root_id,
            context.contour_soma_ids,
        )
        morpho = self._build_morpho(branches, context.nodes)
        if return_report:
            return morpho, context.report
        return morpho

    def check(self, path: str | PathLike[str]) -> SwcReport:
        return self._run_pipeline(Path(path), mark_fix_applied=False).report

    def _run_pipeline(self, path: Path, *, mark_fix_applied: bool) -> _SwcContext:
        context = _SwcContext(
            path=path,
            options=self.options,
            use_corrections=self.options.standardize_safe_fixes,
            mark_fix_applied=mark_fix_applied,
        )
        context.raw_rows = self._parse_rows(path)
        apply_swc_rules(context)
        return context

    def _parse_rows(self, path: Path) -> list[_SwcRawRow]:
        rows: list[_SwcRawRow] = []
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_SwcRawRow(fields=tuple(line.split()), line_number=line_number))
        return rows

    def _build_graph_index(self, context: _SwcContext) -> None:
        context.nodes = {
            row.node_id: row
            for row in context.rows
            if None not in (row.node_id, row.type_code, row.x, row.y, row.z, row.radius, row.parent_id)
        }
        context.children = {node_id: [] for node_id in context.nodes}
        context.root_ids = []
        for node in context.nodes.values():
            if node.parent_id == -1:
                context.root_ids.append(node.node_id)
                continue
            if node.parent_id in context.children:
                context.children[node.parent_id].append(node.node_id)
        for child_ids in context.children.values():
            child_ids.sort()
        context.root_ids.sort()
        context.root_id = context.root_ids[0] if len(context.root_ids) == 1 else None

    def _extract_branches(
        self,
        rows: list[_SwcRow],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
        root_id: int,
        contour_soma_ids: set[int],
    ) -> list[_SwcBranch]:
        if map_swc_type_code(nodes[root_id].type_code) != "soma":
            return self._extract_generic_branches(nodes, children, root_id)
        return self._extract_soma_branches(rows, nodes, children, root_id, contour_soma_ids)

    def _extract_generic_branches(
        self,
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
        root_id: int,
    ) -> list[_SwcBranch]:
        branches: list[_SwcBranch] = []

        def walk_root_branch() -> tuple[tuple[int, ...], int, tuple[int, ...]]:
            root_type = map_swc_type_code(nodes[root_id].type_code)
            point_ids = [root_id]
            current_id = root_id
            root_side_child_ids: tuple[int, ...] = tuple()

            child_ids = children[current_id]
            same_type_child_ids = [child_id for child_id in child_ids if map_swc_type_code(nodes[child_id].type_code) == root_type]
            if len(child_ids) != 1 and same_type_child_ids:
                main_child_id = same_type_child_ids[0]
                point_ids.append(main_child_id)
                current_id = main_child_id
                root_side_child_ids = tuple(child_id for child_id in child_ids if child_id != main_child_id)

            while True:
                child_ids = children[current_id]
                if len(child_ids) != 1:
                    break
                child_id = child_ids[0]
                if map_swc_type_code(nodes[child_id].type_code) != root_type:
                    break
                point_ids.append(child_id)
                current_id = child_id
            return tuple(point_ids), current_id, root_side_child_ids

        def walk_child_branch(parent_index: int, attach: _SwcAttach, start_node_id: int) -> None:
            branch_type = map_swc_type_code(nodes[start_node_id].type_code)
            point_ids = [start_node_id]
            current_id = start_node_id
            while True:
                child_ids = children[current_id]
                if len(child_ids) != 1:
                    break
                child_id = child_ids[0]
                if map_swc_type_code(nodes[child_id].type_code) != branch_type:
                    break
                point_ids.append(child_id)
                current_id = child_id
            branch_index = len(branches)
            branches.append(
                _SwcBranch(
                    point_ids=tuple(point_ids),
                    branch_type=branch_type,
                    parent_index=parent_index,
                    start_node_id=start_node_id,
                    attach=attach,
                )
            )
            for child_id in children[current_id]:
                walk_child_branch(branch_index, _SwcAttach(node_id=current_id), child_id)

        root_point_ids, root_end_id, root_side_child_ids = walk_root_branch()
        root_index = len(branches)
        branches.append(
            _SwcBranch(
                point_ids=root_point_ids,
                branch_type=map_swc_type_code(nodes[root_id].type_code),
                parent_index=None,
                start_node_id=root_id,
            )
        )
        for child_id in root_side_child_ids:
            walk_child_branch(root_index, _SwcAttach(node_id=root_id), child_id)
        for child_id in children[root_end_id]:
            walk_child_branch(root_index, _SwcAttach(node_id=root_end_id), child_id)
        return branches

    def _extract_soma_branches(
        self,
        rows: list[_SwcRow],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
        root_id: int,
        contour_soma_ids: set[int],
    ) -> list[_SwcBranch]:
        soma_ids = self._collect_soma_component(nodes, children, root_id)
        ordered_soma_rows = [row for row in rows if row.node_id in soma_ids]
        ordered_soma_ids = tuple(row.node_id for row in ordered_soma_rows)
        if not ordered_soma_rows:
            return self._extract_generic_branches(nodes, children, root_id)

        if len(ordered_soma_rows) == 1:
            root_branch, child_tasks = self._make_single_point_soma_branch(ordered_soma_rows[0], nodes, children)
            branches = [root_branch]
        else:
            is_special, ordered_special_rows = is_special_three_point_soma(ordered_soma_rows)
            if is_special and ordered_special_rows is not None:
                root_branch, child_tasks = self._make_special_three_point_soma_branch(
                    ordered_special_rows,
                    nodes,
                    children,
                )
                branches = [root_branch]
            elif self._is_nonbranching_soma_component(soma_ids, children, nodes):
                if contour_soma_ids and set(ordered_soma_ids).issubset(contour_soma_ids):
                    root_branch, child_tasks = self._make_contour_soma_branch(ordered_soma_rows, nodes, children)
                else:
                    root_branch, child_tasks = self._make_regular_soma_branch(ordered_soma_ids, nodes, children)
                branches = [root_branch]
            else:
                branches, child_tasks = self._make_branched_soma_branches(
                    nodes=nodes,
                    children=children,
                    root_id=root_id,
                    soma_ids=soma_ids,
                )

        return self._append_child_subtrees(branches, child_tasks, nodes, children)

    def _append_child_subtrees(
        self,
        branches: list[_SwcBranch],
        child_tasks: tuple[tuple[_SwcAttach, int, int], ...],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> list[_SwcBranch]:
        def walk_child_branch(parent_index: int, attach: _SwcAttach, start_node_id: int) -> None:
            parent_branch_type = branches[parent_index].branch_type
            branch_type = map_swc_type_code(nodes[start_node_id].type_code)
            point_ids = [start_node_id]
            current_id = start_node_id
            side_child_ids: tuple[int, ...] = tuple()
            branch_attach = attach

            if parent_branch_type == "soma":
                child_ids = children[current_id]
                same_type_child_ids = [
                    child_id
                    for child_id in child_ids
                    if map_swc_type_code(nodes[child_id].type_code) == branch_type
                ]
                if len(child_ids) != 1 and same_type_child_ids:
                    main_child_id = same_type_child_ids[0]
                    point_ids.append(main_child_id)
                    current_id = main_child_id
                    side_child_ids = tuple(child_id for child_id in child_ids if child_id != main_child_id)
                    branch_attach = self._con2prox_attach(attach, start_node_id, nodes)

            while True:
                child_ids = children[current_id]
                if len(child_ids) != 1:
                    break
                child_id = child_ids[0]
                if map_swc_type_code(nodes[child_id].type_code) != branch_type:
                    break
                point_ids.append(child_id)
                current_id = child_id
            branch_index = len(branches)
            branches.append(
                _SwcBranch(
                    point_ids=tuple(point_ids),
                    branch_type=branch_type,
                    parent_index=parent_index,
                    start_node_id=start_node_id,
                    attach=branch_attach,
                )
            )
            for child_id in side_child_ids:
                walk_child_branch(branch_index, _SwcAttach(node_id=start_node_id), child_id)
            for child_id in children[current_id]:
                walk_child_branch(branch_index, _SwcAttach(node_id=current_id), child_id)

        for attach, child_id, parent_index in child_tasks:
            walk_child_branch(parent_index, attach, child_id)
        return branches

    def _con2prox_attach(
        self,
        attach: _SwcAttach,
        start_node_id: int,
        nodes: dict[int, _SwcRow],
    ) -> _SwcAttach:
        row = nodes[start_node_id]
        return _SwcAttach(
            node_id=attach.node_id,
            point=tuple(row_point(row)),
            radius=row_radius(row),
            parent_x=attach.parent_x,
        )

    def _build_morpho(
        self,
        branches: list[_SwcBranch],
        nodes: dict[int, _SwcRow],
    ) -> Morpho:
        if not branches:
            raise ValueError("SWC extraction produced no branches.")

        root_branch = self._make_branch(branches[0], nodes)
        root_name = "soma" if branches[0].branch_type == "soma" else None
        tree = Morpho.from_root(root_branch, name=root_name)
        branch_views = {0: tree.root}

        for branch_index, branch_info in enumerate(branches[1:], start=1):
            parent_index = branch_info.parent_index
            if parent_index is None:
                raise ValueError("Non-root SWC branch is missing a parent.")
            parent_view = branch_views[parent_index]
            parent_x = self._attachment_x(branches[parent_index], branch_info.attach, nodes)
            branch_views[branch_index] = tree.attach(
                parent=parent_view,
                child_branch=self._make_branch(
                    branch_info,
                    nodes,
                    parent_branch_type=branches[parent_index].branch_type,
                ),
                child_name=None,
                parent_x=parent_x,
                child_x=0.0,
            )
        return tree

    def _make_branch(
        self,
        branch: _SwcBranch,
        nodes: dict[int, _SwcRow],
        *,
        parent_branch_type: str | None = None,
    ) -> Branch:
        if branch.override_points is not None and branch.override_radii is not None:
            points = np.array(branch.override_points, dtype=float) * u.um
            radii = np.array(branch.override_radii, dtype=float) * u.um
            return Branch.from_points(points=points, radii=radii, type=branch.branch_type)

        point_ids = list(branch.point_ids)
        points = [row_point(nodes[node_id]) for node_id in point_ids]
        radii = [row_radius(nodes[node_id]) for node_id in point_ids]

        if branch.attach is not None:
            attach_point, attach_radius = self._attach_geometry(branch.attach, nodes)
            should_copy_attach = True
            if parent_branch_type == "soma" and branch.attach.parent_x == 0.5 and len(point_ids) > 1:
                should_copy_attach = False
            if np.allclose(points[0], attach_point):
                should_copy_attach = False

            attach_radius_for_child = float(attach_radius)
            if parent_branch_type == "soma":
                attach_radius_for_child = float(radii[0])

            if should_copy_attach:
                points.insert(0, attach_point)
                radii.insert(0, attach_radius_for_child)
            elif np.allclose(points[0], attach_point):
                radii[0] = attach_radius_for_child

        return Branch.from_points(
            points=np.array(points, dtype=float) * u.um,
            radii=np.array(radii, dtype=float) * u.um,
            type=branch.branch_type,
        )

    def _attachment_x(
        self,
        parent_branch: _SwcBranch,
        attach: _SwcAttach | None,
        nodes: dict[int, _SwcRow],
    ) -> float:
        if attach is None:
            return 1.0
        if attach.parent_x is not None:
            return attach.parent_x
        if attach.node_id is None:
            raise ValueError("SWC child attachment is missing a parent anchor node.")

        parent_points = [row_point(nodes[node_id]) for node_id in parent_branch.point_ids]
        if len(parent_points) == 1:
            return 1.0
        attach_idx = parent_branch.point_ids.index(attach.node_id)
        lengths = [np.linalg.norm(parent_points[index + 1] - parent_points[index]) for index in range(len(parent_points) - 1)]
        total = sum(lengths)
        if total <= 0.0:
            return 1.0
        prefix = sum(lengths[:attach_idx])
        return float(prefix / total)

    def _collect_soma_component(
        self,
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
        root_id: int,
    ) -> set[int]:
        soma_ids: set[int] = set()
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            if node_id in soma_ids:
                continue
            node = nodes[node_id]
            if map_swc_type_code(node.type_code) != "soma":
                continue
            soma_ids.add(node_id)
            if node.parent_id in nodes and map_swc_type_code(nodes[node.parent_id].type_code) == "soma":
                stack.append(node.parent_id)
            for child_id in children[node_id]:
                if map_swc_type_code(nodes[child_id].type_code) == "soma":
                    stack.append(child_id)
        return soma_ids

    def _make_single_point_soma_branch(
        self,
        soma_row: _SwcRow,
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int, int], ...]]:
        center = row_point(soma_row)
        radius = max(row_radius(soma_row), MIN_SYNTHETIC_LENGTH_UM)
        root_branch = self._synthetic_soma_branch(
            center=center,
            radius=radius,
            point_ids=(soma_row.node_id,),
            start_node_id=soma_row.node_id,
        )
        child_tasks = tuple(
            (_SwcAttach(point=tuple(center), radius=radius, parent_x=0.5), child_id, 0)
            for child_id in children[soma_row.node_id]
            if map_swc_type_code(nodes[child_id].type_code) != "soma"
        )
        return (
            root_branch,
            child_tasks,
        )

    def _make_special_three_point_soma_branch(
        self,
        soma_rows: tuple[_SwcRow, _SwcRow, _SwcRow],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int, int], ...]]:
        center_row = soma_rows[0]
        center = row_point(center_row)
        radius = max(row_radius(center_row), MIN_SYNTHETIC_LENGTH_UM)
        child_tasks = tuple(
            (
                _SwcAttach(
                    node_id=center_row.node_id,
                    point=tuple(center),
                    radius=radius,
                    parent_x=0.5,
                ),
                child_id,
                0,
            )
            for child_id in children[center_row.node_id]
            if map_swc_type_code(nodes[child_id].type_code) != "soma"
        )
        return (
            self._synthetic_soma_branch(
                center=center,
                radius=radius,
                point_ids=tuple(row.node_id for row in soma_rows),
                start_node_id=center_row.node_id,
            ),
            child_tasks,
        )

    def _make_regular_soma_branch(
        self,
        soma_ids: tuple[int, ...],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int, int], ...]]:
        child_tasks: list[tuple[_SwcAttach, int, int]] = []
        for soma_id in soma_ids:
            for child_id in children[soma_id]:
                if map_swc_type_code(nodes[child_id].type_code) == "soma":
                    continue
                child_tasks.append((_SwcAttach(node_id=soma_id, parent_x=self._section_attach_x(soma_ids, soma_id)), child_id, 0))
        return (
            _SwcBranch(
                point_ids=soma_ids,
                branch_type="soma",
                parent_index=None,
                start_node_id=soma_ids[0],
            ),
            tuple(child_tasks),
        )

    def _make_contour_soma_branch(
        self,
        soma_rows: list[_SwcRow],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int, int], ...]]:
        center, radius = contour_equivalent_center_radius(soma_rows)
        radius = max(radius, MIN_SYNTHETIC_LENGTH_UM)
        child_tasks: list[tuple[_SwcAttach, int, int]] = []
        for row in soma_rows:
            for child_id in children[row.node_id]:
                if map_swc_type_code(nodes[child_id].type_code) == "soma":
                    continue
                child_tasks.append((_SwcAttach(point=tuple(center), radius=radius, parent_x=0.5), child_id, 0))
        return (
            self._synthetic_soma_branch(
                center=center,
                radius=radius,
                point_ids=tuple(row.node_id for row in soma_rows),
                start_node_id=soma_rows[0].node_id,
            ),
            tuple(child_tasks),
        )

    def _make_branched_soma_branches(
        self,
        *,
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
        root_id: int,
        soma_ids: set[int],
    ) -> tuple[list[_SwcBranch], tuple[tuple[_SwcAttach, int, int], ...]]:
        branches: list[_SwcBranch] = []
        child_tasks: list[tuple[_SwcAttach, int, int]] = []

        def walk_section(parent_index: int | None, start_node_id: int, attach: _SwcAttach | None) -> int:
            point_ids = [start_node_id]
            current_id = start_node_id
            root_side_soma_child_ids: tuple[int, ...] = tuple()

            if parent_index is None:
                soma_child_ids = self._soma_child_ids(current_id, children, nodes)
                if len(soma_child_ids) > 1:
                    main_child_id = soma_child_ids[0]
                    point_ids.append(main_child_id)
                    current_id = main_child_id
                    root_side_soma_child_ids = tuple(child_id for child_id in soma_child_ids[1:])

            while True:
                soma_child_ids = self._soma_child_ids(current_id, children, nodes)
                if len(soma_child_ids) != 1:
                    break
                child_id = soma_child_ids[0]
                point_ids.append(child_id)
                current_id = child_id

            branch_index = len(branches)
            branches.append(
                _SwcBranch(
                    point_ids=tuple(point_ids),
                    branch_type="soma",
                    parent_index=parent_index,
                    start_node_id=start_node_id,
                    attach=attach,
                )
            )

            effective_point_ids = self._effective_branch_point_ids(tuple(point_ids), attach)
            for node_id in point_ids:
                parent_x = self._section_attach_x(effective_point_ids, node_id)
                for child_id in children[node_id]:
                    if map_swc_type_code(nodes[child_id].type_code) == "soma":
                        continue
                    child_tasks.append((_SwcAttach(node_id=node_id, parent_x=parent_x), child_id, branch_index))

            for child_id in root_side_soma_child_ids:
                walk_section(branch_index, child_id, _SwcAttach(node_id=start_node_id, parent_x=0.0))
            for child_id in self._soma_child_ids(current_id, children, nodes):
                walk_section(branch_index, child_id, _SwcAttach(node_id=current_id, parent_x=1.0))
            return branch_index

        walk_section(None, root_id, None)
        return branches, tuple(child_tasks)

    def _synthetic_soma_branch(
        self,
        *,
        center: np.ndarray,
        radius: float,
        point_ids: tuple[int, ...],
        start_node_id: int,
    ) -> _SwcBranch:
        offset = np.array([radius, 0.0, 0.0], dtype=float)
        return _SwcBranch(
            point_ids=point_ids,
            branch_type="soma",
            parent_index=None,
            start_node_id=start_node_id,
            override_points=(tuple(center - offset), tuple(center), tuple(center + offset)),
            override_radii=(radius, radius, radius),
        )

    def _section_attach_x(self, point_ids: tuple[int, ...], attach_node_id: int) -> float:
        if attach_node_id == point_ids[0]:
            return 0.0
        if attach_node_id == point_ids[-1]:
            return 1.0
        return 0.5

    def _effective_branch_point_ids(
        self,
        point_ids: tuple[int, ...],
        attach: _SwcAttach | None,
    ) -> tuple[int, ...]:
        if attach is None or attach.node_id is None:
            return point_ids
        if attach.parent_x == 0.5 and len(point_ids) > 1:
            return point_ids
        if point_ids and attach.node_id == point_ids[0]:
            return point_ids
        return (attach.node_id, *point_ids)

    def _is_nonbranching_soma_component(
        self,
        soma_ids: set[int],
        children: dict[int, list[int]],
        nodes: dict[int, _SwcRow],
    ) -> bool:
        return all(len(self._soma_child_ids(node_id, children, nodes)) <= 1 for node_id in soma_ids)

    def _soma_child_ids(
        self,
        node_id: int,
        children: dict[int, list[int]],
        nodes: dict[int, _SwcRow],
    ) -> list[int]:
        return [child_id for child_id in children[node_id] if map_swc_type_code(nodes[child_id].type_code) == "soma"]

    def _attach_geometry(
        self,
        attach: _SwcAttach,
        nodes: dict[int, _SwcRow],
    ) -> tuple[np.ndarray, float]:
        if attach.point is not None and attach.radius is not None:
            return np.array(attach.point, dtype=float), float(attach.radius)
        if attach.node_id is None:
            raise ValueError("SWC attachment is missing both point coordinates and node id.")
        row = nodes[attach.node_id]
        return row_point(row), row_radius(row)
