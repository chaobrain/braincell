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

        def walk_root_branch() -> tuple[tuple[int, ...], int]:
            root_type = map_swc_type_code(nodes[root_id].type_code)
            point_ids = [root_id]
            current_id = root_id
            while True:
                child_ids = children[current_id]
                if len(child_ids) != 1:
                    break
                child_id = child_ids[0]
                if map_swc_type_code(nodes[child_id].type_code) != root_type:
                    break
                point_ids.append(child_id)
                current_id = child_id
            return tuple(point_ids), current_id

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

        root_point_ids, root_end_id = walk_root_branch()
        root_index = len(branches)
        branches.append(
            _SwcBranch(
                point_ids=root_point_ids,
                branch_type=map_swc_type_code(nodes[root_id].type_code),
                parent_index=None,
                start_node_id=root_id,
            )
        )
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
        branches: list[_SwcBranch] = []
        soma_ids = self._collect_soma_component(nodes, children, root_id)
        ordered_soma_rows = [row for row in rows if row.node_id in soma_ids]
        ordered_soma_ids = tuple(row.node_id for row in ordered_soma_rows)
        if not ordered_soma_rows:
            return self._extract_generic_branches(nodes, children, root_id)

        if len(ordered_soma_rows) == 1:
            root_branch, child_tasks = self._make_single_point_soma_branch(ordered_soma_rows[0], nodes, children)
        elif contour_soma_ids and set(ordered_soma_ids).issubset(contour_soma_ids):
            root_branch, child_tasks = self._make_contour_soma_branch(ordered_soma_rows, nodes, children)
        else:
            is_special, ordered_special_rows = is_special_three_point_soma(ordered_soma_rows)
            if is_special and ordered_special_rows is not None:
                root_branch, child_tasks = self._make_special_three_point_soma_branch(
                    ordered_special_rows,
                    nodes,
                    children,
                )
            else:
                root_branch, child_tasks = self._make_regular_soma_branch(ordered_soma_ids, nodes, children)

        branches.append(root_branch)

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

        for attach, child_id in child_tasks:
            walk_child_branch(0, attach, child_id)
        return branches

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
                child=f"swc_{branch_info.start_node_id}",
                branch=self._make_branch(
                    branch_info,
                    nodes,
                    parent_branch_type=branches[parent_index].branch_type,
                ),
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
            return Branch.xyz_shared(points=points, radii=radii, type=branch.branch_type)

        point_ids = list(branch.point_ids)
        points = [row_point(nodes[node_id]) for node_id in point_ids]
        radii = [row_radius(nodes[node_id]) for node_id in point_ids]

        if branch.attach is not None:
            attach_point, attach_radius = self._attach_geometry(branch.attach, nodes)
            attach_radius_for_child = self._child_attach_radius(
                parent_branch_type=parent_branch_type,
                child_branch_type=branch.branch_type,
                parent_attach_radius=attach_radius,
                first_real_child_radius=radii[0],
            )
            if not np.allclose(points[0], attach_point):
                points.insert(0, attach_point)
                radii.insert(0, attach_radius_for_child)
            else:
                radii[0] = attach_radius_for_child

        return Branch.xyz_shared(
            points=np.array(points, dtype=float) * u.um,
            radii=np.array(radii, dtype=float) * u.um,
            type=branch.branch_type,
        )

    def _child_attach_radius(
        self,
        *,
        parent_branch_type: str | None,
        child_branch_type: str,
        parent_attach_radius: float,
        first_real_child_radius: float,
    ) -> float:
        if parent_branch_type == "soma" and child_branch_type != "soma":
            return float(first_real_child_radius)
        return float(parent_attach_radius)

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
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int], ...]]:
        center = row_point(soma_row)
        radius = max(row_radius(soma_row), MIN_SYNTHETIC_LENGTH_UM)
        offset = np.array([radius, 0.0, 0.0], dtype=float)
        points = (tuple(center - offset), tuple(center), tuple(center + offset))
        radii = (radius, radius, radius)
        child_tasks = tuple((_SwcAttach(point=tuple(center), radius=radius, parent_x=0.5), child_id) for child_id in children[soma_row.node_id])
        return (
            _SwcBranch(
                point_ids=(soma_row.node_id,),
                branch_type="soma",
                parent_index=None,
                start_node_id=soma_row.node_id,
                override_points=points,
                override_radii=radii,
            ),
            child_tasks,
        )

    def _make_special_three_point_soma_branch(
        self,
        soma_rows: tuple[_SwcRow, _SwcRow, _SwcRow],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int], ...]]:
        center_row = soma_rows[0]
        linear_rows = self._linearize_special_three_point_soma_rows(soma_rows)
        child_tasks = tuple(
            (
                _SwcAttach(
                    node_id=center_row.node_id,
                    point=tuple(row_point(center_row)),
                    radius=row_radius(center_row),
                    parent_x=0.5,
                ),
                child_id,
            )
            for child_id in children[center_row.node_id]
            if map_swc_type_code(nodes[child_id].type_code) != "soma"
        )
        return (
            _SwcBranch(
                point_ids=tuple(row.node_id for row in linear_rows),
                branch_type="soma",
                parent_index=None,
                start_node_id=linear_rows[0].node_id,
            ),
            child_tasks,
        )

    def _linearize_special_three_point_soma_rows(
        self,
        soma_rows: tuple[_SwcRow, _SwcRow, _SwcRow],
    ) -> tuple[_SwcRow, _SwcRow, _SwcRow]:
        center_row, side_a_row, side_b_row = soma_rows
        center_point = row_point(center_row)
        side_a_point = row_point(side_a_row)
        side_b_point = row_point(side_b_row)
        direction = side_a_point - center_point
        if np.linalg.norm(direction) == 0.0:
            direction = side_b_point - center_point
        projection_a = float(np.dot(side_a_point - center_point, direction))
        projection_b = float(np.dot(side_b_point - center_point, direction))
        left_row, right_row = (
            (side_a_row, side_b_row) if projection_a <= projection_b else (side_b_row, side_a_row)
        )
        return (left_row, center_row, right_row)

    def _make_regular_soma_branch(
        self,
        soma_ids: tuple[int, ...],
        nodes: dict[int, _SwcRow],
        children: dict[int, list[int]],
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int], ...]]:
        child_tasks: list[tuple[_SwcAttach, int]] = []
        for soma_id in soma_ids:
            for child_id in children[soma_id]:
                if map_swc_type_code(nodes[child_id].type_code) == "soma":
                    continue
                child_tasks.append((_SwcAttach(node_id=soma_id), child_id))
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
    ) -> tuple[_SwcBranch, tuple[tuple[_SwcAttach, int], ...]]:
        center, radius = contour_equivalent_center_radius(soma_rows)
        radius = max(radius, MIN_SYNTHETIC_LENGTH_UM)
        offset = np.array([radius, 0.0, 0.0], dtype=float)
        points = (tuple(center - offset), tuple(center), tuple(center + offset))
        radii = (radius, radius, radius)
        child_tasks: list[tuple[_SwcAttach, int]] = []
        for row in soma_rows:
            for child_id in children[row.node_id]:
                if map_swc_type_code(nodes[child_id].type_code) == "soma":
                    continue
                child_tasks.append((_SwcAttach(point=tuple(center), radius=radius, parent_x=0.5), child_id))
        return (
            _SwcBranch(
                point_ids=tuple(row.node_id for row in soma_rows),
                branch_type="soma",
                parent_index=None,
                start_node_id=soma_rows[0].node_id,
                override_points=points,
                override_radii=radii,
            ),
            tuple(child_tasks),
        )

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
