# tree_vis.py
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List
import networkx as nx
from graphviz import Digraph
from collections import defaultdict
import random
from _topo_tree import Tree, Branch

if TYPE_CHECKING:
    from _topo_tree import Tree as TreeType

# ---- color utils ----
def lerp_color(c0, c1, t):
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))

def rgb_hex(c):
    return "#{:02x}{:02x}{:02x}".format(*c)

def text_color_for_bg(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.299*r + 0.587*g + 0.114*b
    return "black" if luminance > 140 else "orange"

# ---- demo tree ----
def demo_tree(depth: int = 4, prob_branch: float = 1.0) -> Tree:
    tree = Tree(verbose=False)
    tree.attach(Branch.empty(), label="soma")

    def add_children(parent_label: str, cur_depth: int):
        if cur_depth >= depth:
            return
        if random.random() < prob_branch:
            left_label = parent_label + "L"
            tree.attach(Branch.empty(), parent=parent_label, label=left_label)
            add_children(left_label, cur_depth + 1)
        if random.random() < prob_branch:
            right_label = parent_label + "R"
            tree.attach(Branch.empty(), parent=parent_label, label=right_label)
            add_children(right_label, cur_depth + 1)

    add_children("soma", 0)
    return tree

# ---- TreeVis ----
@dataclass
class TreeVis:
    tree: TreeType

    def depth_groups(self, G, root=None):
        root = root or next(n for n in G.nodes() if G.in_degree(n) == 0)
        depths = nx.single_source_shortest_path_length(G, root)
        groups = defaultdict(list)
        for n, d in depths.items():
            groups[d].append(n)
        return [groups[d] for d in sorted(groups)]

    def peel_groups_safe(self, G):
        H = G.copy()
        result = []
        while H.nodes():
            leaves = [n for n in H.nodes() if H.out_degree(n) == 0]
            result.append(leaves)
            H.remove_nodes_from(leaves)
        return result

    def color_by_groups(self, groups, color_cfg=None):
        color_cfg = color_cfg or {}
        c0 = color_cfg.get("start", (230, 240, 255))
        c1 = color_cfg.get("end", (20, 60, 140))
        reverse = color_cfg.get("reverse", False)
        n = max(len(groups)-1, 1)
        node2color = {}
        for i, g in enumerate(groups):
            t = i/n
            if reverse:
                t = 1-t
            col = rgb_hex(lerp_color(c0, c1, t))
            for v in g:
                node2color[v] = col
        return node2color

    def plot_2d(
        self,
        *,
        layout: str = "depth",
        label_mode: str = "id+label",
        rankdir: str = "LR",
        arrows: bool = False,
        node_shape: str = "circle",
        node_size: float = 0.35,
        font_size: int = 10,
        edge_style: str = "solid",
        color_mode: str = None,
        color_cfg: dict = None,
        save_path: str = None,
        fmt: str = "pdf",
        cleanup: bool = True,
    ):
        G = self.tree.to_graph()
        engine = {"depth": "dot", "radial": "twopi", "force": "sfdp"}.get(layout)
        if engine is None:
            raise ValueError(f"Unknown layout={layout}")

        dot = Digraph("Tree", engine=engine)

        # ---- sfdp layout params ----
        if engine in {"sfdp", "fdp"}:
            dot.attr(size="12,12!", ratio="expand")
            dot.attr(overlap="scalexy", sep="0.6", nodesep="0.6", ranksep="0.6")

        if engine == "dot":
            dot.attr(rankdir=rankdir)

        dot.attr("node",
                 shape=node_shape,
                 width=str(node_size),
                 height=str(node_size),
                 fixedsize="true" if node_shape in ("circle","point") else "false",
                 fontsize=str(font_size))
        dot.attr("edge", style=edge_style, arrowhead="normal" if arrows else "none")

        # ---- node colors ----
        node2color = None
        if color_mode == "depth":
            groups = self.depth_groups(G)
            node2color = self.color_by_groups(groups, color_cfg)
        elif color_mode == "peel":
            groups = self.peel_groups_safe(G)
            node2color = self.color_by_groups(groups, color_cfg)
        elif color_mode is not None:
            raise ValueError(f"Unknown color_mode={color_mode}")

        for nid, attrs in G.nodes(data=True):
            lab = attrs.get("label","")
            if label_mode=="id":
                label, xlabel = str(nid), None
            elif label_mode=="label":
                label, xlabel = str(lab), None
            elif label_mode=="id+label":
                label, xlabel = str(nid), "" if lab==str(nid) else str(lab)
            else:
                raise ValueError(f"Unknown label_mode={label_mode}")

            kwargs = {}
            if node2color and nid in node2color:
                bg = node2color[nid]
                kwargs["fillcolor"] = bg
                kwargs["style"] = "filled"
                kwargs["fontcolor"] = text_color_for_bg(bg)

            dot.node(str(nid), label=label, xlabel=xlabel, **kwargs)

        for u,v in G.edges():
            ekwargs = {}
            if node2color:
                mode = (color_cfg or {}).get("edge","source")
                if mode=="source": ekwargs["color"] = node2color.get(u)
                elif mode=="target": ekwargs["color"] = node2color.get(v)
                elif mode=="uniform": ekwargs["color"] = "#999999"
            dot.edge(str(u), str(v), **ekwargs)

        # ---- save if requested ----
        if save_path:
            dot.render(filename=save_path, format=fmt, cleanup=cleanup)

        return dot

# ---- attach .vis property ----
def _vis_property(self: Tree) -> TreeVis:
    return TreeVis(self)

def install_tree_vis():
    if not hasattr(Tree, "vis"):
        Tree.vis = property(_vis_property)
install_tree_vis()
