# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import brainunit as u
import numpy as np


class UnionFind:
    '''
    Union-Find (Disjoint Set Union) data structure.

    This data structure efficiently manages disjoint sets and supports:
      - find(x): returns the representative (root) of the set containing x
      - union(x, y): merges the sets containing x and y

    In the context of dendritic or cable models,
    this is used to merge electrically equivalent (equipotential) points.
    For example, if each segment is divided into three points (e.g., at 0, 0.5, 1),
    and connections join two segment at certain positions,
    those positions are physically the same node (equipotential) and should be merged.

    Path compression is used to flatten the structure for fast queries.

    Example
    -------
    >>> uf = UnionFind()
    >>> uf.union(0, 1)
    >>> uf.union(2, 3)
    >>> uf.union(1, 2)
    >>> uf.find(0)
    0
    >>> uf.find(3)
    0
    >>> uf.find(4)
    4

    After these operations:
      - Elements 0, 1, 2, 3 are in the same set with representative 0.
      - Element 4 remains in its own set.

    In a dendritic tree,
    using union(segA_pos, segB_pos) will ensure that the two physically identical points
    are treated as a single equipotential node in subsequent calculations.
    '''

    def __init__(self):
        self.rep = {}

    def find(self, x):
        '''
        Finds the representative ("root") of the set containing x.
        Uses path compression: after calling find(x), self.rep[x] will point directly to the root.
        '''
        if x not in self.rep:
            self.rep[x] = x
        if self.rep[x] != x:
            self.rep[x] = self.find(self.rep[x])
        return self.rep[x]

    def union(self, x, y):
        '''
        Merges the sets containing x and y.
        After union, x and y will have the same representative.
        '''
        self.rep[self.find(y)] = self.find(x)


def nodeid(seg, pos):
    '''
    Assign a unique integer ID to a position on a segment.

    In this scheme, each segment is discretized into three key points:
      - 0   (start)
      - 0.5 (center)
      - 1   (end)

    The global node ID is computed so that each (seg, pos) pair maps to a unique integer.
    This is useful for algorithms (such as Union-Find) that need to quickly check or merge
    equipotential nodes at connections.

    Parameters
    -------
    seg : int
        Segment index.
    pos : float
        Position on the segment (must be 0, 0.5, or 1).

    Returns

    int
        Unique node ID for this (segment, position).

    Example
    -------
    >>> nodeid(0, 0)
    0
    >>> nodeid(0, 0.5)
    1
    >>> nodeid(0, 1)
    2
    >>> nodeid(1, 0)
    3
    >>> nodeid(1, 0.5)
    4
    >>> nodeid(2, 1)
    8
    '''
    return seg * 3 + {0: 0, 0.5: 1, 1: 2}[pos]


def merge_equipotential_segment_nodes(num_segments, parent_id, parentx):
    '''
    Use Union-Find to merge equipotential nodes at segment connections.
    
    Parameters
    ----------
    num_segments : int
        Number of segments.
    parent_id : list of int
        For each segment, index of its parent segment (-1 for root).
    parentx : list of float
        For each segment, the position on the parent segment where the connection is made (e.g., 0, 0.5, or 1).

    Returns
    -------
    uf : UnionFind
        The UnionFind structure after merging all equipotential nodes.

    Note
    ----
    For each segment with a parent, this merges the parent's connection position with the 0-point of the child segment.
    The UnionFind structure is then used to map each node ID to its merged representative.
    '''
    uf = UnionFind()
    for seg in range(num_segments):
        if parent_id[seg] != -1:
            parent = parent_id[seg]
            px = parentx[seg]
            uf.union(nodeid(parent, px), nodeid(seg, 0))
    return uf


def build_segment_internal_edges(num_segments):
    '''
    Create the edges representing internal connectivity within each segment.
    Each segment is discretized into three points, connected as: 0 -- 0.5 -- 1.

    Parameters
    ----------
    num_segments : int

    Returns
    -------
    edges : list of tuple of int
        Each tuple is (node_id_a, node_id_b), representing a directed edge from node_id_a to node_id_b.
    '''
    edges = []
    for seg in range(num_segments):
        n0 = nodeid(seg, 0)
        n05 = nodeid(seg, 0.5)
        n1 = nodeid(seg, 1)
        edges += [(n0, n05), (n05, n1)]
    return edges


def get_merged_edges(edges, uf):
    '''
    Apply Union-Find merging to the internal segment edges,
    producing a list of merged (representative) node edges, eliminating self-loops.

    Parameters
    ----------
    edges : list of tuple of int
        List of original (node_a, node_b) edges.
    uf : UnionFind
        The UnionFind structure containing merged node mappings.

    Returns
    -------
    merged_edges : list of tuple of int
        List of merged node edges (no self-loops).
    '''
    merged_edges = []
    for a, b in edges:
        ma, mb = uf.find(a), uf.find(b)
        if ma != mb:
            merged_edges.append((ma, mb))
    return merged_edges


def build_segment_graph(merged_edges):
    '''
    Construct a directed graph from the merged segment edges using NetworkX.

    Parameters
    ----------
    merged_edges : list of tuple of int

    Returns
    -------
    G : nx.DiGraph
        The directed graph of the merged segment structure.
    '''
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from(merged_edges)
    return G


def classify_segment_nodes(num_segments, uf, G):
    '''
    Classify merged nodes as segment centers, non-center non-leaf, or leaf nodes.

    Parameters
    ----------
    num_segments : int
        Number of segments.
    uf : UnionFind
        UnionFind structure with merged node IDs.
    G : nx.DiGraph
        Directed graph of merged structure.

    Returns
    -------
    center_ids : set of int
        Set of merged node IDs corresponding to all segment centers (0.5 position).
    leaf_ids : list of int
        List of merged node IDs that are leaf nodes (degree 1, not a center).
    noncenter_nonleaf_ids : list of int
        List of merged node IDs that are neither centers nor leaves.
    '''
    segid_to_center = {seg: uf.find(nodeid(seg, 0.5)) for seg in range(num_segments)}
    center_ids = set(segid_to_center.values())
    leaf_ids = [n for n in G.nodes if G.degree[n] == 1 and n not in center_ids]
    noncenter_nonleaf_ids = [n for n in G.nodes if n not in center_ids and n not in leaf_ids]
    return center_ids, leaf_ids, noncenter_nonleaf_ids


def build_segment_node_labels(num_segments, uf):
    '''
    Build string labels for each merged node for visualization.
    All equivalent points (after merging) are grouped.

    Parameters
    ----------
    num_segments : int
    uf : UnionFind

    Returns
    -------
    node_labels : dict
        Key: merged node ID, Value: concatenated label string (one per merged node).
    label_groups : dict
        Key: merged node ID, Value: set of original labels.
    '''
    label_groups = {}
    for seg in range(num_segments):
        for pos in [0, 0.5, 1]:
            nid = uf.find(nodeid(seg, pos))
            label = f"seg{seg}({pos:.1f})"
            label_groups.setdefault(nid, set()).add(label)
    node_labels = {nid: "\n".join(sorted(labels)) for nid, labels in label_groups.items()}
    return node_labels, label_groups


def build_half_segment_maps(num_segments, uf):
    '''
    Construct lookup tables for "half-segments" (connections between points within each segment).
    This is useful for assigning resistances or mapping between merged nodes and physical segment halves.

    Parameters
    ----------
    num_segments : int
    uf : UnionFind

    Returns
    -------
    nid_half_map : dict
        Key: (minid, maxid) tuple of merged node IDs, Value: (segment index, '0-0.5' or '0.5-1')
    node2halves : dict
        Key: merged node ID, Value: set of (segment index, which_half) tuples
    '''
    nid_half_map = dict()
    node2halves = dict()
    for seg in range(num_segments):
        nid0 = uf.find(nodeid(seg, 0))
        nid05 = uf.find(nodeid(seg, 0.5))
        nid1 = uf.find(nodeid(seg, 1))
        for pair, half in [((nid0, nid05), '0-0.5'), ((nid05, nid1), '0.5-1')]:
            pair_sorted = tuple(sorted(pair))
            nid_half_map[pair_sorted] = (seg, half)
            for n in pair:
                node2halves.setdefault(n, set()).add((seg, half))
    return nid_half_map, node2halves


def dhs_group_by_depth(depths_sorted, max_group_size):
    """
    Group row indices by node depth, from deepest (bottom) to shallowest (root),
    ensuring each group contains only nodes at the same depth and does not exceed max_group_size.

    Parameters
    ----------
    depths_sorted : list of int
        List of node depths, sorted according to the matrix row order.
    max_group_size : int
        Maximum allowed size for each group.

    Returns
    -------
    groups : list of list of int
        Each sublist contains row indices grouped together at the same depth.
    """
    n = len(depths_sorted)
    groups = []
    i = n - 1
    while i >= 0:
        group_depth = depths_sorted[i]
        group = []
        # Group together nodes at the same depth, up to max_group_size
        while i >= 0 and depths_sorted[i] == group_depth and len(group) < max_group_size:
            group.append(i)
            i -= 1
        groups.append(sorted(group))
    groups.reverse()
    return groups


def tree_layout(G, root=None, dx=1.5, dy=1.7):
    """
    Compute a simple layered (tree-like) layout for a directed graph.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the tree.
    root : int, optional
        Node id of the root; defaults to the node with in-degree zero.
    dx : float
        Horizontal spacing between nodes.
    dy : float
        Vertical spacing between levels.

    Returns
    -------
    pos : dict
        Dictionary mapping node id to (x, y) coordinates.
    """
    if root is None:
        root = [n for n in G.nodes if G.in_degree(n) == 0][0]
    pos = {}
    width = [0]  # Horizontal position accumulator

    def dfs(node, depth):
        children = list(G.successors(node))
        if not children:
            pos[node] = (width[0], -depth * dy)
            width[0] += dx
        else:
            xs = []
            for c in children:
                dfs(c, depth + 1)
                xs.append(pos[c][0])
            pos[node] = (sum(xs) / len(xs), -depth * dy)

    dfs(root, 0)
    return pos


def plot_tree(G, node_labels, center_ids, noncenter_nonleaf_ids, leaf_ids, root=None):
    """
    Plot a layered tree structure with nodes colored by category.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to plot.
    node_labels : dict
        Node id -> label string.
    center_ids, noncenter_nonleaf_ids, leaf_ids : list or set
        Lists/sets of node ids for each node type (center, non-center non-leaf, leaf).
    root : int, optional
        Node id to use as the root for layout.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.patches as mpatches

    color_map = {}
    for nid in G.nodes:
        if nid in center_ids:
            color_map[nid] = 'cornflowerblue'
        elif nid in noncenter_nonleaf_ids:
            color_map[nid] = 'gold'
        elif nid in leaf_ids:
            color_map[nid] = 'limegreen'
        else:
            color_map[nid] = 'gray'
    node_colors = [color_map[nid] for nid in G.nodes]
    pos = tree_layout(G, root)
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,  # labels=node_labels,
            node_size=100, font_size=4, font_color='black', edge_color='gray', arrowsize=18)
    legend_items = [
        mpatches.Patch(color='cornflowerblue', label='Segment Center Node'),
        mpatches.Patch(color='gold', label='Non-leaf Non-center Node'),
        mpatches.Patch(color='limegreen', label='Leaf Node')
    ]
    plt.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Tree Structure Node Classification (Layered)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def make_branching_tree_parent_id(L=3, n_branches=2, n_levels=3):
    """
    Generate a parent_id array for a branching tree structure.
    Each chain has length L, each branch splits into n_branches, repeated for n_levels.

    Parameters
    ----------
    L : int
        Length of each straight chain segment before branching.
    n_branches : int
        Number of branches at each branch point.
    n_levels : int
        Number of branching levels from root to leaves.

    Returns
    -------
    parent_id : list of int
        parent_id[i] gives the parent segment id for segment i (-1 for root).
    """
    parent_id = [-1]
    node_id = 0
    frontier = [0]
    for level in range(n_levels):
        new_frontier = []
        for src in frontier:
            chain_start = src
            for i in range(L - 1):
                node_id += 1
                parent_id.append(chain_start)
                chain_start = node_id
            for j in range(n_branches):
                node_id += 1
                parent_id.append(chain_start)
                new_frontier.append(node_id)
        frontier = new_frontier
    return parent_id


def dhs_lower_triangularize(A, b, parent_rows):
    """
    Perform in-place back-substitution elimination on matrix A and vector b from deepest node to root,
    using the parent row index array (rowid2parentrowid, -1 means root).
    This transforms A into lower-triangular form suitable for efficient forward substitution.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix. Should be ordered such that parent rows precede their children (e.g. depth order).
    b : ndarray, shape (n,)
        Right-hand side vector.
    parent_rows : list or ndarray of int
        For each row (0 ~ n-1), the parent's row index in A (use -1 for root).

    Returns
    -------
    A, b : tuple of ndarray
        The modified matrix and vector after elimination (lower-triangular form).
    """
    n = len(parent_rows)
    for i in reversed(range(n)):
        parent_row = parent_rows[i]
        if parent_row != -1 and A[parent_row, i] != 0:
            f = A[parent_row, i] / A[i, i]
            A[parent_row, :] -= f * A[i, :]
            b[parent_row] -= f * b[i]
    return A, b


def dhs_back_substitute_lower(A, b, parent_rows):
    """
    Solve a lower-triangular system resulting from DHS elimination,
    using the parent row index array (rowid2parentrowid).

    This performs a forward substitution, assuming A has already been reduced
    to lower-triangular form (i.e., each node depends only on itself and its parent).

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Lower-triangular matrix after elimination.
    b : ndarray, shape (n,)
        Right-hand side vector.
    parent_rows : list or ndarray of int
        For each row (0 ~ n-1), the parent's row index in A (use -1 for root).

    Returns
    -------
    x : ndarray, shape (n,)
        Solution vector, with x[i] corresponding to row i (same order as input).
    """
    import numpy as np
    n = len(parent_rows)
    x = np.zeros(n)
    for i in range(n):
        s = b[i]
        parent_row = parent_rows[i]
        if parent_row != -1:
            s -= A[i, parent_row] * x[parent_row]
        x[i] = s / A[i, i]
    return x


def build_conductance_matrix(G, nid_half_map, seg_resistances):
    """
    Construct the conductance matrix Gmat for all nodes in the graph.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph of the dendritic/cable tree.
    nid_half_map : dict
        Maps node pairs (tuple) to (segment id, '0-0.5' or '0.5-1').
    seg_resistances : list of tuple
        Segment resistance for each segment: [(R0-0.5, R0.5-1), ...]
    unit : physical unit, optional
        Unit to apply at the end, e.g., u.siemens

    Returns
    -------
    Gmat : np.ndarray (optionally with unit)
        Symmetric axial conductance matrix of shape (n, n).
    nodes : list
        List of node ids corresponding to rows/cols of Gmat.
    """
    nodes = sorted(G.nodes)
    n = len(nodes)
    Gmat = np.zeros((n, n), dtype=float)

    for i, nid_i in enumerate(nodes):
        for j, nid_j in enumerate(nodes):
            if i >= j:
                continue
            pair = tuple(sorted([nid_i, nid_j]))
            if pair in nid_half_map:
                sec, which_half = nid_half_map[pair]
                if which_half == '0-0.5':
                    resistance = float(seg_resistances[sec][0])  # 保证为标量
                elif which_half == '0.5-1':
                    resistance = float(seg_resistances[sec][1])
                else:
                    raise ValueError(f"Unexpected segment half '{which_half}' for pair {pair}")
                g = 1.0 / resistance
                Gmat[i, j] = g
                Gmat[j, i] = g

    np.fill_diagonal(Gmat, -Gmat.sum(axis=1))
    return Gmat, nodes


def get_root_and_depths(G):
    """
    Identify the root node and compute node depths (distance from root).

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    root : node id
    depths : dict
        node id -> depth (distance from root)
    """
    import networkx as nx
    root = [n for n in G.nodes if G.in_degree(n) == 0][0]
    depths = nx.single_source_shortest_path_length(G, root)
    return root, depths


def sort_nodes_by_depth(G, depths):
    """
    Sort all nodes in G by their depth (root to deepest).
    Nodes unreachable from root will be placed at the end (with inf depth).

    Returns
    -------
    sorted_nodes : list
        List of node ids sorted by depth.
    """
    all_nodes = sorted(G.nodes)
    sorted_nodes = sorted(all_nodes, key=lambda nid: depths.get(nid, np.inf))
    return sorted_nodes


def reorder_matrix_by_depth(mat, nodes, sorted_nodes):
    """
    Reorder the resistance matrix according to depth order.

    Returns
    -------
    mat_sorted : ndarray
    new_order : list
        List of row indices for reordering.
    """
    node_id2idx = {nid: idx for idx, nid in enumerate(nodes)}
    new_order = [node_id2idx[nid] for nid in sorted_nodes]
    mat_sorted = mat[np.ix_(new_order, new_order)]
    return mat_sorted


def get_depths_sorted(depths, sorted_nodes):
    """
    Generate a list of depths ordered by sorted_nodes.
    """
    return [depths[nid] for nid in sorted_nodes]


def build_parent_dict(G, root):
    """
    Build a dict mapping each node to its parent, using BFS.
    Root will not be present as a key.
    """
    import networkx as nx
    return dict(nx.bfs_predecessors(G, root))


def get_depth_node_idx_map(sorted_nodes):
    """
    Map node id to its row index in sorted_nodes.
    """
    return {nid: idx for idx, nid in enumerate(sorted_nodes)}


def get_parent_rows(sorted_nodes, parent_dict, node_id2rowid):
    """
    For each row (i.e., each node in sorted_nodes), find its parent's row index.
    If the node is a root, set parent index to -1.

    Parameters
    ----------
    sorted_nodes : list
        List of node ids, ordered as in the matrix/algorithm.
    parent_dict : dict
        Mapping from node id to its parent node id (as from BFS).
    node_id2rowid : dict
        Mapping from node id to row index in the sorted_nodes/matrix.

    Returns
    -------
    parent_rows : list of int
        For each row index (0 ~ n-1), the parent's row index in matrix (or -1 for root).
    """
    parent_rows = []
    for i, nid in enumerate(sorted_nodes):
        parent_id = parent_dict.get(nid, None)
        if parent_id is not None:
            parent_row = node_id2rowid[parent_id]
        else:
            parent_row = -1
        parent_rows.append(parent_row)
    return parent_rows


def get_segment2rowid(num_segments, uf, sorted_nodes):
    """
    Map each segment index (midpoint, 0.5) to its corresponding row index in Rmat_sorted.

    Parameters
    ----------
    num_segments : int
        Number of original segments.
    uf : UnionFind
        UnionFind structure after merging equipotential nodes.
    sorted_nodes : list
        Node ids in the row order of Rmat_sorted (length = number of physical nodes).

    Returns
    -------
    segment2rowid : dict
        Key: segment index (0-based), Value: row index in Rmat_sorted.
    """
    # For each segment, find the merged node id at 0.5 (center)
    segid_to_center_nid = {seg: uf.find(nodeid(seg, 0.5)) for seg in range(num_segments)}
    # Map node id to row index in sorted_nodes (i.e., Rmat_sorted)
    nid_to_rowid = {nid: rowid for rowid, nid in enumerate(sorted_nodes)}
    # Build segment to row mapping
    segment2rowid = {seg: nid_to_rowid[segid_to_center_nid[seg]] for seg in range(num_segments)}
    return segment2rowid


def build_uppers_lowers(Gmat, parent_rows):
    n = len(parent_rows)
    lowers = u.math.zeros(n) * u.get_unit(Gmat)
    uppers = u.math.zeros(n) * u.get_unit(Gmat)
    for i in range(n):
        p = parent_rows[i]
        if p == -1:
            lowers = lowers.at[i].set(0 * u.get_unit(Gmat))
            uppers = uppers.at[i].set(0 * u.get_unit(Gmat))
        else:
            lowers = lowers.at[i].set(Gmat[i, p])
            uppers = uppers.at[i].set(Gmat[p, i])
    return lowers, uppers


def build_flipped_comp_edges(dhs_group, parent_rows):
    """
    Build flipped_comp_edges for DHS/Jaxley given groupings and parent_rows.

    Parameters
    ----------
    dhs_group : list of list of int
        Each sublist contains the row indices of nodes in a depth group.
    parent_rows : array-like of int
        For each row, its parent's row index (-1 for root).

    Returns
    -------
    flipped_comp_edges : list of ndarray (n_group, 2)
        Each element: array of [child_row, parent_row] for all pairs in that group.
    """
    flipped_comp_edges = []
    for group in reversed(dhs_group):
        pairs = []
        for child in group:
            parent = parent_rows[child]
            if parent != -1:  # skip root
                pairs.append([child, parent])
        if pairs:  # Only append if non-empty
            flipped_comp_edges.append(np.array(pairs, dtype=int))

    ## padding
    max_len = max(len(group) for group in flipped_comp_edges)
    n_steps = len(flipped_comp_edges)
    # 1. pad 每层到 max_len，空位补 -1 或 0
    padded_edges = []
    edge_masks = []
    for edges in flipped_comp_edges:
        n = len(edges)
        pad_width = max_len - n
        if n > 0:
            pad_block = np.full((pad_width, 2), -1)
            padded = np.vstack([edges, pad_block])
            mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(pad_width, dtype=bool)])
        else:
            padded = np.full((max_len, 2), -1)
            mask = np.zeros(max_len, dtype=bool)
        padded_edges.append(padded)
        edge_masks.append(mask)
    padded_edges = np.stack(padded_edges)  # (n_steps, max_len, 2)
    edge_masks = np.stack(edge_masks)  # (n_steps, max_len)

    return flipped_comp_edges, padded_edges, edge_masks


def preprocess_branching_tree(parent_id, parent_x, seg_resistances, max_group_size=8, plot=False):
    """
    Preprocess a branching tree for DHS matrix algorithms.

    Parameters
    ----------
    parent_id : list of int
        For each segment, the parent segment index (-1 for root).
    parent_x : list
        For each segment, the connection location on parent (not used here, but required for node merging).
    seg_resistances : list of tuple
        For each segment, (R_0-0.5, R_0.5-1).
    max_group_size : int, optional
        Max group size for DHS grouping (default 8).
    plot : bool, optional
        If True, visualize the classified segment tree.

    Returns
    -------
    Rmat_sorted : ndarray
        The resistance matrix after depth-based reordering (n_nodes x n_nodes).
    parent_rows : list of int
        For each node (row in Rmat_sorted), the parent's row index (-1 for root).
    groups : list of list of int
        Each sublist contains row indices (in Rmat_sorted) that form a group for parallel DHS.
    segment2rowid : dict
        Mapping: segment index (0-based) -> row index in Rmat_sorted (corresponds to segment center).
    """
    # Step 1: Merge equipotential nodes at segment connections
    num_segments = len(parent_id)
    uf = merge_equipotential_segment_nodes(num_segments, parent_id, parent_x)

    # Step 2: Build internal edges and merged edges
    edges = build_segment_internal_edges(num_segments)
    merged_edges = get_merged_edges(edges, uf)

    # Step 3: Build directed graph from merged node edges
    G = build_segment_graph(merged_edges)

    # Step 4: (Optional) Build node labels, classify, prepare for visualization
    center_ids, leaf_ids, noncenter_nonleaf_ids = classify_segment_nodes(num_segments, uf, G)
    node_labels, _ = build_segment_node_labels(num_segments, uf)
    nid_half_map, _ = build_half_segment_maps(num_segments, uf)

    if plot:
        plot_tree(G, node_labels, center_ids, noncenter_nonleaf_ids, leaf_ids)

    # Step 5: Construct conductance matrix and get list of all nodes
    unit = u.get_unit(seg_resistances)
    seg_resistances = u.get_magnitude(seg_resistances)
    Gmat, nodes = build_conductance_matrix(G, nid_half_map, seg_resistances)

    # Step 6: Sort nodes by depth, reorder matrix
    root, depths = get_root_and_depths(G)
    sorted_nodes = sort_nodes_by_depth(G, depths)
    Gmat_sorted = reorder_matrix_by_depth(Gmat, nodes, sorted_nodes) * (1 / unit)


    # Step 7: Build parent row indices (rowid2parentrowid) for DHS elimination
    parent_dict = build_parent_dict(G, root)
    node_id2rowid = get_depth_node_idx_map(sorted_nodes)
    parent_rows = get_parent_rows(sorted_nodes, parent_dict, node_id2rowid)

    # Step 8: Group rows by depth (with max group size constraint)
    depths_sorted = [depths[nid] for nid in sorted_nodes]
    dhs_groups = dhs_group_by_depth(depths_sorted, max_group_size)

    # Step 9: Map each segment center (0.5) to the corresponding row index in Rmat_sorted
    segment2rowid = get_segment2rowid(num_segments, uf, sorted_nodes)

    return Gmat_sorted, parent_rows, dhs_groups, segment2rowid

