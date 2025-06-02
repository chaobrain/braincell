# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from functools import partial
from typing import Union

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

def calculate_total_resistance_and_area(
    points: brainstate.typing.Array,
    resistivity: u.Quantity = 100.0
):
    r"""
    Calculate the total axial resistance and surface area of a neurite represented as N-1 frustums 
    (truncated cones) formed by N 3D points.

    Each pair of adjacent points defines a frustum segment, where:
    - The length of the frustum is the Euclidean distance between two 3D points.
    - The radii of the circular ends are given by half of the corresponding diameters.

    Parameters:
    -----------
    points : array-like of shape (N, 4)
        A list or array of N points, where each point is [x, y, z, diameter].
        These points define the 3D structure of the neurite.
    resistivity : float, optional
        The axial resistivity (Ω·cm), default is 100.0.

    Returns:
    --------
    total_resistance : float
        The total axial resistance across all frustum segments (in Ohms).
        For each frustum segment, resistance is computed as:

            R_i = Ra * L_i / (π * r1_i * r2_i)

        where:
            - L_i is the length (height) of the frustum segment i
            - r1_i and r2_i are the radii of the two ends of the frustum
            - Ra is the resistivity

    total_surface_area : float
        The total lateral surface area of the frustum segments (in μm²).
        For each frustum, surface area is computed as:

            A_i = π * (r1_i + r2_i) * s_i

        where:
            - s_i is the slant height of the frustum:
                s_i = sqrt((r2_i - r1_i)² + L_i²)
            - r1_i and r2_i are the radii of the ends
            - L_i is the vertical height

    Notes:
    ------
    - This function assumes that membrane currents and surface areas are associated 
      with the lateral surfaces of the frustums.
    - The resistivity is assumed uniform across all segments.

    Example:
    --------
         points = np.array([[0, 0, 0, 2], [5, 0, 0, 1]])
         R, Area = calculate_total_resistance_and_area(points)
    """
    points = np.asarray(points)
    xyz = points[:, :3]  # Extract the first three columns (x, y, z)
    diameters = points[:, 3]  # Extract the diameter column

    # Calculate the Euclidean distance between adjacent points
    heights = np.linalg.norm(np.diff(xyz, axis=0), axis=1)

    # Calculate the radii of adjacent points
    r1 = diameters[:-1] / 2
    r2 = diameters[1:] / 2

    # Calculate the slant heights (the oblique height)
    slant_heights = np.sqrt(heights ** 2 + (r2 - r1) ** 2)

    # Calculate the surface areas of the frustums
    surface_areas = np.pi * (r1 + r2) * slant_heights
    total_surface_area = np.sum(surface_areas)

    # Calculate the resistances
    resistances = resistivity * heights / (np.pi * r1 * r2)
    total_resistance = np.sum(resistances)

    return total_resistance, total_surface_area

def compute_line_ratios(points: np.ndarray):
    r"""
    Compute the normalized cumulative distance (0 to 1) of each point along a polyline
    defined by 3D coordinates. The output ratio indicates how far along the path 
    each point is, relative to the total length of the line.

    Parameters
    ----------
    points : array-like of shape (N, 3)
        A NumPy array or list where each row is a 3D coordinate (x, y, z) representing 
        a point along a path (e.g., neurite or dendrite structure).

    Returns
    -------
    ratios : np.ndarray of shape (N,)
        An array where each element is the normalized distance from the first point 
        to the current point, i.e.,

            ratios[i] = (distance from points[0] to points[i]) / total_length

        The first point will always be 0.0, and the last point will be 1.0 (unless all points coincide).

    Notes
    -----
    - If all points are identical (zero length), returns an array of zeros.

    Example
    -------
    >>> points = np.array([
    ...     [0, 0, 0],
    ...     [3, 0, 0],
    ...     [6, 4, 0]
    ... ])
    >>> compute_line_ratios(points)
    array([0.0, 0.375, 1.0])

    Explanation:
    - Segment 1: length = 3 (from [0,0,0] to [3,0,0])
    - Segment 2: length = 5 (from [3,0,0] to [6,4,0])
    - Total length = 8
    - Cumulative distances: [0, 3, 8]
    - Normalized: [0/8, 3/8, 8/8] = [0.0, 0.375, 1.0]
    """
    # Convert input to NumPy array if it's not already
    points = np.asarray(points)

    # Calculate the Euclidean distance between adjacent points
    # np.diff computes the difference between adjacent points
    # np.linalg.norm computes the Euclidean norm (distance) for each difference
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    # Calculate the total length
    total_length = np.sum(segment_lengths)
    
    # Compute the cumulative length
    # np.insert adds a 0 at the beginning to represent the first point (distance = 0)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    
    # Normalize by total length to get ratios
    # Handle the case where all points coincide (total_length = 0)
    if total_length > 0:
        ratios = cumulative_lengths / total_length
    else:
        ratios = np.zeros(len(points))
        
    return ratios


def find_ratio_interval(
    ratios: np.ndarray,
    target_ratio: np.ndarray
):
    r"""
    Find the two adjacent indices where the target_ratio falls between in the ratios list.
    If the target_ratio is on the boundary (0 or 1), return valid indices within the range.

    :param ratios: A NumPy array of shape (N,) in increasing order, representing the ratio of each point on the line segment.
    :param target_ratio: The target ratio, which is between [0, 1].
    :return: A tuple (lower_index, upper_index) indicating the two adjacent indices where the target ratio falls.

    Example:
    >>> ratios = np.array([0.0, 0.3, 0.6, 1.0])
    >>> find_ratio_interval(ratios, 0.25)
    (0, 1)
    >>> find_ratio_interval(ratios, 0.6)
    (1, 2)
    >>> find_ratio_interval(ratios, 0.0)
    (0, 1)
    >>> find_ratio_interval(ratios, 1.0)
    (2, 3)
    """
    ratios = np.asarray(ratios)
    N = len(ratios)

    idx = np.searchsorted(ratios, target_ratio) - 1
    idx = np.where(target_ratio <= ratios[0], 0, idx)
    idx = np.where(target_ratio >= ratios[-1], N - 2, idx)
    return idx, idx + 1

def generate_interpolated_nodes(node_pre, nseg: int):
    """
    Generate 2*nseg + 1 interpolated nodes and calculate their coordinates and diameters.

    :param node_pre: A NumPy array of shape (N, 4), where each row represents (x, y, z, diam).
    :param nseg: The number of segments for subdivision; 2*nseg+1 points will be generated.
    :return: A NumPy array of shape (2*nseg+1, 4) containing the interpolated node set.
    """
    node_pre = np.asarray(node_pre)  # Ensure it is a NumPy array
    xyz_pre = node_pre[:, :3]  # Extract the first three columns (x, y, z)
    diam_pre = node_pre[:, 3]  # Extract the diameter column

    # 1. Compute the ratio for node_pre
    ratios_pre = compute_line_ratios(xyz_pre)

    # 2. Generate 2*nseg+1 equally spaced ratios (including 0 and 1)
    ratios_new = np.linspace(0, 1, 2 * nseg + 1)

    # 3. Interpolate for each new ratio
    xyz_new = []
    diam_new = []

    for r in ratios_new:
        # Find the adjacent indices for r in the node_pre ratio
        i1, i2 = find_ratio_interval(ratios_pre, r)
        # Extract the adjacent points' information
        r1, r2 = ratios_pre[i1], ratios_pre[i2]
        x1, y1, z1 = xyz_pre[i1]
        x2, y2, z2 = xyz_pre[i2]
        d1, d2 = diam_pre[i1], diam_pre[i2]

        # Interpolation
        alpha = np.where(r2 != r1, (r - r1) / (r2 - r1), 0)  # Avoid division by zero
        x_new = x1 + alpha * (x2 - x1)
        y_new = y1 + alpha * (y2 - y1)
        z_new = z1 + alpha * (z2 - z1)
        d_new = d1 + alpha * (d2 - d1)

        xyz_new.append([x_new, y_new, z_new])
        diam_new.append(d_new)

    # 4. Combine to form the final node_after
    node_after = np.column_stack([np.asarray(xyz_new),
                                      np.asarray(diam_new)])

    return node_after


def compute_connection_seg(nseg_list, connection_sec_list):
    r"""
    Compute the connections between segments based on the given segment list and connection information.

    :param nseg_list: A list of integers where each element represents the number of segments in each section.
                      For example, [2, 3] means section 0 has 2 segments, section 1 has 3 segments.
    :param connection_sec_list: A list of tuples (child_sec, parent_sec, parent_loc) representing the connection
                                between sections. child_sec is the index of the current section,
                                parent_sec is the index of its parent section (or -1 if root),
                                and parent_loc is the normalized location on the parent section (0 to 1).

                                For example, [(0, -1, -1), (1, 0, 1.0)] means:
                                - section 0 is the root section
                                - section 1 connects to section 0 at its end (location 1.0)

    :return: A list of tuples (seg_index, parent_seg_index, site), where:
             - seg_index is the global index of the segment
             - parent_seg_index is the global index of its parent segment (or -1 if root)
             - site is the relative location (0~1) where this segment connects to its parent

    Example:
    >>> nseg_list = [2, 3]
    >>> connections = [(0, -1, -1), (1, 0, 0.75)]
    >>> compute_connection_seg(nseg_list, connections)
    [(0, -1, -1),   # Segment 0: section 0's first segment, no parent (root)
     (1, 0, 1),     # Segment 1: section 0's second segment, connects to segment 0 (previous)
     (2, 1, 0.5),   # Segment 2: section 1's first segment, connects to segment 1 (which is section 0's second segment),
                    #            because 0.75 * 2 segments = 1.5 => falls in segment 1, at 0.5 of its length
     (3, 2, 1),     # Segment 3: section 1's second segment, connects to previous segment
     (4, 3, 1)]     # Segment 4: section 1's third segment, connects to previous segment
    """
    sec_to_segs = {}
    seg_counter = 0
    n_compartment = np.sum(nseg_list)

    for sec_index, num_segs in enumerate(nseg_list):
        sec_to_segs[sec_index] = list(range(seg_counter, seg_counter + num_segs))
        seg_counter += num_segs

    parent_indices = []
    site_list = []

    for sec_index, seg_list in sec_to_segs.items():
        for relative_position, seg in enumerate(seg_list):
            if relative_position > 0:
                parent_index = seg - 1
                site = 1
            else:
                parent_sec = connection_sec_list[sec_index][1]
                parent_sec_site = connection_sec_list[sec_index][2]
                if parent_sec != -1:
                    position_in_parent_sec = int(np.ceil(nseg_list[parent_sec] * parent_sec_site) - 1)
                    parent_index = sec_to_segs[parent_sec][position_in_parent_sec]
                    site = nseg_list[parent_sec] * parent_sec_site - position_in_parent_sec
                else:
                    parent_index = -1
                    site = -1

            parent_indices.append(parent_index)
            site_list.append(site)

    connection_seg = [(i, parent_indices[i], site_list[i]) for i in range(n_compartment)]
    return connection_seg


def init_coupling_weight_nodes(g_left, g_right, connection):
    r"""
    Initialize the axial conductance (coupling weight) matrix between segments in a multi-compartment neuron model.

    Parameters:
        g_left (list or array): List of left-side axial conductances for each segment. Length = N_segments.
        g_right (list or array): List of right-side axial conductances for each segment. Length = N_segments.
        connection (list of tuples): Each tuple is (child_index, parent_index, site), where:
            - child_index: index of the current segment
            - parent_index: index of the parent segment (-1 if root)
            - site: location on the parent segment where the child connects (typically 1.0 or 0.5)

    Returns:
        np.ndarray: An (N x N) symmetric matrix representing the axial coupling conductance between segments.
                    Element [i, j] is the conductance between segment i and j, or 0 if they are not connected.

    Mechanics:
        - If child connects to the **end** of parent segment (site = 1):
            Use the equivalent parallel conductance formula:
                G_ij = g_i * g_j / (g_i + g_j + ...)

            Including parent and all its children connected at site=1:
                If segments {p, c1, c2, ..., cn} connect at parent end:
                    G_pc1 = g_right[p] * g_left[c1] / (g_right[p] + Σ g_left[ci])
                    G_c1c2 = g_left[c1] * g_left[c2] / (g_right[p] + Σ g_left[ci])

        - If child connects to the **middle** of parent segment (site = 0.5):
            Directly add bidirectional conductance:
                G_pc = g_left[c]

    Example:
    >>> g_left = [1.0, 2.0, 3.0, 4.0]
    >>> g_right = [1.5, 2.5, 3.5, 4.5]
    >>> connection = [
            (0, -1, -1),  # root segment
            (1, 0, 1.0),  # connects to end of segment 0
            (2, 0, 1.0),  # also connects to end of segment 0
            (3, 1, 0.5)   # connects to middle of segment 1
        ]
    >>> A = init_coupling_weight_nodes(g_left, g_right, connection)

    Explanation:
        - Segment 1 and 2 connect to end of segment 0 → sharing junction conductance with parent 0
        - Segment 3 connects to middle of segment 1 → direct g_left[3] added to G[1,3] and G[3,1]

    Returns a 4x4 matrix A where:
        - A[0,1], A[0,2], A[1,2] filled via parallel conductance rule
        - A[1,3] = A[3,1] = g_left[3]
    """

    parent_child_dict = {}
    processed_connection = []
    for child, parent, connection_site in connection:
        processed_connection.append([int(child), int(parent), float(connection_site)])

    for child, parent, connection_site in processed_connection:
        if parent not in parent_child_dict:
            parent_child_dict[parent] = {0.5: [], 1: []}
        if connection_site == 0.5:
            parent_child_dict[parent][0.5].append(child)
        elif connection_site == 1:
            parent_child_dict[parent][1].append(child)

    num_segments = len(connection)

    axial_conductance_matrix = np.zeros((num_segments, num_segments))

    for parent, children_dict in parent_child_dict.items():
        if parent != -1:
            # deal with the situation where connetion site is 1
            children_at_1 = children_dict[1]
            if len(children_at_1) > 0:
                all_nodes_at_1 = [parent] + children_at_1
                denominator_at_1 = (np.sum(np.array([g_left[i] for i in children_at_1])) +
                                    np.array(g_right[parent]))
                for i in all_nodes_at_1:
                    for j in all_nodes_at_1:
                        if i != j:
                            if i == parent:
                                axial_conductance_matrix[i, j] = g_right[i] * g_left[
                                    j] / denominator_at_1  # type: ignore
                            elif j == parent:
                                axial_conductance_matrix[i, j] = g_left[i] * g_right[j] / denominator_at_1
                            else:
                                axial_conductance_matrix[i, j] = g_left[i] * g_left[j] / denominator_at_1

            # deal with the situation where connetion site is 0.5
            children_at_0_5 = children_dict[0.5]
            for child in children_at_0_5:
                axial_conductance_matrix[parent, child] = g_left[child]
                axial_conductance_matrix[child, parent] = g_left[child]

    return axial_conductance_matrix * u.mS


def get_coo_ids_and_values(conductance_matrix):
    """
    Given a dense matrix, return the COO format indices and the corresponding values in a vector.

    :param conductance_matrix: A dense numpy matrix.
    :return: A tuple (coo_ids, values) where:
        - coo_ids is a list of tuples (i, j) representing the non-zero indices.
        - values is a vector containing the non-zero values at those indices.
    """
    row, col = u.math.nonzero(conductance_matrix)  # row and column indices of non-zero elements
    values = conductance_matrix[row, col]
    coo_ids = list(zip(row, col))  # List of (i, j) pairs
    return coo_ids, values


def diffusive_coupling(potentials, coo_ids, conductances):
    """
    Compute the diffusive coupling currents between neurons based on conductance.

    :param potentials: The membrane potential of the neurons.
    :param coo_ids: The COO format of the adjacency matrix.
    :param conductances: The conductances of each connection.
    :return: The output of the operator, which computes the diffusive coupling currents.
    """
    # potentials: [n,]
    #    The membrane potential of neuron.
    #    Should be a 1D array.
    # coo_ids: [m, 2]
    #    The COO format of the adjacency matrix.
    #    Should be a 2D array. Each row is a pair of (i, j).
    #    Note that (i, j) indicates the connection from neuron i to neuron j,
    #    and also the connection from neuron j to i.
    # conductances: [m]
    #    The conductance of each connection.
    #    conductances[i] is the conductance of the connection from coo_ids[i, 0] to coo_ids[i, 1],
    #    and also the connection from coo_ids[i, 1] to coo_ids[i, 0].
    # outs: [n]
    #    The output of the operator, which computes the summation of all differences of potentials.
    #    outs[i] = sum((potentials[i] - potentials[j]) * conductances[j] for j in neighbors of i)

    assert isinstance(potentials, u.Quantity), 'The potentials should be a Quantity.'
    assert isinstance(conductances, u.Quantity), 'The conductances should be a Quantity.'
    # assert potentials.ndim == 1, f'The potentials should be a 1D array. Got {potentials.shape}.'
    assert conductances.shape[-1] == coo_ids.shape[0], ('The length of conductance should be equal '
                                                        'to the number of connections.')
    assert coo_ids.ndim == 2, f'The coo_ids should be a 2D array. Got {coo_ids.shape}.'
    assert conductances.ndim == 1, f'The conductances should be a 1D array. Got {conductances.shape}.'

    # Initialize the output array with zeros
    outs = u.Quantity(u.math.zeros(potentials.shape), unit=potentials.unit * conductances.unit)

    pre_ids = coo_ids[:, 0]
    post_ids = coo_ids[:, 1]

    # Calculate the diffusive coupling based on the conductance (potentials difference * conductance)
    diff = (potentials[..., pre_ids] - potentials[..., post_ids]) * conductances
    outs = outs.at[..., pre_ids].add(-diff)
    outs = outs.at[..., post_ids].add(diff)

    return outs


def get_type_name(type_code):
    
    type_map = {
        1: "soma",
        2: "axon",
        3: "dend",
        4: "apic",
        5: "custom",
    }
    return type_map.get(type_code, f"type{type_code}")