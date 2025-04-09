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

from __future__ import annotations

from typing import Union, Optional, Callable, Sequence, Tuple

import brainstate
import brainunit as u
import jax
import numpy as np

from braincell._base import HHTypedNeuron, IonChannel
from braincell._protocol import DiffEqState
from braincell._integrator import get_integrator

__all__ = [
    'MultiCompartment',
]


def calculate_total_resistance_and_area(points, resistivity=1.0):
    """
    Calculate the total resistance and total surface area of N-1 frustums formed by N points.
    :param points: A list of N points, each represented as a NumPy array (x, y, z, diam).
    :param resistivity: The resistivity, default is 1.0.
    :return: A tuple of total resistance and total surface area.
    """
    points = np.asarray(points)  # Ensure points is a NumPy array
    xyz = points[:, :3]  # Extract the first three columns (x, y, z)
    diameters = points[:, 3]  # Extract the diameter column
    
    # Calculate the Euclidean distance between adjacent points
    heights = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    
    # Calculate the radii of adjacent points
    r1 = diameters[:-1] / 2
    r2 = diameters[1:] / 2
    
    # Calculate the slant heights (the oblique height)
    slant_heights = np.sqrt(heights**2 + (r2 - r1)**2)
    
    # Calculate the surface areas of the frustums
    surface_areas = np.pi * (r1 + r2) * slant_heights
    total_surface_area = np.sum(surface_areas)
    
    # Calculate the resistances
    resistances = resistivity * heights / (np.pi * r1 * r2)
    total_resistance = np.sum(resistances)
    
    return total_resistance, total_surface_area

def compute_line_ratios(points):
    """
    Compute the ratio of each point along the total length of the line segment.
    :param points: A NumPy array of shape (N, 3), where each row represents a 3D coordinate point (x, y, z).
    :return: A NumPy array of shape (N,) representing the ratio of each point along the line segment.
    """
    points = np.asarray(points)  # Ensure it is a NumPy array

    # Calculate the Euclidean distance between adjacent points
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)

    # Calculate the total length
    total_length = np.sum(segment_lengths)

    if total_length == 0:
        return np.zeros(len(points))  # Handle the case where all points coincide

    # Compute the cumulative length and normalize it
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)  # Insert 0 at the beginning
    ratios = cumulative_lengths / total_length  # Normalize

    return ratios

def find_ratio_interval(ratios, target_ratio):
    """
    Find the two adjacent indices where the target_ratio falls between in the ratios list.
    If the target_ratio is on the boundary (0 or 1), return valid indices within the range.
    
    :param ratios: A NumPy array of shape (N,) in increasing order, representing the ratio of each point on the line segment.
    :param target_ratio: The target ratio, which is between [0, 1].
    :return: A tuple (lower_index, upper_index) indicating the two adjacent indices where the target ratio falls.
    """
    ratios = np.asarray(ratios)
    N = len(ratios)

    if target_ratio <= ratios[0]:
        return 0, 1
    elif target_ratio >= ratios[-1]:
        return N - 2, N - 1
    else:
        idx = np.searchsorted(ratios, target_ratio) - 1
        return idx, idx + 1

def generate_interpolated_nodes(node_pre, nseg):
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
    ratios_new = np.linspace(0, 1, 2*nseg + 1)

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
        alpha = (r - r1) / (r2 - r1) if r2 != r1 else 0  # Avoid division by zero
        x_new = x1 + alpha * (x2 - x1)
        y_new = y1 + alpha * (y2 - y1)
        z_new = z1 + alpha * (z2 - z1)
        d_new = d1 + alpha * (d2 - d1)

        xyz_new.append([x_new, y_new, z_new])
        diam_new.append(d_new)

    # 4. Combine to form the final node_after
    node_after = np.column_stack([xyz_new, diam_new])

    return node_after

def compute_resistance_and_conductance(node_pre, nseg, resistivity=100):
    """
    Calculate the left resistance (resistance_left) and right conductance (conductance_right) for each segment.
    
    :param node_pre: A NumPy array of shape (N, 4), where each row represents (x, y, z, diam).
    :param node_after: A NumPy array of shape (M, 4), where each row represents (x, y, z, diam), with 2*nseg+1 points.
    :param nseg: The number of segments to divide.
    :return: A list of tuples (surface_area, resistance_left, resistance_right) for each segment.
    """
    node_pre = np.asarray(node_pre)
    node_after = generate_interpolated_nodes(node_pre, nseg) 
    node_after = np.asarray(node_after)

    # Extract xyz and diameters
    xyz_pre, diam_pre = node_pre[:, :3], node_pre[:, 3]
    xyz_after, diam_after = node_after[:, :3], node_after[:, 3]

    # Compute the ratio for node_pre and node_after
    ratios_pre = compute_line_ratios(xyz_pre)
    ratios_after = np.linspace(0, 1, 2*nseg + 1)

    results = []

    # Iterate over node_after in steps of 2 to ensure there are nseg groups
    for i in range(0, len(node_after) - 2, 2):
        r1, r2, r3 = ratios_after[i], ratios_after[i+1], ratios_after[i+2]

        # Compute the left resistance (i → i+1), ensuring endpoints are included
        mask_left = (ratios_pre > r1) & (ratios_pre < r2)
        selected_left = np.vstack([node_after[i], node_pre[mask_left], node_after[i+1]])

        # Compute the right resistance (i+1 → i+2), ensuring endpoints are included
        mask_right = (ratios_pre > r2) & (ratios_pre < r3)
        selected_right = np.vstack([node_after[i+1], node_pre[mask_right], node_after[i+2]])

        # Compute resistance
        resistance_left, surface_area_left = calculate_total_resistance_and_area(selected_left, resistivity)
        resistance_right, surface_area_right = calculate_total_resistance_and_area(selected_right, resistivity)
        
        results.append((surface_area_left + surface_area_right, resistance_left, resistance_right))

    return results



def diffusive_coupling(potentials, coo_ids, resistances):
    """
    Compute the diffusive coupling currents between neuron.

    :param potentials: The membrane potential of neuron.
    :param coo_ids: The COO format of the adjacency matrix.
    :param resistances: The weight/resistances of each connection.
    :return: The output of the operator, which computes the diffusive coupling currents.
    """
    # potential: [n,]
    #    The membrane potential of neuron.
    #    Should be a 1D array.
    # coo_ids: [m, 2]
    #    The COO format of the adjacency matrix.
    #    Should be a 2D array. Each row is a pair of (i, j).
    #    Note that (i, j) indicates the connection from neuron i to neuron j,
    #    and also the connection from neuron j to i.
    # resistances: [m]
    #    The weight of each connection.
    #    resistances[i] is the weight of the connection from coo_ids[i, 0] to coo_ids[i, 1],
    #    and also the connection from coo_ids[i, 1] to coo_ids[i, 0].
    # outs: [n]
    #    The output of the operator, which computes the summation of all differences of potentials.
    #    outs[i] = sum((potentials[i] - potentials[j]) / resistances[j] for j in neighbors of i)

    assert isinstance(potentials, u.Quantity), 'The potentials should be a Quantity.'
    assert isinstance(resistances, u.Quantity), 'The conductance should be a Quantity.'
    # assert potentials.ndim == 1, f'The potentials should be a 1D array. Got {potentials.shape}.'
    assert resistances.shape[-1] == coo_ids.shape[0], ('The length of conductance should be equal '
                                                       'to the number of connections.')
    assert coo_ids.ndim == 2, f'The coo_ids should be a 2D array. Got {coo_ids.shape}.'
    assert resistances.ndim == 1, f'The conductance should be a 1D array. Got {resistances.shape}.'

    outs = u.Quantity(u.math.zeros(potentials.shape), unit=potentials.unit / resistances.unit)
    pre_ids = coo_ids[:, 0]
    post_ids = coo_ids[:, 1]
    diff = (potentials[..., pre_ids] - potentials[..., post_ids]) / resistances
    outs = outs.at[..., pre_ids].add(-diff)
    outs = outs.at[..., post_ids].add(diff)
    return outs


def init_coupling_weight(n_compartment, connection, diam, L, Ra):

    assert isinstance(connection, (np.ndarray, jax.Array)), 'The connection should be a numpy/jax array.'


    g_values = np.pi * diam**2 / (2 * Ra * L)

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

    num_segments = len(L)

    axial_conductance_matrix = np.zeros((num_segments, num_segments)) * u.get_unit(g_values)

    for parent, children_dict in parent_child_dict.items():
        if parent != -1:
            # 处理连接位点为 1 的 children
            children_at_1 = children_dict[1]
            all_nodes_at_1 = [parent] + children_at_1
            denominator_at_1 = u.math.sum(u.math.array([g_values[i] for i in all_nodes_at_1]))
            for i in all_nodes_at_1:
                for j in all_nodes_at_1:
                    if i != j:
                        axial_conductance_matrix[i, j] = g_values[i] * g_values[j] / denominator_at_1

            # 处理连接位点为 0.5 的 children
            children_at_0_5 = children_dict[0.5]
            for child in children_at_0_5:
                axial_conductance_matrix[parent, child] = g_values[child]
                axial_conductance_matrix[child, parent] = g_values[child]
                            
    return axial_conductance_matrix

def init_coupling_weight_nodes(g_left, g_right, connection):

    #assert isinstance(connection, (np.ndarray, jax.Array)), 'The connection should be a numpy/jax array.'

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

    axial_conductance_matrix = np.zeros((num_segments, num_segments)) * u.get_unit(g_left)

    for parent, children_dict in parent_child_dict.items():
        if parent != -1:
            # deal with the situation where connetion site is 1
            children_at_1 = children_dict[1]
            if len(children_at_1)>0:
                all_nodes_at_1 = [parent] + children_at_1
                denominator_at_1 = u.math.sum(u.math.array([g_left[i] for i in children_at_1])) + u.math.array(g_right[parent])
                for i in all_nodes_at_1:
                    for j in all_nodes_at_1:
                        if i != j:
                            if i ==parent:
                                axial_conductance_matrix[i, j] = g_right[i] * g_left[j] / denominator_at_1
                            elif j ==parent:
                                axial_conductance_matrix[i, j] = g_left[i] * g_right[j] / denominator_at_1
                            else:
                                axial_conductance_matrix[i, j] = g_left[i] * g_left[j] / denominator_at_1

            # deal with the situation where connetion site is 0.5
            children_at_0_5 = children_dict[0.5]
            for child in children_at_0_5:
                axial_conductance_matrix[parent, child] = g_left[child]
                axial_conductance_matrix[child, parent] = g_left[child]
                            
    return axial_conductance_matrix

def compute_connection_seg(nseg_list, connection_sec):
    """
    Compute the connections between segments based on the given segment list and connection information.
    :param nseg_list: A list of the number of segments for each section.
    :param connection_sec: A list of tuples containing the section index, parent section index, and connection site.
    :return: A list of tuples containing the segment index, parent segment index, and connection site.
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
                parent_sec = connection_sec[sec_index][1]
                parent_sec_site = connection_sec[sec_index][2]
                if parent_sec != -1:
                    position_in_parent_sec = int(np.ceil(nseg_list[parent_sec] * parent_sec_site) - 1)
                    parent_index = sec_to_segs[parent_sec][position_in_parent_sec]
                    site = nseg_list[parent_sec] * parent_sec_site - position_in_parent_sec
                else:
                    parent_index = -1
                    site = -1

            parent_indices.append(parent_index)
            site_list.append(site)

    connection_seg = [(i, parent_indices[i], site_list[i]) for  i in range(n_compartment)]
    return connection_seg


def compute_mor_info(mor_info):
    """
    Compute morphological information such as areas, membrane capacitances, and conductances for each segment.
    :param mor_info: A dictionary containing information about the points, number of segments, resistance (Ra), 
                     membrane capacitance (cm), and connection for each section.
    :return: Lists of areas, membrane capacitances, left and right conductances, segment list, and connection info.
    """

    Area = []
    cm_list = []
    g_left = []
    g_right = []
    nseg_list = []
    connection_sec = []

    for values in mor_info.values():
        
        points = values['points']
        nseg = values['nseg']
        Ra = values['Ra']
        cm = values['cm']
        nseg = values['nseg']
        connection = values['connection']

        areas_i, g_l_i, g_r_i = zip(*compute_resistance_and_conductance(points, nseg, resistivity=Ra))

        Area.extend(areas_i)
        g_left.extend(g_l_i)
        g_right.extend(g_r_i)
        
        nseg_list.append(nseg)
        cm_list.extend([cm]*nseg)
        connection_sec.append(connection)

    return Area, cm_list, g_left, g_right, nseg_list, connection_sec


class MultiCompartment(HHTypedNeuron):
    __module__ = 'braincell.neuron'

    def __init__(
        self,
        size: brainstate.typing.Size,

        # morphology parameters
        connection: Sequence[Tuple[int, int, int]] | np.ndarray,

        # neuron parameters
        Ra: brainstate.typing.ArrayLike = 100. * (u.ohm * u.cm),
        cm: brainstate.typing.ArrayLike = 1.0 * (u.uF / u.cm ** 2),
        diam: brainstate.typing.ArrayLike = 1. * u.um,
        L: brainstate.typing.ArrayLike = 10. * u.um,

        # membrane potentials
        V_th: Union[brainstate.typing.ArrayLike, Callable] = 0. * u.mV,
        V_initializer: Union[brainstate.typing.ArrayLike, Callable] = brainstate.init.Uniform(-70 * u.mV, -60. * u.mV),
        spk_fun: Callable = brainstate.surrogate.ReluGrad(),

        # others
        solver: str | Callable = 'exp_euler',
        name: Optional[str] = None,
        Gl: brainstate.typing.ArrayLike = 0 * (u.mS / u.cm ** 2),  # for splitting 
        El: brainstate.typing.ArrayLike = -65 * u.mV,  # for splitting

        mor_info: dict = None,  # 添加 mor_info 字典参数
        **ion_channels
    ):
        super().__init__(size, name=name, **ion_channels)

        ## for splitting linear term 
        self.Gl = brainstate.init.param(Gl, self.varshape)
        self.El = brainstate.init.param(El, self.varshape)

        # parameters for morphology
        if mor_info is None:
            self.Ra = brainstate.init.param(Ra, self.varshape)
            self.cm = brainstate.init.param(cm, self.varshape)
            self.diam = brainstate.init.param(diam, self.varshape)
            self.L = brainstate.init.param(L, self.varshape)
            self.A = np.pi * self.diam * self.L  # surface area

            connection = np.asarray(connection)
            assert connection.shape[1] == 3, 'The connection should be a sequence of tuples with three elements. '
            'The first elelement is the children node, the second element is the parent node, '
            'the third element is the connection position in the parent node, only for 0.5(middle) and 1(end)'

            self.connection = connection

            if self.connection.max() >= self.n_compartment:
                raise ValueError('The connection should be within the range of compartments. '
                                f'But we got {self.connection.max()} >= {self.n_compartment}.')
            
            self.resistances = init_coupling_weight(self.n_compartment, connection, self.diam, self.L, self.Ra)
            

        else:
            Area, cm, g_left, g_right, nseg_list, connection_sec = compute_mor_info(mor_info)
            connection_seg = compute_connection_seg(nseg_list, connection_sec)

            self.A = Area * u.um**2
            self.cm = cm * u.uF/(u.cm**2)
            self.resistances = init_coupling_weight_nodes(1/(g_left * u.ohm*u.cm/u.um), 1/(g_right * u.ohm*u.cm/u.um), connection_seg)

        # parameters for membrane potentials
        self.V_th = V_th
        self.V_initializer = V_initializer
        self.spk_fun = spk_fun

        # numerical solver
        self.solver = get_integrator(solver)

    @property
    def pop_size(self) -> Tuple[int, ...]:
        return self.varshape[:-1]

    @property
    def n_compartment(self) -> int:
        return self.varshape[-1]

    def init_state(self, batch_size=None):
        self.V = DiffEqState(brainstate.init.param(self.V_initializer, self.varshape, batch_size))
        super().init_state(batch_size)

    def reset_state(self, batch_size=None):
        self.V.value = brainstate.init.param(self.V_initializer, self.varshape, batch_size)
        super().reset_state(batch_size)

    def pre_integral(self, *args):
        """
        Perform pre-integration operations on the neuron's ion channels.

        This method is called before the integration step to prepare the ion channels
        for the upcoming computation. It iterates through all ion channels associated
        with this neuron and calls their respective pre_integral methods.

        Parameters
        -----------
        *args : tuple
            Variable length argument list. Not used in the current implementation
            but allows for future extensibility.

        Returns
        --------
        None
            This method doesn't return any value but updates the internal state
            of the ion channels.
        """
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.pre_integral(self.V.value)

    def compute_derivative(self, I_ext=0. * u.nA):
        """
        Compute the derivative of the membrane potential for the multi-compartment neuron model.

        This method calculates the derivative of the membrane potential by considering
        external currents, axial currents between compartments, synaptic currents,
        and ion channel currents. It also computes the derivatives for all ion channels.

        Parameters
        -----------
        I_ext : Quantity, optional
            External current input to the neuron. Default is 0 nanoamperes.

        Returns
        --------
        None
            This method doesn't return a value but updates the internal state of the neuron,
            specifically the derivative of the membrane potential (self.V.derivative).

        Notes
        ------
        The method performs the following steps:
        1. Normalizes external currents by the compartment surface area.
        2. Calculates axial currents between compartments.
        3. Computes synaptic currents.
        4. Sums up all ion channel currents.
        5. Calculates the final derivative of the membrane potential.
        6. Computes derivatives for all associated ion channels.
        """

        # [ Compute the derivative of membrane potential ]
        # 1. external currents
        I_ext = I_ext / self.A

        # 2.axial currents
        _compute_axial_current = brainstate.environ.get('compute_axial_current', True)
        if _compute_axial_current:
            I_axial = diffusive_coupling(self.V.value, self.connection, self.resistances) / self.A
        else:
            I_axial = self.Gl * self.El #u.Quantity(0., unit=u.get_unit(I_ext))

        # 3. synapse currents
        I_syn = self.sum_current_inputs(0. * u.nA / u.cm ** 2, self.V.value)

        # 4. channel currents
        I_channel = None
        for key, ch in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            I_channel = ch.current(self.V.value) if I_channel is None else (I_channel + ch.current(self.V.value))
            
        # 5. derivatives
        self.V.derivative = (I_ext + I_axial + I_syn + I_channel) / self.cm

        # [ integrate dynamics of ion and ion channel ]
        # check whether the children channel have the correct parents.
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.compute_derivative(self.V.value)

    def post_integral(self, I_ext=0. * u.nA):
        """
        Perform post-integration operations on the neuron's state.

        This method is called after the integration step to update the membrane potential
        and perform any necessary post-integration operations on ion channels.

        Parameters
        -----------
        I_ext : Quantity, optional
            External current input to the neuron. Default is 0 nanoamperes.
            Note: This parameter is not used in the current implementation but is
            included for consistency with other methods.

        Returns
        --------
        None
            This method doesn't return any value but updates the neuron's internal state.
        """
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        for key, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            node.post_integral(self.V.value)

    def update(self, I_ext=0. * u.nA):
        """
        Update the neuron's state and compute spike occurrences.

        This function performs a single update step for the neuron, solving its
        differential equations and determining if a spike has occurred.

        Parameters
        -----------
        I_ext : Quantity, optional
            External current input to the neuron. Default is 0 nanoamperes.

        Returns
        --------
        Quantity
            A binary value indicating whether a spike has occurred (1) or not (0)
            for each compartment of the neuron.
        """
        last_V = self.V.value
        t = brainstate.environ.get('t')
        self.solver(self, t, I_ext)
        return self.get_spike(last_V, self.V.value)

    def get_spike(self, last_V, next_V):
        """
        Determine if a spike has occurred based on the membrane potential change.

        This function calculates whether a spike has occurred by comparing the previous
        and current membrane potentials to the threshold potential.

        Parameters
        -----------
        last_V : Quantity
            The membrane potential at the previous time step.
        next_V : Quantity
            The membrane potential at the current time step.

        Returns
        --------
        Quantity
            A value between 0 and 1 indicating the likelihood of a spike occurrence.
            A value closer to 1 suggests a higher probability of a spike.

        Notes
        ------
        The function uses a surrogate gradient function (self.spk_fun) to approximate
        the non-differentiable spike event, allowing for backpropagation in learning algorithms.
        """
        denom = 20.0 * u.mV
        return (
            self.spk_fun((next_V - self.V_th) / denom) *
            self.spk_fun((self.V_th - last_V) / denom)
        )
