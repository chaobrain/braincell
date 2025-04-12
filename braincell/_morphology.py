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

from typing import Union, Optional, Sequence, Dict, Hashable

import brainstate
import brainunit as u
import numpy as np

from ._typing import SectionName
from ._utils import (
    calculate_total_resistance_and_area,
    generate_interpolated_nodes,
    compute_connection_seg,
    compute_line_ratios,
    init_coupling_weight_nodes,
)

__all__ = [
    'Section',
    'CylinderSection',
    'PointSection',
    'Morphology',
]


class Section(brainstate.util.PrettyObject):
    """Base class for representing a neuron section in compartmental modeling.

    A Section is a fundamental building block that represents a discrete part of a neuron's
    morphology, such as a soma, axon, or dendrite section. It provides the foundation
    for electrical and geometric properties of the neuronal compartment.

    Each section is divided into `nseg` segments, and each segment has computed properties:
        - surface area
        - left axial resistance (to previous segment)
        - right axial resistance (to next segment)

    Attributes
    ----------
    name : Hashable
        The identifier for this section
    nseg : int
        Number of segments the section is divided into
    Ra : u.Quantity[u.ohm * u.cm]
        Axial resistivity of the section
    cm : u.Quantity[u.uF / u.cm ** 2]
        Specific membrane capacitance
    positions : np.ndarray
        3D coordinates of section points
    diam : u.Quantity
        Diameter at each position
    parent : dict or None
        Parent section connection information
    segments : list
        List of dictionaries containing segment properties
    children : set
        Set of child section names connected to this section

    Notes
    -----
    This is an abstract base class that should be subclassed by specific section
    implementations like :py:class:`CylinderSection` and :py:class:`PointSection`.
    """

    def __init__(
        self,
        name: SectionName,
        positions: u.Quantity,
        diam: u.Quantity,
        nseg: int,
        Ra: u.Quantity,
        cm: u.Quantity,
    ):
        """
        Initialize the Section.

        Parameters:
            name (Hashable): Section name identifier.
            length (float, optional): Length of the cylinder (if using simple geometry).
            diam (float, optional): Diameter of the cylinder (if using simple geometry).
            points (list or np.ndarray, optional): Array of shape (N, 4) with [x, y, z, diameter].
            nseg (int): Number of segments to divide the section into.
            Ra (float): Axial resistivity in ohm·cm.
            cm (float): Membrane capacitance in µF/cm².
        """
        self.name = name
        self.nseg = nseg
        self.Ra = Ra
        self.cm = cm
        self.positions = positions
        self.diam = diam
        self.parent: Dict = None
        self.children = set()
        self.segments = []

        self._compute_area_and_resistance()

    def _compute_area_and_resistance(self):
        """
        Divide the section into `nseg` segments and compute per segment:
            - Total surface area
            - Left resistance (from current segment to previous)
            - Right resistance (from current segment to next)

        Segment info is stored as a list of dictionaries in `self.segments`, each containing:
            - section_name (str): The name of the section to which this segment belongs
            - index (int): Segment index within the section
            - area (float): Surface area of the segment
            - R_left (float): Resistance from the segment’s left half
            - R_right (float): Resistance from the segment’s right half
        """

        node_pre = np.hstack((self.positions, self.diam))
        node_after = generate_interpolated_nodes(node_pre, self.nseg)
        node_after = np.asarray(node_after)

        xyz_pre = node_pre[:, :3]
        ratios_pre = compute_line_ratios(xyz_pre)
        ratios_after = np.linspace(0, 1, 2 * self.nseg + 1)

        for i in range(0, len(node_after) - 2, 2):
            r1, r2, r3 = ratios_after[i], ratios_after[i + 1], ratios_after[i + 2]

            # Segment left half: i → i+1
            mask_left = (ratios_pre > r1) & (ratios_pre < r2)
            selected_left = np.vstack([node_after[i], node_pre[mask_left], node_after[i + 1]])

            # Segment right half: i+1 → i+2
            mask_right = (ratios_pre > r2) & (ratios_pre < r3)
            selected_right = np.vstack([node_after[i + 1], node_pre[mask_right], node_after[i + 2]])

            # Compute axial resistance and surface area
            R_left, area_left = calculate_total_resistance_and_area(selected_left, self.Ra)
            R_right, area_right = calculate_total_resistance_and_area(selected_right, self.Ra)

            segment = {
                "section_name": self.name,
                "index": i,
                "area": area_left + area_right,
                "R_left": R_left,
                "R_right": R_right
            }
            self.segments.append(segment)

    def add_parent(self, name: SectionName, loc: float):
        """
        Add a parent connection to this section.

        This method establishes a parent-child relationship by setting the parent of this
        section. It specifies which section is the parent and where along the parent's
        length this section connects.

        Parameters
        ----------
        name : Hashable
            The name of the parent section to connect to.
        loc : float
            The location on the parent section to connect to, ranging from 0.0 (beginning)
            to 1.0 (end).

        Raises
        ------
        ValueError
            If this section already has a different parent
        AssertionError
            If loc is not between 0.0 and 1.0

        Notes
        -----
        This method is primarily called by the Morphology.connect() method rather than
        being used directly.
        """

        if self.parent is not None:
            if self.parent["name"] != name:
                raise ValueError(f"Warning: Section '{self.name}' already has a parent: {self.parent['name']}.")

        assert 0.0 <= loc <= 1.0, "parent_loc must be between 0.0 and 1.0"
        self.parent = {"name": name, "loc": loc}

    def add_child(self, name: SectionName):
        """
        Add a child connection to this section.

        This method registers another section as a child of this section
        by adding the child section's name to this section's children set.

        Parameters
        ----------
        name : Hashable
            The name of the child section to add

        Notes
        -----
        This method is primarily called by the Morphology.connect() method rather than
        being used directly.
        """
        self.children.add(name)


class CylinderSection(Section):
    """A section class representing a cylindrical compartment with uniform diameter.

    This class provides a simplified way to create a cylindrical neuron section
    with uniform diameter throughout its length. The cylinder is represented
    by two points: one at the origin and one at distance 'length' along the x-axis.

    Parameters
    ----------
    name : Hashable
        Unique identifier for the section
    length : u.Quantity
        Length of the cylindrical section
    diam : u.Quantity
        Diameter of the cylindrical section
    nseg : int, optional
        Number of segments to divide the section into, default=1
    Ra : u.Quantity[u.ohm * u.cm], optional
        Axial resistivity of the section, default=100
    cm : u.Quantity[u.uF / u.cm ** 2], optional
        Specific membrane capacitance, default=1.0

    Notes
    -----
    This is a concrete implementation of the abstract Section class specifically
    for cylindrical geometries. It simplifies section creation by requiring only
    length and diameter rather than full 3D point specifications.
    """

    def __init__(
        self,
        name: SectionName,
        length: u.Quantity[u.meter],
        diam: u.Quantity[u.meter],
        nseg: int = 1,
        Ra: u.Quantity = 100 * u.ohm * u.cm,
        cm: u.Quantity = 1.0 * u.uF / u.cm ** 2,
    ):
        assert length > 0, "Length must be positive."
        assert diam > 0, "Diameter must be positive."
        points = np.array([
            [0.0, 0.0, 0.0, diam],
            [length, 0.0, 0.0, diam]
        ])
        positions = points[:, :3]
        diam = points[:, -1].reshape(-1, 1)
        super().__init__(
            name=name,
            positions=positions,
            diam=diam,
            nseg=nseg,
            Ra=Ra,
            cm=cm,
        )


class PointSection(Section):
    """A section class representing a compartment defined by multiple 3D points with varying diameters.

    This class creates a more complex neuronal section defined by a series of points in 3D space,
    each with its own diameter. The points form a sequence of connected frustums that can
    represent detailed morphological structures like dendrites with varying thickness.

    Parameters
    ----------
    name : Hashable
        Unique identifier for the section
    points : u.Quantity[u.meter]
        Array of shape (N, 4) containing points as [x, y, z, diameter]
    nseg : int, optional
        Number of segments to divide the section into, default=1
    Ra : u.Quantity[u.ohm * u.cm], optional
        Axial resistivity of the section, default=100
    cm : u.Quantity[u.uF / u.cm ** 2], optional
        Specific membrane capacitance, default=1.0

    Notes
    -----
    This class allows for more complex and realistic representations of neuronal
    morphology compared to the simplified CylinderSection. The points must include
    at least two points, and all diameters must be positive values.
    """

    def __init__(
        self,
        name: SectionName,
        points: u.Quantity[u.meter],
        nseg: int = 1,
        Ra: u.Quantity = 100 * u.ohm * u.cm,
        cm: u.Quantity = 1.0 * u.uF / u.cm ** 2,
    ):
        """
        Initialize the Section.

        Parameters:
            name (str): Section name identifier.
            points (list or np.ndarray, optional): Array of shape (N, 4) with [x, y, z, diameter].
            nseg (int): Number of segments to divide the section into.
            Ra (float): Axial resistivity in ohm·cm.
            cm (float): Membrane capacitance in µF/cm².
        """
        # Case 1: user provides custom 3D points
        points = u.math.array(points)
        assert points.shape[1] == 4, "points must be shape (N, 4): [x, y, z, diameter]"
        assert u.math.all(points[:, 3] > 0), "All diameters must be positive."
        positions = points[:, :3]
        diam = points[:, -1].reshape(-1, 1)

        super().__init__(
            name=name,
            positions=positions,
            diam=diam,
            nseg=nseg,
            Ra=Ra,
            cm=cm,
        )


class Morphology(brainstate.util.PrettyObject):
    """
    A class representing the morphological structure of a neuron.

    This class provides tools for creating and managing multi-compartmental neuron models,
    where each compartment represents a different part of the neuron (e.g., soma, axon,
    dendrites). It supports both cylindrical sections and more complex 3D point-based sections.

    The Morphology class allows for:
    - Creating different types of neuronal sections
    - Establishing parent-child relationships between sections
    - Batch creation of sections and connections
    - Computing electrical properties like conductance matrices

    Attributes
    ----------
    sections : dict
        Dictionary mapping section names to Section objects
    segments : list
        List of all segments across all sections

    Examples
    --------
    >>> morph = Morphology()
    >>> # Add a cylindrical soma section
    >>> morph.add_cylinder_section('soma', length=20.0 * u.um, diam=20.0 * u.um)
    >>> # Add an axon and connect it to the soma
    >>> morph.add_cylinder_section('axon', length=800.0 * u.um, diam=1.0 * u.um)
    >>> morph.connect('axon', 'soma', 0.0)
    """

    def __init__(self):
        """
        Initializes the Morphology object.
        This model allows for the creation of a multi-compartmental neuron, where each compartment
        represents a different part of the neuron (e.g., soma, axon, dendrite).

        Attributes:
            sections (dict): Dictionary to store sections by their name.
            segments (list): List of all segments across sections, combined.
        """
        self.sections = {}  # Dictionary to store section objects by name
        self.segments = []

    def add_cylinder_section(
        self,
        name: SectionName,
        length: u.Quantity[u.meter],
        diam: u.Quantity[u.meter],
        nseg: int = 1,
        Ra: u.Quantity = 100 * u.ohm * u.cm,
        cm: u.Quantity = 1.0 * u.uF / u.cm ** 2,
    ):
        """
        Create a cylindrical section and add it to the morphology.

        This method creates a simple cylindrical compartment with uniform diameter and adds it
        to the morphology. The cylinder is represented by two points: one at the origin
        and one at distance 'length' along the x-axis.

        Parameters
        ----------
        name : Hashable
            Unique identifier for the section
        length : u.Quantity[u.cm]
            Length of the cylindrical section
        diam : u.Quantity[u.cm]
            Diameter of the cylindrical section
        nseg : int, optional
            Number of segments to divide the section into, default=1
        Ra : u.Quantity[u.ohm * u.cm], optional
            Axial resistivity of the section, default=100
        cm : u.Quantity[u.uF / u.cm ** 2], optional
            Specific membrane capacitance, default=1.0

        Raises
        ------
        ValueError
            If a section with the same name already exists

        Notes
        -----
        After creation, this section can be connected to other sections using the
        `connect` method.
        """
        section = CylinderSection(name, length=length, diam=diam, nseg=nseg, Ra=Ra, cm=cm)
        if name in self.sections:
            raise ValueError(f"Section with name '{name}' already exists.")
        self.sections[name] = section
        self.segments.extend(section.segments)

    def add_point_section(
        self,
        name: SectionName,
        points: u.Quantity[u.meter],
        nseg: int = 1,
        Ra: u.Quantity = 100 * u.ohm * u.cm,
        cm: u.Quantity = 1.0 * u.uF / u.cm ** 2,
    ):
        """
        Create a section defined by custom 3D points and add it to the morphology.

        This method creates a section based on multiple points defining a 3D trajectory with
        varying diameters. Each point is specified in the format [x, y, z, diameter],
        forming a sequence of connected frustums.

        Parameters
        ----------
        name : Hashable
            Unique identifier for the section
        points : u.Quantity[u.cm]
            Array of shape (N, 4) with each point as [x, y, z, diameter]
        nseg : int, optional
            Number of segments to divide the section into, default=1
        Ra : u.Quantity[u.ohm * u.cm], optional
            Axial resistivity of the section, default=100
        cm : u.Quantity[u.uF / u.cm ** 2], optional
            Specific membrane capacitance, default=1.0

        Raises
        ------
        ValueError
            If a section with the same name already exists

        Notes
        -----
        The points array must contain at least two points, and all diameters must be positive.
        After creation, this section can be connected to other sections using the
        `connect` method.
        """
        section = PointSection(name, points=points, nseg=nseg, Ra=Ra, cm=cm)
        if name in self.sections:
            raise ValueError(f"Section with name '{name}' already exists.")
        self.sections[name] = section
        self.segments.extend(section.segments)

    def get_section(self, name: SectionName) -> Optional[Section]:
        """
        Retrieve a section by its name.

        Parameters:
            name (str): The name of the section to retrieve.

        Returns:
            Section object if found, otherwise None.
        """
        return self.sections.get(name, None)

    def connect(
        self,
        child_name: SectionName,
        parent_name: SectionName,
        parent_loc: Union[float, int] = 1.0
    ):
        """
        Connect one section to another, establishing a parent-child relationship.

        This method creates a connection between two sections in the morphology, where
        one section (child) connects to another section (parent) at a specific location
        along the parent's length.

        Parameters
        ----------
        child_name : Hashable
            The name of the child section to be connected
        parent_name : Hashable
            The name of the parent section to which the child connects
        parent_loc : Union[float, int], optional
            The location on the parent section to connect to, ranging from 0.0 (beginning)
            to 1.0 (end), default=1.0

        Raises
        ------
        ValueError
            If either the child or parent section does not exist
        AssertionError
            If parent_loc is not between 0.0 and 1.0

        Notes
        -----
        If the child section already has a parent, the old connection will be removed
        and a warning message will be displayed.
        """

        child = self.get_section(child_name)
        if child is None:
            raise ValueError('Child section does not exist.')

        parent = self.get_section(parent_name)
        if parent is None:
            raise ValueError('Parent section does not exist.')

        # If the child already has a parent, remove the old connection and notify the user
        if child.parent is not None:
            raise ValueError(f"Warning: Section '{child_name}' already has a parent: {child.parent['parent_name']}.")

        # Set the new parent for the child
        child.add_parent(parent.name, parent_loc)

        # Add the child to the new parent's children list
        parent.add_child(child.name)

    def add_multiple_sections(self, section_dicts: Dict):
        """
        Add multiple sections to the morphology in one operation.

        This method allows batch creation of multiple sections by providing a dictionary of
        section specifications. Each section can be either a point-based or cylindrical section
        depending on the parameters provided.

        Parameters
        ----------
        section_dicts : Dict
            A dictionary mapping section names to their specifications. Each specification is a dictionary
            containing either:
            - 'points', 'nseg' (optional), 'Ra' (optional), 'cm' (optional) for point sections, or
            - 'length', 'diam', 'nseg' (optional), 'Ra' (optional), 'cm' (optional) for cylinder sections

        Raises
        ------
        AssertionError
            If section_dicts is not a dictionary or if any section specification is not a dictionary
        ValueError
            If a section specification doesn't contain either 'points' or both 'length' and 'diam'

        Examples
        --------
        >>> morph = Morphology()
        >>> morph.add_multiple_sections({
        ...     'soma': {'length': 20.0, 'diam': 20.0},
        ...     'axon': {'length': 800.0, 'diam': 1.0, 'nseg': 5},
        ...     'dendrite': {'points': [[0,0,0,2], [10,10,0,1.5], [20,20,0,1]]}
        ... })

        Notes
        -----
        This is a convenience method that calls either `add_point_section` or `add_cylinder_section`
        for each section specification based on the parameters provided.
        """
        assert isinstance(section_dicts, dict), 'section_dicts must be a dictionary'

        for section_name, section_data in section_dicts.items():
            assert isinstance(section_data, dict), 'section_data must be a dictionary.'
            if 'points' in section_data:
                self.add_point_section(name=section_name, **section_data)
            elif 'length' in section_data and 'diam' in section_data:
                self.add_cylinder_section(name=section_name, **section_data)
            else:
                raise ValueError('section_data must contain either points or length and diam.')

    def connect_sections(self, connections: Sequence[Sequence]):
        """
        Establish multiple parent-child connections between sections in one operation.

        This method allows for batch connection of multiple sections by providing a sequence
        of connection specifications. Each connection is specified as a tuple or list with
        exactly three elements: (child_name, parent_name, parent_loc).

        Parameters
        ----------
        connections : Sequence[Sequence]
            A sequence of connection specifications, where each specification is a sequence
            containing exactly three elements:
            - child_name: The name of the child section to be connected
            - parent_name: The name of the parent section to connect to
            - parent_loc: The location on the parent section (0.0 to 1.0) where the connection occurs

        Raises
        ------
        AssertionError
            If the connections parameter is not a list or tuple
        ValueError
            If any connection specification does not contain exactly 3 elements

        Examples
        --------
        >>> morph = Morphology()
        >>> # Add some sections first...
        >>> morph.connect_sections([
        ...     ('dendrite1', 'soma', 0.5),
        ...     ('dendrite2', 'soma', 0.7),
        ...     ('axon', 'soma', 0.0)
        ... ])

        Notes
        -----
        This is a convenience method that calls the `connect` method for each specified connection.
        """
        assert isinstance(connections, (tuple, list)), 'connections must be a list or tuple.'
        for sec in connections:
            if len(sec) != 3:
                raise ValueError('connections must contain exactly 3 elements.')
            child_name, parent_name, parent_loc = sec
            self.connect(child_name, parent_name, parent_loc)

    def _connection_sec_list(self):
        """
        Extract section connection information in the form of tuples.

        Returns:
            List of tuples (child_idx, parent_idx, parent_loc) for each section.
        """
        section_names = list(self.sections.keys())
        name_to_idx = {name: idx for idx, name in enumerate(section_names)}

        connections = []
        for child_name, child_section in self.sections.items():
            if child_section.parent is not None:
                parent_name = child_section.parent["name"]
                parent_loc = child_section.parent["loc"]

                child_idx = name_to_idx[child_name]
                parent_idx = name_to_idx[parent_name]

                connections.append((child_idx, parent_idx, parent_loc))
            else:
                child_idx = name_to_idx[child_name]
                connections.append((child_idx, -1, -1))
        return connections

    def construct_conductance_matrix(self):
        """
        Construct the conductance matrix for the model. This matrix represents the conductance
        between sections based on the resistance of each segment and their connectivity.

        The matrix is populated using the left and right conductances of each section segment.
        """

        # TODO

        nseg_list = []
        g_left = []
        g_right = []

        for seg in self.segments:
            g_left.append(1 / (seg['R_left'] * u.ohm * u.cm / u.um))
            g_right.append(1 / (seg['R_right'] * u.ohm * u.cm / u.um))

        for sec in self.sections.values():
            nseg_list.append(sec.nseg)

        connection_sec_list = self._connection_sec_list()
        connection_seg_list = compute_connection_seg(nseg_list, connection_sec_list)

        self.conductance_matrix = init_coupling_weight_nodes(g_left, g_right, connection_seg_list)

    def list_sections(self):
        """List all sections in the model with their properties (e.g., number of segments)."""
        # TODO
        for name, section in self.sections.items():
            print(f"Section: {name}, nseg: {section.nseg}, Points: {section.positions.shape[0]}")

    @classmethod
    def from_swc(self, *args, **kwargs) -> 'Morphology':
        raise NotImplementedError

    @classmethod
    def from_asc(self, *args, **kwargs) -> 'Morphology':
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError
