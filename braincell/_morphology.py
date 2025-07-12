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

from typing import Union, Optional, Sequence, Dict, Hashable, NamedTuple

import brainstate
import brainunit as u
import numpy as np

from ._morphology_dhs_utils import (
    preprocess_branching_tree,
    build_flipped_comp_edges,
)
from ._morphology_from_asc import from_asc
from ._morphology_from_swc import from_swc
from ._morphology_utils import (
    calculate_total_resistance_and_area,
    generate_interpolated_nodes,
    compute_connection_seg,
    compute_line_ratios,
    init_coupling_weight_nodes,
    get_type_name,
)
from ._typing import SectionName

__all__ = [
    'Section',
    'CylinderSection',
    'PointSection',
    'Morphology',
]


class Segment(NamedTuple):
    """
    A named tuple representing a segment of a neuronal section.

    Each segment is a discrete part of a section with specific electrical
    and geometric properties used in compartmental modeling of neurons.

    Attributes
    ----------
    section_name : SectionName
        The identifier of the section this segment belongs to
    index : int
        The position index of this segment within its parent section
    area : u.Quantity[u.meter2]
        Surface area of the segment in square micrometers
    R_left : u.Quantity[u.meter]
        Axial resistance from the segment to its left neighbor
        (previous segment) in micrometers
    R_right : u.Quantity[u.meter]
        Axial resistance from the segment to its right neighbor
        (next segment) in micrometers
    """
    section_name: SectionName
    index: int
    cm: u.Quantity[u.uF / (u.cm ** 2)]
    area: u.Quantity[u.um2]
    R_left: u.Quantity[u.ohm]
    R_right: u.Quantity[u.ohm]


class Section:
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
        positions: u.Quantity[u.um],
        diam: u.Quantity[u.um],
        nseg: int,
        Ra: u.Quantity[u.ohm * u.cm],
        cm: u.Quantity[u.uF / (u.cm ** 2)],
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
        self._nseg = nseg
        self._Ra = Ra
        self._cm = cm
        self.positions = positions
        self.diam = diam
        self.parent: Dict = None
        self.children = set()
        self.segments = []

        self.init_unit()
        self._compute_area_and_resistance()

    @property
    def L(self):
        pos = u.get_magnitude(self.positions)
        return np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1)) * u.um

    @property
    def nseg(self):
        return self._nseg

    @nseg.setter
    def nseg(self, value):
        self._nseg = value
        self._compute_area_and_resistance()

    @property
    def Ra(self):
        return self._Ra

    @Ra.setter
    def Ra(self, value):
        self._Ra = self._ensure_unit(value, u.ohm * u.cm)
        self._compute_area_and_resistance()

    @property
    def cm(self):
        return self._cm

    @cm.setter
    def cm(self, value):
        self._cm = self._ensure_unit(value, u.uF / u.cm ** 2)
        self._compute_area_and_resistance()

    def _ensure_unit(self, value, unit):
        if u.is_unitless(value):
            return value * unit
        else:
            return value.in_unit(unit)

    def init_unit(self):
        self.Ra = self._ensure_unit(self.Ra, u.ohm * u.cm)
        self.cm = self._ensure_unit(self.cm, u.uF / u.cm ** 2)
        self.positions = self._ensure_unit(self.positions, u.um)
        self.diam = self._ensure_unit(self.diam, u.um)

    def __repr__(self):
        n_points = getattr(self.positions, "shape", [len(self.positions)])[0]
        if self.parent and "name" in self.parent and "loc" in self.parent:
            parent_str = f"{self.parent['name']!r}"
            parent_loc = f"{self.parent['loc']}"
        else:
            parent_str = None
            parent_loc = None
        return (
            f"<section_name={self.name!r}, nseg={self.nseg}, points={n_points}, "
            f"Ra={self.Ra}, cm={self.cm}, parent={parent_str}, parent_loc = {parent_loc}>"
        )

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
        self.segments.clear()
        node_pre = np.hstack((u.get_magnitude(self.positions), u.get_magnitude(self.diam)))
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
            R_left, area_left = calculate_total_resistance_and_area(selected_left, u.get_magnitude(self.Ra))
            R_right, area_right = calculate_total_resistance_and_area(selected_right, u.get_magnitude(self.Ra))

            segment = Segment(
                section_name=self.name,
                index=int(i / 2),
                cm=self.cm,
                area=(area_left + area_right) * u.get_unit(self.positions) ** 2,
                R_left=R_left * u.get_unit(self.Ra) * u.get_unit(self.positions) / u.get_unit(self.diam) ** 2,
                R_right=R_right * u.get_unit(self.Ra) * u.get_unit(self.positions) / u.get_unit(self.diam) ** 2,
            )
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
        length: u.Quantity[u.um],
        diam: u.Quantity[u.um],
        nseg: int = 1,
        Ra: u.Quantity = 100 * u.ohm * u.cm,
        cm: u.Quantity = 1.0 * u.uF / u.cm ** 2,
    ):
        assert u.get_magnitude(length) > 0, "Length must be positive."
        assert u.get_magnitude(diam) > 0, "Diameter must be positive."
        positions = np.array([
            [0.0, 0.0, 0.0],
            [u.get_magnitude(length), 0.0, 0.0]
        ]) * u.get_unit(length)
        diam = np.array([
            [u.get_magnitude(diam)],
            [u.get_magnitude(diam)]
        ]) * u.get_unit(diam)
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
        points: u.Quantity[u.um],
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

        points = np.array(u.get_magnitude(points))
        assert points.shape[0] >= 2, "at least have 2 points"
        assert points.shape[1] == 4, "points must be shape (N, 4): [x, y, z, diameter]"
        assert np.all(points[:, 3] > 0), "All diameters must be positive."
        positions = points[:, :3] * u.get_unit(points)
        diam = points[:, -1].reshape((-1, 1)) * u.get_unit(points)

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
        self._conductance_matrix = None
        self._area = None
        self._cm = None
        self._nseg = None
        self._parent_id = None
        self._parent_x = None
        self._seg_ri = None
        self._dhs = None

    @property
    def segments(self):
        return [seg for section in self.sections.values() for seg in section.segments]

    def dhs_init(self, plot=False):
        Gmat_sorted, parent_rows, dhs_groups, segment2rowid = preprocess_branching_tree(
            self.parent_id, self.parent_x, self.seg_ri, max_group_size=32, plot=plot
        )

        flipped_comp_edges = build_flipped_comp_edges(dhs_groups, parent_rows)
        cm_segmid = self.cm
        area_segmid = self.area

        cm = u.math.ones(len(parent_rows)) * u.get_unit(cm_segmid)
        area = u.math.ones(len(parent_rows)) * u.get_unit(area_segmid)
        seg_mid_ids = u.math.array(list(segment2rowid.values()))
        cm[seg_mid_ids] = cm_segmid
        area[seg_mid_ids] = area_segmid

        Gmat_sorted = -Gmat_sorted / (cm * area)[:, u.math.newaxis]

        n = len(parent_rows)
        lowers = u.math.zeros(n) * u.get_unit(Gmat_sorted)
        uppers = u.math.zeros(n) * u.get_unit(Gmat_sorted)

        for i in range(n):
            p = parent_rows[i]
            if p == -1:
                lowers = lowers.at[i].set(0 * u.get_unit(Gmat_sorted))
                uppers = uppers.at[i].set(0 * u.get_unit(Gmat_sorted))
            else:
                lowers = lowers.at[i].set(Gmat_sorted[i, p])
                uppers = uppers.at[i].set(Gmat_sorted[p, i])

        diags = u.math.diag(Gmat_sorted)

        parent_lookup = u.math.array(parent_rows + [-1])
        internal_node_inds = u.math.array(list(segment2rowid.values()))

        self.diags = diags
        self.uppers = uppers
        self.lowers = lowers
        self.flipped_comp_edges = flipped_comp_edges
        self.parent_lookup = parent_lookup
        self.internal_node_inds = internal_node_inds

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

    def add_point_section(
        self,
        name: SectionName,
        points: u.Quantity[u.meter],
        nseg: int = 1,
        Ra: u.Quantity = 100,
        cm: u.Quantity = 1.0,
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

        nseg_list = []
        g_left = []
        g_right = []

        for seg in self.segments:
            g_left.append((1 / seg.R_left).to(u.siemens).magnitude)
            g_right.append((1 / seg.R_right).to(u.siemens).magnitude)

        for sec in self.sections.values():
            nseg_list.append(sec.nseg)

        connection_sec_list = self._connection_sec_list()
        connection_seg_list, _, _ = compute_connection_seg(nseg_list, connection_sec_list)
        self._conductance_matrix = init_coupling_weight_nodes(g_left, g_right, connection_seg_list)

    def construct_area(self):
        area_list = []
        for seg in self.segments:
            area_list.append(seg.area)
        self._area = u.math.array(area_list)

    def construct_cm(self):
        cm_list = []
        for seg in self.segments:
            cm_list.append(seg.cm)
        self._cm = u.math.array(cm_list)

    def construct_nseg(self):
        nseg_list = []
        for sec in self.sections.values():
            nseg_list.append(sec.nseg)
        self._nseg = nseg_list

    def construct_seg_pid_px(self):
        connection_sec_list = self._connection_sec_list()
        _, pid, px = compute_connection_seg(self.nseg, connection_sec_list)
        self._parent_id = pid
        self._parent_x = px

    def construct_seg_ri(self):
        sec_ri = []
        for seg in self.segments:
            sec_ri.append((seg.R_left, seg.R_right))
        self._seg_ri = u.math.array(sec_ri)

    @property
    def seg_ri(self):
        self.construct_seg_ri()
        return self._seg_ri

    @property
    def nseg(self):
        self.construct_nseg()
        return self._nseg

    @property
    def parent_id(self):
        self.construct_seg_pid_px()
        return self._parent_id

    @property
    def parent_x(self):
        self.construct_seg_pid_px()
        return self._parent_x

    @property
    def conductance_matrix(self):
        self.construct_conductance_matrix()
        return self._conductance_matrix

    @property
    def area(self):
        self.construct_area()
        return self._area

    @property
    def cm(self):
        self.construct_cm()
        return self._cm

    def list_sections(self):
        """List all sections in the model with their properties (e.g., number of segments)."""
        # TODO
        for name, section in self.sections.items():
            print(f"Section: {name}, nseg: {section.nseg}, Points: {section.positions.shape[0]}")

    @classmethod
    def from_swc(cls, filename):
        """
        Class method to create a Morphology object from an SWC file (factory method).
        
        Parameters
        ----------
        filename : str
            Path to the SWC file
            
        Returns
        -------
        Morphology
            A Morphology object created from the SWC file
        """
        morphology = cls()

        sections, section_dicts = from_swc(filename)

        # Add all sections using add_multiple_sections method
        morphology.add_multiple_sections(section_dicts)

        # Prepare connection information and establish connections
        connections = []
        for swc_section in sections:
            if swc_section.parentsec is not None:
                child_name = f"{get_type_name(swc_section.type)}_{swc_section.id}"
                parent_name = f"{get_type_name(swc_section.parentsec.type)}_{swc_section.parentsec.id}"
                parent_loc = swc_section.parentx  # Connection position
                connections.append((child_name, parent_name, parent_loc))
        morphology.connect_sections(connections)
        return morphology

    @classmethod
    def from_asc(cls, filename):
        """
        Class method to create a Morphology object from an ASC file (factory method).

        Parameters
        ----------
        filename : str
            Path to the ASC file

        Returns
        -------
        Morphology
            A Morphology object created from the ASC file
        """
        morphology = cls()

        section_dicts, sections, section_id_map = from_asc(filename)

        # Add all sections
        morphology.add_multiple_sections(section_dicts)

        # Prepare and add connection info
        connections = []
        for sec in sections:
            if sec.parent_id is not None:
                child_name = section_id_map[sec.sec_id]
                parent_name = section_id_map[sec.parent_id]
                parent_loc = getattr(sec, "parent_x", 0.0)  # Use 0.0 if attribute missing
                connections.append((child_name, parent_name, parent_loc))
        morphology.connect_sections(connections)
        return morphology

    def visualize(self):
        """
        Visualize the morphology in 3D.

        If the morphology was loaded from an SWC file, uses the SWC visualization.
        Otherwise, implements a basic visualization of sections.

        Returns
        -------
        plotly.graph_objects.Figure
            3D visualization of the neuron morphology
        """
        if hasattr(self, '_filename'):
            # Use the SWC-specific visualization if available
            return visualize_neuron(process_swc_pipeline(self._filename))
        else:
            # Implement basic visualization using the morphology sections
            import plotly.graph_objects as go

            fig = go.Figure()

            # Create traces for each section
            for name, section in self.sections.items():
                # Get 3D points representing the section
                if hasattr(section, 'positions'):
                    # For PointSection
                    x = section.positions[:, 0].magnitude
                    y = section.positions[:, 1].magnitude
                    z = section.positions[:, 2].magnitude

                    # Line representation
                    fig.add_trace(
                        go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='lines',
                            name=name,
                            line=dict(width=2)
                        )
                    )

                    # Points representation
                    fig.add_trace(
                        go.Scatter3d(
                            x=x, y=y, z=z,
                            mode='markers',
                            name=f"{name}_points",
                            marker=dict(
                                size=section.diam.flatten().magnitude / 2,
                                opacity=0.5
                            )
                        )
                    )
                else:
                    # For CylinderSection - simplified representation
                    # Create line from start to end
                    pass  # Implement based on CylinderSection specifics

            # Update layout
            fig.update_layout(
                title="Neuron Morphology",
                scene=dict(
                    xaxis_title="X (μm)",
                    yaxis_title="Y (μm)",
                    zaxis_title="Z (μm)",
                    aspectmode='data'
                )
            )

            return fig
