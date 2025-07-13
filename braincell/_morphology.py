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

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Sequence, Dict, Hashable, NamedTuple

import brainstate
import brainunit as u
import numpy as np

from ._morphology_branch_tree import BranchingTree
from ._morphology_from_asc import from_asc
from ._morphology_from_swc import from_swc
from ._morphology_utils import (
    calculate_total_resistance_and_area,
    generate_interpolated_nodes,
    compute_connection_seg,
    compute_line_ratios,
    init_coupling_weight_nodes,
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


@dataclass
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
        diams: u.Quantity[u.um],
        nseg: int,
        Ra: u.Quantity[u.ohm * u.cm],
        cm: u.Quantity[u.uF / (u.cm ** 2)],
    ):
        self.name = name
        assert u.fail_for_dimension_mismatch(positions, u.um, 'positions must be in meter')
        assert u.fail_for_dimension_mismatch(diams, u.um, 'diameters must be in meter')
        assert u.fail_for_dimension_mismatch(Ra, u.ohm * u.cm, 'diameters must be in u.ohm * u.cm')
        assert u.fail_for_dimension_mismatch(cm, u.uF / (u.cm ** 2), 'diameters must be in u.uF / (u.cm ** 2)')
        self._nseg = nseg
        self._Ra = Ra
        self._cm = cm
        self.positions = positions
        self.diams = diams
        self.parent: Dict = None
        self.children = set()
        self.segments = []

        self._compute_area_and_resistance()

    @property
    def L(self):
        """
        Returns the total length of the section in micrometers.

        The length is computed as the sum of Euclidean distances between consecutive
        3D points in the `positions` array, which represent the coordinates of the section.

        Returns
        -------
        u.Quantity
            The total length of the section in micrometers (u.um).
        """
        pos = self.positions / u.um
        return np.sum(np.linalg.norm(pos[1:] - pos[:-1], axis=1)) * u.um

    @property
    def nseg(self) -> int:
        """
        Get the number of segments the section is divided into.

        Returns
        -------
        int
            The number of segments (`nseg`) for this section.
        """
        return self._nseg

    @nseg.setter
    def nseg(self, value):
        raise ValueError(
            'nseg cannot be set directly. Use replace() method to change nseg value.'
        )

    @property
    def Ra(self):
        """
        Get the axial resistivity of the section.

        Returns
        -------
        u.Quantity
            The axial resistivity (`Ra`) of the section in ohm·cm.
        """
        return self._Ra

    @Ra.setter
    def Ra(self, value):
        raise ValueError(
            'Ra cannot be set directly. Use replace() method to change Ra value.'
        )

    @property
    def cm(self):
        """
        Get the specific membrane capacitance of the section.

        Returns
        -------
        u.Quantity
            The specific membrane capacitance (`cm`) in µF/cm².
        """
        return self._cm

    @cm.setter
    def cm(self, value):
        raise ValueError(
            'cm cannot be set directly. Use replace() method to change cm value.'
        )

    def replace(
        self,
        nseg: int = None,
        Ra: u.Quantity = None,
        cm: u.Quantity = None,
    ) -> 'Section':
        """
        Create a new Section instance with updated properties.

        This method returns a new Section object that is a copy of the current section,
        but with optionally replaced values for the number of segments (`nseg`), axial resistivity (`Ra`),
        and specific membrane capacitance (`cm`). All other properties (name, positions, diameters)
        are preserved from the original section.

        Parameters
        ----------
        nseg : int, optional
            The number of segments for the new section. If not provided, uses the current value.
        Ra : u.Quantity, optional
            The axial resistivity for the new section (must have units of ohm·cm). If not provided, uses the current value.
        cm : u.Quantity, optional
            The specific membrane capacitance for the new section (must have units of uF/cm²). If not provided, uses the current value.

        Returns
        -------
        Section
            A new Section instance with the specified updated properties.

        Raises
        ------
        AssertionError
            If `nseg` is provided and is not an integer.
        ValueError
            If `Ra` or `cm` are provided and have incompatible units.

        Notes
        -----
        This method does not modify the original section; it returns a new instance.
        """
        name = self.name
        positions = self.positions
        diams = self.diams
        if nseg is not None:
            assert isinstance(nseg, (int, np.integer)), f'nseg must be an integer, but got {nseg}'
        else:
            nseg = self.nseg
        if Ra is not None:
            u.fail_for_dimension_mismatch(Ra, u.ohm * u.cm, 'Ra must be in u.ohm * u.cm')
        else:
            Ra = self.Ra
        if cm is not None:
            u.fail_for_dimension_mismatch(cm, u.uF / (u.cm ** 2), 'cm must be in u.uF / (u.cm ** 2)')
        else:
            cm = self.cm
        return Section(name=name, positions=positions, diams=diams, nseg=nseg, Ra=Ra, cm=cm)

    def __repr__(self):
        n_points = getattr(self.positions, "shape", [len(self.positions)])[0]
        if self.parent and "name" in self.parent and "loc" in self.parent:
            parent_str = f"{self.parent['name']!r}"
            parent_loc = f"{self.parent['loc']}"
        else:
            parent_str = None
            parent_loc = None
        return (
            f"Section<name={self.name!r}, nseg={self.nseg}, points={n_points}, "
            f"Ra={self.Ra}, cm={self.cm}, parent={parent_str}, parent_loc={parent_loc}>"
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
        node_pre = np.hstack(
            [self.positions / u.um, (self.diams / u.um).reshape((-1, 1))]
        )
        Ra = self.Ra / (u.ohm * u.cm)
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
            R_left, area_left = calculate_total_resistance_and_area(selected_left, Ra)
            R_right, area_right = calculate_total_resistance_and_area(selected_right, Ra)

            segment = Segment(
                section_name=self.name,
                index=int(i / 2),
                cm=self.cm,
                area=(area_left + area_right) * u.um ** 2,
                R_left=R_left * (u.ohm * u.cm / u.um),
                R_right=R_right * (u.ohm * u.cm / u.um),
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
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [u.get_magnitude(length), 0.0, 0.0]
            ]
        ) * u.get_unit(length)
        diam = np.array(
            [
                [u.get_magnitude(diam)],
                [u.get_magnitude(diam)]
            ]
        ) * u.get_unit(diam)
        super().__init__(
            name=name,
            positions=positions,
            diams=diam,
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
        positions: u.Quantity[u.um],
        diams: u.Quantity[u.um],
        nseg: int = 1,
        Ra: u.Quantity = 100 * u.ohm * u.cm,
        cm: u.Quantity = 1.0 * u.uF / u.cm ** 2,
    ):
        """
        Initialize the Section.

        Parameters:
            name (str): Section name identifier.
            points (list or np.ndarray, optional): Array of shape (N, 3) with [x, y, z].
            diams (list or np.ndarray, optional): Array of shape (N, 1) 
            nseg (int): Number of segments to divide the section into.
            Ra (float): Axial resistivity in ohm·cm.
            cm (float): Membrane capacitance in µF/cm².
        """

        assert positions.shape[0] >= 2, "at least have 2 points"
        assert positions.shape[1] == 3, "points must be shape (N, 3): [x, y, z]"
        assert np.all(np.array(u.get_magnitude(diams)) > 0), "All diameters must be positive."

        super().__init__(
            name=name,
            positions=positions,
            diams=diams,
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
        self.sections = {}  # Dictionary to store section objects by name
        self._conductance_matrix = None
        self._area = None
        self._cm = None
        self._nseg = None
        self._parent_id = None
        self._parent_x = None
        self._seg_ri = None

    @property
    def segments(self):
        """
        Returns a flat list of all segments across all sections.

        Returns
        -------
        list
            A list of Segment objects representing all segments in the morphology.
        """
        return [seg for section in self.sections.values() for seg in section.segments]

    def to_branching_tree(self):
        """
        Convert the morphology to a BranchingTree representation for dendritic hierarchy analysis.

        This method transforms the morphological structure into a BranchingTree object
        that can be used for the Dendritic Hierarchy and Structure (DHS) algorithm.
        The transformation involves computing the electrical properties of all segments
        and their connectivity relationships.

        The method calculates:
        - Segment resistances (left and right) for all segments
        - Number of segments per section
        - Connection information between segments
        - Parent-child relationships between segments
        - Specific membrane capacitance for each segment
        - Surface area for each segment

        Returns
        -------
        BranchingTree
            A BranchingTree object representing the morphological structure
            with all necessary electrical properties for DHS algorithm processing.

        Notes
        -----
        The BranchingTree representation is optimized for efficient computation
        of electrical properties in compartmental modeling and enables analysis
        of the dendritic hierarchy. This representation is particularly useful
        for studying signal propagation and integration in neuronal structures.

        See Also
        --------
        BranchingTree : The class that handles dendritic hierarchy processing
        """

        # section resistances
        seg_ri = u.math.array([(seg.R_left, seg.R_right) for seg in self.segments]) / u.ohm

        # number of segments per section
        nsegs = np.asarray([sec.nseg for sec in self.sections.values()])

        # connection sections, parent section id and location
        connection_sec_list = self._connection_sec_list()
        _, parent_id, parent_x = compute_connection_seg(nsegs, connection_sec_list)

        # cm for each segment
        cm_segmid = u.math.array([seg.cm for seg in self.segments])

        # area for each segment
        area_segmid = u.math.array([seg.area for seg in self.segments])

        return BranchingTree(seg_ri, parent_id, parent_x, cm_segmid, area_segmid)

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
        length : Quantity
            Length of the cylindrical section, u.cm
        diam : Quantity
            Diameter of the cylindrical section, u.cm
        nseg : int, optional
            Number of segments to divide the section into, default=1
        Ra : Quantity
            Axial resistivity of the section, default=100 * u.ohm * u.cm
        cm : Quantity
            Specific membrane capacitance, default=1.0 * u.uF / u.cm ** 2

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
            raise ValueError(f"Section with name '{name}' already exists, please choose a different name.")
        self.sections[name] = section

    def add_point_section(
        self,
        name: SectionName,
        positions,
        diams,
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
        positions : u.Quantity[u.cm]
            Array of shape (N, 4) with each point as [x, y, z, diameter]
        diams: u.Quantity[u.cm]
            Array of shape (N, 1) with diameters at each point.
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
        section = PointSection(name, positions=positions, diams=diams, nseg=nseg, Ra=Ra, cm=cm)
        if name in self.sections:
            raise ValueError(f"Section with name '{name}' already exists, please choose a different name.")
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
        >>> import brainunit as u
        >>> morph.add_multiple_sections({
        ...     'soma': {'length': 20.0 * u.um, 'diam': 20.0 * u.um},
        ...     'axon': {'length': 800.0 * u.um, 'diam': 1.0 * u.um, 'nseg': 5},
        ...     'dendrite': {'points': [[0,0,0,2], [10,10,0,1.5], [20,20,0,1]] * u.um}
        ... })

        Notes
        -----
        This is a convenience method that calls either `add_point_section` or `add_cylinder_section`
        for each section specification based on the parameters provided.
        """
        assert isinstance(section_dicts, dict), 'section_dicts must be a dictionary'

        for section_name, section_data in section_dicts.items():
            assert isinstance(section_data, dict), 'section_data must be a dictionary.'
            if 'positions' in section_data:
                self.add_point_section(name=section_name, **section_data)
            elif 'length' in section_data and 'diam' in section_data:
                self.add_cylinder_section(name=section_name, **section_data)
            else:
                raise ValueError('section_data must contain either positions or length and diam.')

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

    def conductance_matrix(self):
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
        return init_coupling_weight_nodes(g_left, g_right, connection_seg_list)

    @classmethod
    def from_swc(cls, filename: str | Path):
        """
        Class method to create a Morphology object from an SWC file (factory method).
        
        Parameters
        ----------
        filename : str, Path
            Path to the SWC file
            
        Returns
        -------
        Morphology
            A Morphology object created from the SWC file
        """
        return from_swc(filename)

    @classmethod
    def from_asc(cls, filename: str | Path):
        """
        Class method to create a Morphology object from an ASC file (factory method).

        Parameters
        ----------
        filename : str, Path
            Path to the ASC file

        Returns
        -------
        Morphology
            A Morphology object created from the ASC file
        """
        return from_asc(filename)

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
        # Implement basic visualization using the morphology sections
        import plotly.graph_objects as go

        fig = go.Figure()

        # Create traces for each section
        for name, section in self.sections.items():
            # Get 3D points representing the section
            if isinstance(section, PointSection):
                # For PointSection
                x = section.positions[:, 0] / u.um
                y = section.positions[:, 1] / u.um
                z = section.positions[:, 2] / u.um

                # Line representation
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=name, line=dict(width=2)))

                # Points representation
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z, mode='markers', name=f"{name}_points",
                        marker=dict(size=section.diams.flatten() / u.um / 2, opacity=0.5)
                    )
                )
            else:
                # TODO:
                # For CylinderSection - simplified representation
                # Create line from start to end
                raise NotImplementedError('CylinderSection visualization not implemented yet.')

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

    def set_passive_params(
        self,
        nseg_length: u.Quantity = 40 * u.um,
        Ra_soma: u.Quantity = 122. * u.ohm * u.cm,
        cm_soma: u.Quantity = 1 * u.uF / u.cm ** 2,
        Ra_dend: u.Quantity = 122. * u.ohm * u.cm,
        cm_dend: u.Quantity = 2.5 * u.uF / u.cm ** 2,
        Ra_axon: u.Quantity = 122. * u.ohm * u.cm,
        cm_axon: u.Quantity = 1 * u.uF / u.cm ** 2,
    ):
        """
        Configure passive electrical properties and segment counts for all sections in the morphology.

        This method iterates over all sections in the morphology and sets the number of segments (`nseg`),
        axial resistivity (`Ra`), and specific membrane capacitance (`cm`) for each section based on its type.
        The section type is determined by the presence of the substrings 'soma', 'dend', or 'axon' in the section name.

        Parameters
        ----------
        nseg_length : u.Quantity, optional
            Target length for each segment (default: 40 µm). The number of segments for each section is computed as
            `1 + 2 * floor(section_length / nseg_length)`.
        Ra_soma : u.Quantity, optional
            Axial resistivity for soma sections (default: 122 Ω·cm).
        cm_soma : u.Quantity, optional
            Specific membrane capacitance for soma sections (default: 1 µF/cm²).
        Ra_dend : u.Quantity, optional
            Axial resistivity for dendrite sections (default: 122 Ω·cm).
        cm_dend : u.Quantity, optional
            Specific membrane capacitance for dendrite sections (default: 2.5 µF/cm²).
        Ra_axon : u.Quantity, optional
            Axial resistivity for axon sections (default: 122 Ω·cm).
        cm_axon : u.Quantity, optional
            Specific membrane capacitance for axon sections (default: 1 µF/cm²).

        Processing Steps
        ----------------
        1. For each section in the morphology:
            a. Compute the number of segments (`nseg`) based on the section's length and `nseg_length`.
            b. Set `Ra` and `cm` according to the section type:
                - If 'soma' in section name: use `Ra_soma` and `cm_soma`.
                - If 'dend' in section name: use `Ra_dend` and `cm_dend`.
                - If 'axon' in section name: use `Ra_axon` and `cm_axon`.

        Notes
        -----
        - Section type is determined by substring matching in the section name.
        - This method is useful for initializing passive properties before simulation or analysis.
        - All units must be compatible with the expected quantities.
        """
        for k, v in tuple(self.sections.items()):
            # Update nseg based on section length
            nseg = int(1 + 2 * np.floor(v.L / nseg_length))

            # Set Ra and cm by section type
            if 'soma' in k:
                self.sections[k] = v.replace(nseg=nseg, Ra=Ra_soma, cm=cm_soma)
            elif 'dend' in k:
                self.sections[k] = v.replace(nseg=nseg, Ra=Ra_dend, cm=cm_dend)
            elif 'axon' in k:
                self.sections[k] = v.replace(nseg=nseg, Ra=Ra_axon, cm=cm_axon)
