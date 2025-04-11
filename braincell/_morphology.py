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

from typing import Union

from ._utils import *

__all__ = [
    'Section',
    'Morphology',
]


class Section:
    r'''
    Neuron Section Object:

    This class defines a neuron section that can be either:
    1. A simple cylinder (using length and diameter), which creates a frustum between two points:
       (0, 0, 0, diam) and (L, 0, 0, diam)
    2. A more complex morphology defined by N 4D points (x, y, z, diameter), forming a sequence of connected frustums.

    Each section is divided into `nseg` segments, and each segment has computed properties:
        - surface area
        - left axial resistance (to previous segment)
        - right axial resistance (to next segment)
    '''

    def __init__(
        self,
        name: str,
        length=None,
        diam=None,
        points=None,
        nseg=1,
        Ra=100,
        cm=1.0,
    ):
        """
        Initialize the Section.

        Parameters:
            name (str): Section name identifier.
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
        self.parent = None
        self.children = []
        self.segments = []

        # Case 1: user provides custom 3D points
        if points is not None:
            points = np.array(points)
            assert points.shape[1] == 4, "points must be shape (N, 4): [x, y, z, diameter]"
            assert np.all(points[:, 3] > 0), "All diameters must be positive."
            self.positions = points[:, :3]
            self.diam = points[:, -1].reshape(-1, 1)

        # Case 2: user provides a simple cylinder
        elif length is not None and diam is not None:
            assert length > 0, "Length must be positive."
            assert diam > 0, "Diameter must be positive."
            points = np.array([
                [0.0, 0.0, 0.0, diam],
                [length, 0.0, 0.0, diam]
            ])
            self.positions = points[:, :3]
            self.diam = points[:, -1].reshape(-1, 1)

        else:
            raise ValueError("You must provide either `points` or both `length` and `diam`.")

        self.compute_area_and_resistance()

    def compute_area_and_resistance(self):
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


class Morphology:
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

    def create_section(
        self,
        name: str,
        points=None,
        length=None,
        diam=None,
        nseg=1,
    ):
        """
        Create a new section in the model.

        Parameters:
            name (str): The name of the section.
            points (list or np.ndarray, optional): A list of points defining the section in [x, y, z, diameter] format.
            length (float, optional): Length of the section (used for simple cylinder).
            diam (float, optional): Diameter of the section (used for simple cylinder).
            nseg (int, optional): The number of segments to discretize the section into.
        """
        section = Section(name, points=points, length=length, diam=diam, nseg=nseg)
        self.sections[name] = section
        self.segments.extend(section.segments)

    def get_section(self, name: str):
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
        child_name: str,
        parent_name: str,
        parent_loc: Union[float, int] = 1.0
    ):
        """
        Connect one section to another, establishing a parent-child relationship.

        Parameters:
            child_name (str): The name of the child section to be connected.
            parent_name (str): The name of the parent section to which the child connects.
            parent_loc (float, optional): The location on the parent section to connect to (0 to 1). Default is 1.0.
        """
        assert 0.0 <= parent_loc <= 1.0, "parent_loc must be between 0.0 and 1.0"

        child = self.get_section(child_name)
        parent = self.get_section(parent_name)

        if child is None or parent is None:
            raise ValueError("Both child and parent sections must exist.")

        # If the child already has a parent, remove the old connection and notify the user
        if child.parent is not None:
            print(f"Warning: Section '{child_name}' already has a parent: {child.parent['parent_name']}.")
            # Remove the child from the old parent's children list
            old_parent_name = child.parent['parent_name']
            old_parent = self.get_section(old_parent_name)
            if old_parent is not None:
                old_parent.children.remove(child_name)

        # Set the new parent for the child
        child.parent = {
            "parent_name": parent_name,
            "parent_loc": parent_loc
        }

        # Add the child to the new parent's children list
        parent.children.append(child_name)

    def create_sections_from_dict(self, section_dicts):
        """
        Create multiple sections from a list of dictionaries.

        Parameters:
            section_dicts (list of dicts): List of dictionaries containing section properties.

        Example format:
            section_dicts = [
                {'name': 'soma', 'length': 20, 'diam': 10, 'nseg': 1},
                {'name': 'axon', 'length': 100, 'diam': 1, 'nseg': 10},
                {'name': 'dendrite', 'points': [[0, 0, 0, 2], [100, 0, 0, 2], [200, 0, 0, 2]], 'nseg': 5}
            ]
        """
        for section_data in section_dicts:
            name = section_data['name']
            points = section_data.get('points', None)
            length = section_data.get('length', None)
            diam = section_data.get('diam', None)
            nseg = section_data.get('nseg', 1)
            self.create_section(name, points=points, length=length, diam=diam, nseg=nseg)

    def connect_sections_from_list(self, connection_sec_list):
        """
        Connect multiple sections based on a list of tuples containing parent-child relationships.

        Parameters:
            connection_sec_list (list of tuples): Each tuple is (child_idx, parent_idx, parent_loc), specifying the connections.

        Example format:
            connection_sec_list = [
                (1, 0, 1.0)  # Connect axon (index 1) to soma (index 0) at location 1.0
            ]
        """
        for child_idx, parent_idx, parent_loc in connection_sec_list:
            child_name = list(self.sections.keys())[child_idx]
            parent_name = list(self.sections.keys())[parent_idx]
            self.connect(child_name, parent_name, parent_loc)

    def connection_sec_list(self):
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
                parent_name = child_section.parent["parent_name"]
                parent_loc = child_section.parent["parent_loc"]

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
            g_left.append(1 / (seg['R_left'] * u.ohm * u.cm / u.um))
            g_right.append(1 / (seg['R_right'] * u.ohm * u.cm / u.um))

        for sec in self.sections.values():
            nseg_list.append(sec.nseg)

        connection_sec_list = self.connection_sec_list()
        connection_seg_list = compute_connection_seg(nseg_list, connection_sec_list)

        self.conductance_matrix = init_coupling_weight_nodes(g_left, g_right, connection_seg_list)

    def list_sections(self):
        """List all sections in the model with their properties (e.g., number of segments)."""
        for name, section in self.sections.items():
            print(f"Section: {name}, nseg: {section.nseg}, Points: {section.positions.shape[0]}")

    @classmethod
    def from_swc(self, *args, **kwargs) -> 'Morphology':
        raise NotImplementedError

    @classmethod
    def from_asc(self, *args, **kwargs) -> 'Morphology':
        raise NotImplementedError
