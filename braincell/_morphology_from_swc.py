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


import os

import numpy as np

from ._morphology_utils import get_type_name


class Import3dSection:
    """
    Represents an unbranched section (segment) of a neuron morphology.
    
    A section is a sequence of points with the same type (e.g., soma, axon, dendrite)
    that form an unbranched structure. Sections connect to other sections to
    form the complete neuron morphology.
    
    Attributes:
        id (int): Index of the first point in this section
        first (int): Flag indicating if first point is included (0) or skipped (1)
        parentsec (Import3dSection): Parent section this section connects to
        parentx (float): Connection position on parent (0.0-1.0)
        pid (int): Parent ID for this section
        type (int): Cell compartment type (1=soma, 2=axon, 3=dendrite, etc.)
        x (numpy.ndarray): X-coordinates of points in this section
        y (numpy.ndarray): Y-coordinates of points in this section
        z (numpy.ndarray): Z-coordinates of points in this section
        d (numpy.ndarray): Diameters of points in this section
    """

    def __init__(self, first_index, length):
        """
        Initialize a section with first point index and number of points.
        
        Parameters:
            first_index (int): Index of the first point in this section
            length (int): Number of points in this section
        """
        self.id = first_index  # Index of first point
        self.first = 0  # 0=include first point, 1=skip first point
        self.parentsec = None  # Parent section (connected to)
        self.parentx = 1.0  # Position on parent (0-1)
        self.pid = -1  # Parent ID
        self.type = -1  # Cell type (1=soma, 2=axon, 3=dendrite, etc.)

        # Data arrays for the section points
        self.x = np.array([])  # X coordinates
        self.y = np.array([])  # Y coordinates
        self.z = np.array([])  # Z coordinates
        self.d = np.array([])  # Diameters

    def append(self, flag, start_index, count, x_data, y_data, z_data, d_data):
        """
        Append points to the section from source data arrays.
        
        Parameters:
            flag (int): 0 for first point, 1 for remaining points
            start_index (int): Index of the first point to append
            count (int): Number of points to append
            x_data (numpy.ndarray): Source array of X coordinates
            y_data (numpy.ndarray): Source array of Y coordinates
            z_data (numpy.ndarray): Source array of Z coordinates
            d_data (numpy.ndarray): Source array of diameters
        """
        if count <= 0:
            return

        indices = np.arange(start_index, start_index + count)

        if flag == 0:  # First point - create new arrays
            self.x = np.array([x_data[start_index]])
            self.y = np.array([y_data[start_index]])
            self.z = np.array([z_data[start_index]])
            self.d = np.array([d_data[start_index]])
        else:  # Remaining points - append to existing arrays
            self.x = np.append(self.x, x_data[indices])
            self.y = np.append(self.y, y_data[indices])
            self.z = np.append(self.z, z_data[indices])
            self.d = np.append(self.d, d_data[indices])

    def __repr__(self):
        """
        Provide a string representation of the section.
        
        Returns:
            str: String describing the section
        """
        points = len(self.x)
        return f"Section(id={self.id}, type={self.type}, points={points})"


class Import3dSWCRead:
    """
    Class to read SWC files and create neuron morphology.
    
    SWC is a standard format for representing neuron morphology where each line has:
    1. ID (integer)
    2. Type (integer: 1=soma, 2=axon, 3=dendrite, etc.)
    3. X coordinate (float)
    4. Y coordinate (float)
    5. Z coordinate (float)
    6. Radius (float)
    7. Parent ID (integer, -1 for root)
    
    This class reads SWC files, validates the tree structure, identifies sections
    (unbranched sequences), and creates section objects.
    
    Attributes:
        id (numpy.ndarray): Point IDs from SWC file
        type (numpy.ndarray): Point types (1=soma, 2=axon, 3=dendrite, etc.)
        x, y, z (numpy.ndarray): Spatial coordinates
        d (numpy.ndarray): Diameters (note: stored as 2*radius)
        pid (numpy.ndarray): Parent IDs
        iline (numpy.ndarray): Line numbers for error messages
        header (list): Comment lines from the SWC file
        lines (list): Data lines from the SWC file
        sections (list): List of Import3dSection objects
        point2sec (numpy.ndarray): Maps point indices to section indices
        sec2point (numpy.ndarray): Lists the last point of each section
        id2index_ (numpy.ndarray): Maps original IDs to array indices
        connect2prox (numpy.ndarray): Flags for connection to proximal end
        nchild_soma (numpy.ndarray): Number of children for soma points
    """

    def __init__(self):
        """
        Initialize the SWC reader with empty arrays and default values.
        """
        self.quiet = False  # Suppress warnings if True
        self.filetype = "SWC"  # File type identifier
        self.header = []  # Comment lines
        self.lines = []  # Data lines
        self.vectors = {}  # For backward compatibility
        self.sections = []  # Section objects
        self.err = False  # Error flag
        self.idoffset = 0  # Offset for ID normalization
        self.soma3geom = False  # Flag for special 3-point soma

        # Data arrays
        self.id = np.array([], dtype=int)  # Point IDs
        self.type = np.array([], dtype=int)  # Point types
        self.x = np.array([])  # X coordinates
        self.y = np.array([])  # Y coordinates
        self.z = np.array([])  # Z coordinates
        self.d = np.array([])  # Diameters
        self.pid = np.array([], dtype=int)  # Parent IDs
        self.iline = np.array([], dtype=int)  # Line numbers

        # Maps and indices
        self.id2index_ = None  # Maps original IDs to array indices
        self.connect2prox = None  # Flags for connection to proximal end
        self.point2sec = None  # Maps point indices to section indices
        self.nchild_soma = None  # Number of children for soma points
        self.sec2point = None  # Lists the last point of each section

    def input(self, filename):
        """
        Main entry point - read and process an SWC file.
        
        This method follows the full processing pipeline:
        1. Read the SWC file
        2. Validate the tree structure
        3. Identify sections
        4. Create section objects
        
        Parameters:
            filename (str): Path to the SWC file
        
        Returns:
            bool: False if error occurred, True otherwise
        """
        self.err = False
        self.rdfile(filename)  # Read the file
        self.check_pid()  # Validate tree structure and create id2index_
        self.sectionify()  # Create point2sec index map
        self.mksections()  # Create Import3dSection list

        return not self.err

    def rdfile(self, filename):
        """
        Read the SWC file, line by line.
        
        Parameters:
            filename (str): Path to the SWC file or SWC content string
                           (if starting with '#')
        """
        # Special case for direct string input
        if isinstance(filename, str) and filename.startswith('#'):
            lines = filename.strip().split('\n')
            for i, line in enumerate(lines, 1):
                self.parse(i, line)
            return

        # Regular file reading
        if not os.path.exists(filename):
            self.err = True
            print(f"Could not open {filename}")
            return

        # Read file line by line
        with open(filename, 'r') as file:
            for i, line in enumerate(file, 1):
                self.parse(i, line)

    def parse(self, line_num, line_str):
        """
        Parse a single line from the SWC file.
        
        Parameters:
            line_num (int): Line number for error reporting
            line_str (str): Content of the line to parse
        """
        line_str = line_str.strip()

        # Skip empty lines
        if not line_str:
            return

        # Save comments
        if line_str.startswith('#'):
            self.header.append(line_str)
            return

        # Parse data line
        parts = line_str.split()
        if len(parts) == 7:
            try:
                values = [float(p) for p in parts]

                # Initialize arrays on first valid data line
                if len(self.id) == 0:
                    self.id = np.array([int(values[0])], dtype=int)  # ID
                    self.type = np.array([int(values[1])], dtype=int)  # Type
                    self.x = np.array([values[2]])  # X coordinate
                    self.y = np.array([values[3]])  # Y coordinate
                    self.z = np.array([values[4]])  # Z coordinate
                    self.d = np.array([values[5] * 2])  # Radius to diameter
                    self.pid = np.array([int(values[6])], dtype=int)  # Parent ID
                    self.iline = np.array([line_num], dtype=int)  # Line number
                else:
                    # Append to existing arrays
                    self.id = np.append(self.id, int(values[0]))
                    self.type = np.append(self.type, int(values[1]))
                    self.x = np.append(self.x, values[2])
                    self.y = np.append(self.y, values[3])
                    self.z = np.append(self.z, values[4])
                    self.d = np.append(self.d, values[5] * 2)  # Radius to diameter
                    self.pid = np.append(self.pid, int(values[6]))
                    self.iline = np.append(self.iline, line_num)

                self.lines.append(line_str)
            except ValueError:
                self.err = True
                print(f"Error line {line_num}: could not parse: {line_str}")
        else:
            self.err = True
            print(f"Error line {line_num}: could not parse: {line_str}")

    def id2index(self, id_val):
        """
        Convert raw ID to index in the arrays.
        
        Parameters:
            id_val (int): Raw ID from SWC file
            
        Returns:
            int: Index in the data arrays
        """
        return self.id2index_[id_val]

    def pix2ix(self, index):
        """
        Find parent index for a given point index.
        
        Parameters:
            index (int): Point index
            
        Returns:
            int: Parent index, or -1 if parent ID is negative
        """
        pid_val = self.pid[index]
        if pid_val < 0:
            return -1
        return self.id2index_[pid_val]

    def check_pid(self):
        """
        Validate parent-child relationships and create ID to index mapping.
        
        This method:
        1. Checks if IDs are sorted and sorts if needed
        2. Verifies tree topology (pid[i] < id[i])
        3. Checks for multiple trees (multiple root points)
        4. Checks for duplicate IDs
        5. Creates id2index_ mapping from IDs to array indices
        """
        if len(self.id) == 0:
            return

        # Check if IDs are sorted and sort if needed
        needsort = False
        for i in range(1, len(self.id)):
            if self.id[i] <= self.id[i - 1]:
                needsort = True
                break

        if needsort:
            # Sort all arrays by ID
            sort_indices = np.argsort(self.id)
            self.id = self.id[sort_indices]
            self.pid = self.pid[sort_indices]
            self.x = self.x[sort_indices]
            self.y = self.y[sort_indices]
            self.z = self.z[sort_indices]
            self.d = self.d[sort_indices]
            self.type = self.type[sort_indices]
            self.iline = self.iline[sort_indices]

            # Recreate lines array in sorted order
            sorted_lines = []
            for i in sort_indices:
                sorted_lines.append(self.lines[i])
            self.lines = sorted_lines

        # Check tree topology condition: pid[i] < id[i]
        for i in range(len(self.id)):
            if self.pid[i] >= self.id[i]:
                self.err = True
                print(f"Error: index {i} pid={self.pid[i]} is not less than id={self.id[i]}")

        # Check for multiple trees (pid < 0)
        roots = np.where(self.pid < 0)[0]
        if len(roots) > 1:
            self.err = True
            if not self.quiet:
                print(f"Warning: more than one tree:")
                for i in roots:
                    print(f"  Root at line {self.iline[i]}")

        # Check for duplicate IDs
        for i in range(1, len(self.id)):
            if self.id[i] == self.id[i - 1]:
                self.err = True
                print(f"Error: duplicate id:")
                print(f"  {self.iline[i - 1]}: {self.lines[i - 1]}")
                print(f"  {self.iline[i]}: {self.lines[i]}")

        # Create id2index_ map (from ID to array index)
        max_id = int(np.max(self.id))
        self.id2index_ = np.full(max_id + 1, -1, dtype=int)
        for i in range(len(self.id)):
            self.id2index_[self.id[i]] = i

    def neuromorph_3point_soma(self, nchild):
        """
        Special handling for neuromorpho.org 3-point soma representation.
        
        In some SWC files from neuromorpho.org, the soma is represented as 
        3 points with specific properties that indicate it should be treated
        as a sphere.
        
        Parameters:
            nchild (numpy.ndarray): Number of children for each point
            
        Returns:
            bool: True if special 3-point soma was detected
        """
        self.soma3geom = False

        # Check if we have a 3-point soma with specific properties
        if len(self.id) >= 3 and self.pix2ix(1) == 0 and self.pix2ix(2) == 0:
            if nchild[1] == 0 and nchild[2] == 0:
                if self.d[1] == self.d[0] and self.d[2] == self.d[0]:
                    # Check if distance from center to other points equals diameter
                    length = 0
                    for i in range(1, 3):
                        length += np.sqrt((self.x[i] - self.x[0]) ** 2 +
                                          (self.y[i] - self.y[0]) ** 2 +
                                          (self.z[i] - self.z[0]) ** 2)

                    if abs(length / self.d[0] - 1) < 0.01:
                        self.soma3geom = True
                        self.pid[2] = self.id[1]  # Prevent treating as two soma sections

        return self.soma3geom

    def mark_branch(self):
        """
        Mark branch points based on number of children.
        
        This method identifies branch points by counting child nodes,
        handles special connection cases, and forces section breaks
        at type changes.
        
        Returns:
            numpy.ndarray: Number of children for each point
                         (non-contiguous children add 1.01, contiguous add 1)
        """
        # nchild stores number of child nodes with pid equal to i
        nchild = np.zeros(len(self.id))

        # Warn if first two points have different types
        if len(self.type) > 1 and self.type[0] != self.type[1]:
            self.err = True
            if not self.quiet:
                print(f"\nNotice:")
                print("The first two points have different types but a single point NEURON section is not allowed.")
                print(
                    f"Interpreting the point as center of sphere of radius {self.d[0] / 2} at ({self.x[0]}, {self.y[0]}, {self.z[0]})")

        # Create connect2prox to indicate parent point is not
        # distal end but proximal end of parent section
        self.connect2prox = np.zeros(len(self.id), dtype=int)

        for i in range(len(self.id)):
            p = self.pix2ix(i)  # Parent index
            if p >= 0:
                nchild[p] += 1  # Increment child count

                # If non-contiguous (not adjacent points)
                if p != i - 1:
                    nchild[p] += 0.01  # Add extra to indicate non-contiguous

                    # Special case for branch connecting to proximal end of parent
                    if p > 1:
                        if self.type[p] != 1 and self.type[self.pix2ix(p)] == 1:
                            # Dendrite connected to initial point of another dendrite
                            # that's connected to the soma by a wire
                            self.connect2prox[i] = 1  # Connect to proximal end
                            nchild[p] = 1  # p not treated as a 1pt section
                    elif p == 0:  # parent is root point
                        if self.type[p] != 1:  # and parent is not a soma point
                            self.connect2prox[i] = 1  # Connect to proximal end
                            nchild[p] = 1

                # Force section break on type change (e.g., soma to dendrite)
                if self.type[p] != self.type[i]:
                    nchild[p] += 0.01  # Add extra to force section break

        return nchild

    def sectionify(self):
        """
        Create point-to-section mapping and find section boundaries.
        
        This method:
        1. Calls mark_branch() to identify branch points
        2. Counts soma points and tracks soma children
        3. Checks for special 3-point soma representation
        4. Adjusts branching for contiguous soma points
        5. Finds section boundaries
        6. Creates point2sec mapping from points to sections
        """
        if len(self.id) < 1:
            return

        # Mark branch points
        nchild = self.mark_branch()

        # Count soma points and track soma children
        self.nchild_soma = np.zeros(len(self.id))
        nsoma_pts = 0

        if self.type[0] == 1:  # First point is soma
            nsoma_pts += 1

        for i in range(1, len(self.id)):
            if self.type[i] == 1:  # This point is soma
                nsoma_pts += 1
                pix = self.pix2ix(i)  # Parent index
                if pix >= 0 and self.type[pix] == 1:  # Parent is also soma
                    self.nchild_soma[pix] += 1  # Increment soma child count

        # Special neuromorpho.org policy for 3-point soma
        if nsoma_pts == 3:
            self.neuromorph_3point_soma(nchild)

        # Adjust nchild for contiguity of soma points
        for i in range(len(self.id) - 1):
            # Adjacent parent,child soma points - parent not a branch
            # unless there is more than one soma child for that parent
            if (self.type[i] == 1 and self.type[i + 1] == 1 and
                self.pix2ix(i + 1) == i):
                if i != 0 and self.nchild_soma[i] > 1:
                    pass  # More than one soma child so section branch
                else:
                    nchild[i] = 1  # Not a section end

        # Find section boundaries (points where nchild != 1)
        self.sec2point = np.where(nchild != 1)[0]

        # Create point2sec mapping
        self.point2sec = np.zeros(len(self.id), dtype=int)
        self.point2sec[0] = 0  # First point is in section 0

        si = 0  # Section index
        for i in range(1, len(self.id)):
            if i > self.sec2point[si]:
                si += 1
            self.point2sec[i] = si

    def mksection(self, isec, first, i):
        """
        Create a section object.
        
        This method creates an Import3dSection object for a range of points,
        handling special cases for root and connecting sections.
        
        Parameters:
            isec (int): Section index
            first (int): Index of first point in section
            i (int): Index of one-past-last point in section
        """
        if isec == 0:  # Root section
            if self.soma3geom:  # Treat as single point sphere
                i = 1
            # Create section and add points
            sec = Import3dSection(first, i - first)
            sec.append(1, first, i - first, self.x, self.y, self.z, self.d)

        else:  # Not root section
            # Create section with space for parent point
            sec = Import3dSection(first, i - first + 1)

            # Find parent section
            parent_idx = self.pix2ix(first)
            parent_sec_idx = self.point2sec[parent_idx]
            sec.parentsec = self.sections[parent_sec_idx]
            psec = sec.parentsec

            # Determine connection point and properties
            den_con_soma = (psec.type == 1) and (self.type[first] != 1)  # Dendrite to soma
            con_soma = (psec.type == 1)  # Connection to soma
            handled = False

            if psec == self.sections[0]:  # Connect to root
                handled = True
                if den_con_soma and len(psec.d) == 1:  # Parent is single point soma
                    sec.parentx = 0.5  # Connect to middle (for single point soma)
                    if i - first > 1:  # Connect by wire if multiple points
                        sec.first = 1
                elif self.pix2ix(first) == psec.id:  # Connect to first point of root
                    sec.parentx = 0.0  # Connect to beginning
                    if (self.type[first] != 1 and
                        self.nchild_soma[self.pix2ix(first)] > 1):
                        sec.first = 1
                else:
                    handled = False

            if not handled:
                if con_soma:  # Connection to soma
                    offset = -2
                    if psec.id == 0:
                        offset = -1

                    if self.pix2ix(first) < psec.id + len(psec.d) + offset:
                        # Not last point of soma, so must be interior
                        sec.parentx = 0.5  # Connect to middle
                        if den_con_soma and i - first > 1:
                            sec.first = 1
                    elif (i - first > 1 and
                          self.nchild_soma[self.pix2ix(first)] > 1):
                        if self.type[first] != 1:  # Not soma
                            sec.first = 1

            # Append points: first parent point, then section points
            sec.append(0, self.pix2ix(first), 1, self.x, self.y, self.z, self.d)
            sec.append(1, first, i - first, self.x, self.y, self.z, self.d)

        # Set section type
        sec.type = self.type[first]
        self.sections.append(sec)

        # Special diameter handling for dendrite-soma connection
        if hasattr(sec, 'parentsec') and sec.parentsec is not None:
            if sec.parentsec.type == 1 and sec.type != 1:
                sec.d[0] = sec.d[1]  # Use dendrite diameter, not soma

        # Handle connect2prox case (connection to proximal end)
        if first < len(self.connect2prox) and self.connect2prox[first]:
            sec.pid = sec.parentsec.id
            sec.parentx = 0  # Connect to beginning (proximal)

    def mksections(self):
        """
        Create all section objects.
        
        This method iterates through all points and calls mksection
        for each new section boundary.
        """
        self.sections = []
        isec = 0  # Section index
        first = 0  # First point index

        for i in range(len(self.id)):
            if self.point2sec[i] > isec:
                # Point belongs to a new section
                self.mksection(isec, first, i)
                isec += 1
                first = i

        # Create last section
        self.mksection(isec, first, len(self.id))


def process_swc_pipeline(swc_file):
    """
    Process SWC file to extract neuron morphology data.
    
    Args:
        swc_file (str): Path to the SWC file
        
    Returns:
        dict: Visualization data containing:
            - coords: numpy array of [x, y, z, diameter] for each point
            - types: numpy array of types for each point
            - edges: list of (parent_idx, child_idx) tuples representing connections
    """
    # Initialize reader
    reader = Import3dSWCRead()

    # Read and parse SWC file
    reader.rdfile(swc_file)

    # Validate tree structure
    reader.check_pid()

    # Identify sections
    reader.sectionify()

    # Create section objects
    reader.mksections()

    # Extract point data
    coords = []
    types = []
    edges = []

    # Get coordinates, diameters, and types for each point
    for i in range(len(reader.id)):
        coords.append([reader.x[i], reader.y[i], reader.z[i], reader.d[i]])
        types.append(reader.type[i])

    # Create edges representing connectivity
    for i in range(len(reader.id)):
        parent = reader.pid[i]
        if parent != -1:  # Skip the root node which has no parent
            # Convert 1-based indices to 0-based
            parent_idx = reader.id2index_[parent]
            child_idx = i
            edges.append((parent_idx, child_idx))

    # Convert to numpy arrays
    coords = np.array(coords)
    types = np.array(types)

    # Create dictionary with visualization data
    viz_data = {
        'coords': coords,
        'types': types,
        'edges': edges
    }

    return viz_data


def visualize_neuron(viz_data):
    """
    Visualize neuron structure using Plotly.
    
    Args:
        viz_data (dict): Dictionary containing visualization data
                        (coords, types, edges)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    import plotly.graph_objects as go

    # Extract data from dictionary
    coords = viz_data['coords']
    types = viz_data['types']
    edges = viz_data['edges']

    # Color mapping function
    def get_color(t):
        if t == 1:
            return 'red'  # Soma
        elif t == 2:
            return 'blue'  # Axon
        elif t in [3, 4, 5]:
            return 'green'  # Dendrite
        else:
            return 'gray'  # Other

    # Type name mapping
    type_names = {
        1: "Soma",
        2: "Axon",
        3: "Dendrite",
        4: "Apical dendrite"
    }

    # Create edge traces
    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        x0, y0, z0, _ = coords[i]
        x1, y1, z1, _ = coords[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none',
        name='Connections'  # Adding a descriptive name for the legend
    )

    # Create node traces by type
    node_traces = []
    unique_types = sorted(set(types))
    for t in unique_types:
        mask = types == t
        color = get_color(t)
        type_name = type_names.get(t, f"Type {t}")
        trace = go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode='markers',
            marker=dict(size=4, color=color),
            name=type_name,
            hovertemplate=
            'x: %{x}<br>' +
            'y: %{y}<br>' +
            'z: %{z}<br>' +
            'Type: ' + type_name,
        )
        node_traces.append(trace)

    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)

    # Update layout
    fig.update_layout(
        width=1000,
        height=900,
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(
            title="Neuron Components",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def from_swc(filename):
    # Create SWC reader
    reader = Import3dSWCRead()
    if not reader.input(filename):
        raise ValueError(f"Failed to read SWC file: {filename}")

    # 1. Extract point data from SWC sections and prepare section_dicts
    section_dicts = {}
    for swc_section in reader.sections:
        section_type = swc_section.type
        section_name = f"{get_type_name(section_type)}_{swc_section.id}"

        # Extract point data
        positions = np.column_stack([
            swc_section.x, swc_section.y, swc_section.z
        ])
        diams = swc_section.d

        section_dicts[section_name] = {
            'positions':  positions,
            'diams': diams,
            'nseg': 1  # Default to 1, might need adjustment based on points or length
        }
    return reader.sections, section_dicts
