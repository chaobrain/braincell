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
import re
from pathlib import Path

import brainunit as u
import numpy as np
from scipy.interpolate import interp1d

from ._morphology import Morphology
from ._utils import get_type_name


class Token:
    def __init__(self, typ, value, line_no):
        self.type = typ
        self.value = value
        self.line_no = line_no

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


def tokenize_asc_line(line, line_no):
    """
    Tokenize a single line from a Neurolucida ASC file.

    This function splits a line into tokens that represent different elements,
    including parentheses, commas, numbers, quoted strings, reserved keywords,
    and labels. It skips whitespace and comments, and supports various Neurolucida
    keywords and constructs.
    """
    tokens = []
    i = 0
    length = len(line)
    while i < length:
        c = line[i]
        # Skip whitespace
        if c.isspace():
            i += 1
            continue
        # Skip comments
        if c == ';':
            break
        # Single-character symbols
        if c == '(':
            tokens.append(Token('leftpar', '(', line_no))
            i += 1
            continue
        if c == ')':
            tokens.append(Token('rightpar', ')', line_no))
            i += 1
            continue
        if c == ',':
            tokens.append(Token('comma', ',', line_no))
            i += 1
            continue
        if c == '|':
            tokens.append(Token('bar', '|', line_no))
            i += 1
            continue
        if c == '<':
            tokens.append(Token('leftsp', '<', line_no))
            i += 1
            continue
        if c == '>':
            tokens.append(Token('rightsp', '>', line_no))
            i += 1
            continue
        # String with double quotes
        if c == '"':
            j = i + 1
            while j < length and line[j] != '"':
                j += 1
            if j < length:
                tokens.append(Token('string', line[i + 1:j], line_no))
                i = j + 1
                continue
            else:
                tokens.append(Token('err_', line[i:], line_no))
                break
        # Keywords: set, Set, SET
        if line[i:].startswith('set ') or line[i:].startswith('Set ') or line[i:].startswith('SET '):
            tokens.append(Token('set', 'set', line_no))
            i += line[i:].find(' ') + 1
            continue
        # Keyword: RGB
        if line[i:].startswith('RGB '):
            tokens.append(Token('rgb', 'RGB', line_no))
            i += 4
            continue
        # Numbers
        m = re.match(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', line[i:])
        if m:
            val = m.group(0)
            tokens.append(Token('number', float(val), line_no))
            i += len(val)
            continue
        # Label (identifiers)
        m = re.match(r'[A-Za-z_][A-Za-z0-9_]*', line[i:])
        if m:
            tokens.append(Token('label_', m.group(0), line_no))
            i += len(m.group(0))
            continue
        # Unrecognized character
        tokens.append(Token('err_', c, line_no))
        i += 1
    return tokens


class TokenStream:
    """
    A simple stream wrapper around a list of tokens, providing
    lookahead and cursor movement, used by the ASC parser.
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.idx = 0

    @property
    def current(self):
        """Return the current token, or EOF if at the end."""
        return self.tokens[self.idx] if self.idx < len(self.tokens) else Token("eof", None, -1)

    @property
    def look_ahead(self):
        """Return the next token, or EOF if at the end."""
        return self.tokens[self.idx + 1] if self.idx + 1 < len(self.tokens) else Token("eof", None, -1)

    @property
    def look_ahead2(self):
        """Return the token after next, or EOF if at the end."""
        return self.tokens[self.idx + 2] if self.idx + 2 < len(self.tokens) else Token("eof", None, -1)

    def advance(self):
        """Advance the token pointer by one."""
        self.idx += 1

    def expect(self, typ):
        """
        Ensure the current token is of the expected type, advance, and return it.
        Raise ValueError otherwise.
        """
        if self.current.type != typ:
            raise ValueError(f"Expected {typ}, got {self.current}")
        tok = self.current
        self.advance()
        return tok

    def is_eof(self):
        """Check if the stream has reached the end."""
        return self.current.type == 'eof'


class Point:
    """
    Represents a 3D point in a neuron morphology, possibly with additional misc info.
    """

    def __init__(self, x, y, z, d, misc, idx):
        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.idx = idx  # Global index for the point
        self.misc = misc  # Additional labels or attributes

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self):
        return f"Point({self.idx}: {self.x}, {self.y}, {self.z}, {self.d}, {self.misc})"


class Section:
    """
    Represents a section (branch or soma) in the reconstructed morphology.
    Contains a list of Point objects, type info, parent id, and a contour stack.
    """

    def __init__(self, sec_id, sec_type, parent_id=None, parent_x=-1):
        self.sec_id = sec_id  # Unique identifier for the section
        self.sec_type = sec_type  # soma = 1, dend = 2, axon = 3 ...
        self.points = []  # List of Point objects
        self.parent_id = parent_id  # The parent section, or None if root
        self.contour_stack = []  # Used for complex objects (e.g. multi-contour soma)
        self.parent_x = parent_x

    @property
    def z_range(self):
        """Return (min_z, max_z) of all points in section, or (0,0) if empty."""
        zs = [p.z for p in self.points]
        return (min(zs), max(zs)) if zs else (0, 0)

    @property
    def center(self):
        mean, _, _, _ = contourcenter(self.points)
        return (mean[0], mean[1])

    @property
    def bbox_xy(self):
        """Return bounding box (min_x, max_x, min_y, max_y) in XY-plane."""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), max(xs), min(ys), max(ys))

    @property
    def stk_bbox_xy(self):
        """
        Return the bounding box (xmin, xmax, ymin, ymax) in the XY-plane,
        including both main points and all points in the contour_stack.
        """
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        for contour in self.contour_stack:
            xs += [p.x for p in contour]
            ys += [p.y for p in contour]
        return (min(xs), max(xs), min(ys), max(ys))

    @property
    def stk_center(self):
        """
        Return geometric 'stack center' for the contour stack (multi-layer soma contour).
        Returns: (x, y, z) tuple
        """
        centers = []
        # 1. Center of the main section point set (supports 3D)
        mean, _, _, _ = contourcenter(self.points)
        centers.append(tuple(mean))  # (x, y, z)

        # 2. Each contour in contour_stack
        for contour in getattr(self, 'contour_stack', []):
            if contour:
                # Support contour being either a list of points or Section
                pts = getattr(contour, "points", contour)
                mean, _, _, _ = contourcenter(pts)
                centers.append(tuple(mean))  # (x, y, z)

        # 3. Cumulative principal axis length and interpolation
        lengths = [0.0]
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            dz = centers[i][2] - centers[i - 1][2]
            l = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            lengths.append(lengths[-1] + l)
        half_len = lengths[-1] / 2
        if half_len == 0:
            return centers[0]

        for i in range(1, len(lengths)):
            if lengths[i] > half_len:
                th = (half_len - lengths[i - 1]) / (lengths[i] - lengths[i - 1])
                c0 = centers[i - 1]
                c1 = centers[i]
                center = (
                    th * c1[0] + (1 - th) * c0[0],
                    th * c1[1] + (1 - th) * c0[1],
                    th * c1[2] + (1 - th) * c0[2],
                )
                return center
        return centers[-1]

    def __repr__(self):
        return f"(Section(id={self.sec_id},type={self.sec_type},points={len(self.points)},pid={self.parent_id},px={self.parent_x})"


class Parser:
    """
    The main parser for Neurolucida ASC files.
    Handles parsing of all major ASC constructs, including contours (soma),
    trees (dendrites/axons), properties, spines, and marker lists.
    """

    def __init__(self, tokens):
        self.ts = TokenStream(tokens)  # Token stream

        self.all_points = []  # Flat list of all parsed Point objects
        self.sections = []  # List of parsed Section objects
        self.cur_section_type = 0  # Type flag for new sections

        self.spines = []  # List of spine dicts
        self.blocks = []  # Optional: stores parse block names for debugging

    def parse(self):
        """
        Main entry: parse all tokens into ASC blocks.
        Calls parse_object() for each recognized left parenthesis.
        """
        while not self.ts.is_eof():
            if self.ts.current.type == 'leftpar':
                self.parse_object()
            else:
                self.ts.advance()
        return self.blocks

    def parse_object(self):
        """
        Parse a high-level ASC block (contour, tree, set, marker, etc.)
        Dispatch based on lookahead tokens.
        """
        cur = self.ts.current
        la = self.ts.look_ahead
        la2 = self.ts.look_ahead2

        # print(f"parse_object @{self.ts.idx}: cur={cur} la={la} la2={la2}")

        # 1. Contour blocks: e.g. ("Cell Body" ...)
        if self.ts.look_ahead.type == 'string':
            self.blocks.append(f"contour: {la.value}")
            return self.parse_contour()

        # 2. Tree (axon/dendrite) or text block
        if self.ts.look_ahead.type == 'leftpar':
            self.blocks.append(f"tree or text")
            return self.parse_tree_or_text()

        # 3. Property blocks (Color, CellBody, Class, etc.)
        if self.ts.look_ahead.type == 'label_' and self.ts.look_ahead2.type in ('number', 'string'):
            self.blocks.append(f"property: {la.value}")
            self.skip_unknown_block()
            return

        # 4. Set (metadata block)
        if self.ts.look_ahead.type == 'set':
            self.blocks.append(f"set: {la.value}")
            self.skip_unknown_block()
            return

        # 5. Spine (special labeled block)
        if self.ts.look_ahead.type == 'label_' and self.ts.look_ahead.value == 'Spine':
            self.blocks.append(f"spine: {la.value}")
            self.skip_unknown_block()
            return

        # 6. Marker block
        if self.ts.look_ahead.type == 'label_' and self.ts.look_ahead.value == 'Marker':
            self.blocks.append(f"Maker: {la.value}")
            self.skip_unknown_block()
            return

        else:
            # Unknown or unrecognized block
            return self.skip_unknown_block()

    def skip_unknown_block(self):
        """
        Skip over a complete parenthesized block (from current '(' to matching ')').
        Used for skipping unrecognized or currently unhandled blocks.
        """
        depth = 1
        self.ts.advance()
        while not self.ts.is_eof() and depth > 0:
            if self.ts.current.type == 'leftpar':
                depth += 1
            elif self.ts.current.type == 'rightpar':
                depth -= 1
            self.ts.advance()

    def parse_contour(self):
        """
        Parse a (contour ...) block.
        For soma: at least 3 points are required.
        Handles possible attribute blocks, then points.
        """
        self.ts.expect('leftpar')
        string = self.ts.expect('string').value

        begin = len(self.all_points)
        attributes = []

        while True:
            if self.ts.current.type == 'rightpar':
                break
            # Property/attribute blocks: e.g. (Color Red)
            if self.ts.current.type == 'leftpar':
                if self.ts.look_ahead.type == 'label_':
                    # print('self.parse_property()')
                    attributes.append(self.parse_property())
                elif self.ts.look_ahead.type == 'set':
                    # print('self.parse_set()')
                    self.skip_unknown_block()
                elif self.ts.look_ahead.type == 'number':
                    # print('self.point()')
                    self.parse_point()
                else:
                    print(f"Warning: Unexpected contour block {self.ts.current}, {self.ts.look_ahead}")
                    self.skip_unknown_block()
            else:
                print(f"Skipping unexpected token {self.ts.current}")
                self.ts.advance()

        # Set type for soma (cell body)
        if string in ["Cell Body", "CellBody", "Soma"]:
            self.cur_section_type = 1

        end = len(self.all_points)
        if end - begin > 2:
            section = Section(sec_id=len(self.sections), sec_type=self.cur_section_type)
            section.points = self.all_points[begin:end]
            self.sections.append(section)
        else:
            raise ValueError("soma must at least has 3 points！")

        self.ts.expect('rightpar')
        return {
            "type": "contour",
            "name": string,
            "attributes": attributes,
        }

    def parse_property(self):
        """
        Parse a (label_ ...) property/attribute block.
        E.g. (Color Red), (Axon), (Class 1 'Spine').
        Returns the label and values (can be numbers, strings, RGB, etc).
        """
        self.ts.expect('leftpar')
        label = self.ts.expect('label_').value

        # Set section type by property label
        if label == "Axon":
            self.cur_section_type = 2
        elif label == "Dendrite":
            self.cur_section_type = 3
        elif label == "Apical":
            self.cur_section_type = 4
        elif label == "CellBody" or label == "Cell Body" or label == "Soma":
            self.cur_section_type = 1
        values = []

        # Parse all values until ')'
        while not self.ts.is_eof() and self.ts.current.type != 'rightpar':
            typ = self.ts.current.type
            if typ in ('number', 'string', 'label_', 'rgb'):
                values.append(self.ts.current.value)
                self.ts.advance()

                # Parse RGB triple, e.g. (RGB (1,0,0))
                if typ.lower() == "rgb":
                    if self.ts.current.type == 'leftpar':
                        self.ts.advance()  # consume '('
                    rgb_values = []
                    while len(rgb_values) < 3:
                        if self.ts.current.type == 'comma':
                            self.ts.advance()
                            continue
                        if self.ts.current.type == 'number':
                            rgb_values.append(self.ts.current.value)
                            self.ts.advance()
                        else:
                            break
                    if len(rgb_values) != 3:
                        raise ValueError("RGB needs 3 numbers")
                    values.extend(rgb_values)
                    if self.ts.current.type == 'rightpar':
                        self.ts.advance()
                    else:
                        raise ValueError(f"Parse error: RGB property not closed with ')', got {self.ts.current}")
            elif typ == 'comma':
                self.ts.advance()  # skip comma
            else:
                self.ts.advance()  # skip others

        self.ts.expect('rightpar')
        # print(f"Parsed property: {label} {values}")

    def skip_commas(self):
        """Advance past any comma tokens."""
        while self.ts.current.type == 'comma':
            self.ts.advance()

    def parse_point(self, store=True):
        """
        Parse a (x y z radius [misc...]) point.
        Stores to all_points if store=True.
        """
        self.ts.expect('leftpar')
        x = self.ts.expect('number').value
        self.skip_commas()
        y = self.ts.expect('number').value
        self.skip_commas()
        z = 0.0
        d = 0.0
        if self.ts.current.type == 'number':
            z = self.ts.current.value
            self.ts.advance()
            self.skip_commas()
        if self.ts.current.type == 'number':
            d = self.ts.current.value
            self.ts.advance()
            self.skip_commas()
        misc = []
        while self.ts.current.type in ('label_', 'string'):
            misc.append(self.ts.current.value)
            self.ts.advance()
        self.ts.expect('rightpar')

        idx = len(self.all_points)
        pt = Point(x, y, z, d, misc, idx)

        if store == True:
            self.all_points.append(pt)
        return pt

    def parse_tree_or_text(self, parent_id=None):
        """
        Try to parse either a text block or a tree block.
        If text parsing fails, roll back and parse as a tree.
        """
        old_idx = self.ts.idx
        try:
            res = self.parse_text()
            # print(f"text detected at token {old_idx}")
            return {'type': 'text', 'content': res}
        except Exception as e:
            self.ts.idx = old_idx
            # print(f"text parse failed at token {old_idx}, fallback to tree: {e}")
            return self.parse_tree(parent_id=parent_id)

    def parse_text(self):
        """
        Parse a (point string) text block.
        Returns the string content.
        """
        self.ts.expect('leftpar')
        while self.ts.current.type == 'leftpar' and self.ts.look_ahead.type in ('label_', 'set'):
            if self.ts.look_ahead.type == 'set':
                self.parse_set()
            else:
                self.parse_property()
        pt = self.parse_point(store=False)
        if self.ts.current.type != 'string':
            raise ValueError("Text expects a string after point")
        content = self.ts.current.value
        self.ts.advance()
        self.ts.expect('rightpar')
        return content

    def parse_tree(self, parent_id=None):
        """
        Parse a tree (branch structure).
        Calls parse_properties() for initial attributes,
        then recursively parses the main branch and sub-branches.
        """
        self.ts.expect('leftpar')
        self.parse_properties()
        self.parse_branch(parent_id=parent_id)
        self.ts.expect('rightpar')

    def parse_properties(self):
        """
        Parse all property blocks at the current nesting level.
        Returns a list of parsed property values (if needed).
        """
        properties = []
        while self.ts.current.type == 'leftpar' and self.ts.look_ahead.type in ('label_', 'set'):
            if self.ts.look_ahead.type == 'set':
                self.parse_set()
            else:
                prop = self.parse_property()
                properties.append(prop)
        return properties

    def parse_branch(self, parent_id=None):
        """
        Parse a single branch (list of points), and handle branch ends and splits.
        Registers a new Section for each branch.
        """
        begin = len(self.all_points)
        self.parse_treepoints()
        end = len(self.all_points)
        section = Section(sec_id=len(self.sections), sec_type=self.cur_section_type, parent_id=parent_id, parent_x=1)
        section.points = self.all_points[begin:end]
        self.sections.append(section)
        this_sec_id = section.sec_id
        self.parse_branchend(parent_id=this_sec_id)

    def parse_treepoints(self):
        """
        Parse one or more point blocks (and any attached marker/spine blocks).
        """
        self.parse_treepoint()
        while self.ts.current.type == 'leftpar' and self.ts.look_ahead.type == 'number':
            self.parse_treepoint()

    def parse_treepoint(self):
        """
        Parse a single tree point, and any attached markers, properties, or spines.
        """
        if self.ts.look_ahead.type == 'label_':
            # Marker or property block
            if self.ts.look_ahead2.type == 'leftpar':
                self.parse_marker()
            else:
                self.parse_property()
        else:
            pt = self.parse_point()
            # Parse any attached spine block(s)
            while self.ts.current.type == 'leftsp':  # '<'
                print('spine')
                self.skip_unknown_block()  # TODO: implement actual spine parsing

    def parse_branchend(self, parent_id):
        """
        Handle the end of a branch, including marker lists and branch splits.
        """
        self.skip_commas()
        while self.ts.current.type == 'leftpar' and self.ts.look_ahead.type == 'label_':
            if self.ts.look_ahead2.type == 'leftpar':
                self.parse_marker()
            else:
                self.parse_property()
            self.skip_commas()
        if self.ts.current.type == 'leftpar' or self.ts.current.type == 'label_':
            self.parse_node(parent_id)

    def parse_node(self, parent_id):
        """
        Parse a branching node: either another branch or a label.
        """
        if self.ts.current.type == 'leftpar':
            self.ts.advance()
            self.parse_split(parent_id)
            self.ts.expect('rightpar')
        elif self.ts.current.type == 'label_':
            self.ts.advance()
        else:
            raise ValueError('node: Unexpected token')

    def parse_split(self, parent_id):
        """
        Parse a split node (bifurcation), indicated by bar tokens ('|').
        Each child branch is parsed recursively.
        """
        self.parse_branch(parent_id)
        while self.ts.current.type == 'bar':
            self.ts.advance()
            self.parse_branch(parent_id)

    def parse_spine_proc(self, base_point):
        """
        Parse a <spine> block attached to a point (not fully implemented here).
        """
        self.ts.expect('leftsp')
        spine_properties = []
        while self.ts.current.type == 'leftpar' and self.ts.look_ahead.type == 'label_':
            prop = self.parse_property()
            spine_properties.append(prop)
        pt = self.parse_spine_point()
        self.spines.append({'base_point': base_point, 'spine_tip': pt, 'properties': spine_properties})
        self.ts.expect('rightsp')

    def parse_spine_point(self):
        """
        Parse a spine point (same format as a normal point).
        """
        self.ts.expect('leftpar')
        x = self.ts.expect('number').value
        self.skip_commas()
        y = self.ts.expect('number').value
        self.skip_commas()
        z = 0.0
        d = 0.0
        if self.ts.current.type == 'number':
            z = self.ts.current.value
            self.ts.advance()
            self.skip_commas()
        if self.ts.current.type == 'number':
            d = self.ts.current.value
            self.ts.advance()
            self.skip_commas()
        misc = []
        while self.ts.current.type in ('label_', 'string'):
            misc.append(self.ts.current.value)
            self.ts.advance()
        self.ts.expect('rightpar')
        return {'x': x, 'y': y, 'z': z, 'd': d, 'misc': misc}

    def parse_marker(self):
        """Skip over a marker block (not yet implemented)."""
        self.skip_unknown_block()

    def parse_set(self):
        """Skip over (set ...) blocks, which contain metadata or display info."""
        self.skip_unknown_block()


##########################################
def contourcenter(points, num=101):
    """
    Uniformly resample a 3D contour by arclength and return centroid.

    Args:
        points: list of Point (with .x, .y, .z)
        num: number of resample points

    Returns:
        mean: ndarray, shape (3,), centroid (mean_x, mean_y, mean_z)
        x_new, y_new, z_new: ndarray, resampled coordinates (length=num)
    """
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    z = np.array([p.z for p in points])
    seglens = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    perim = np.zeros(len(x))
    perim[1:] = np.cumsum(seglens)
    d_uniform = np.linspace(0, perim[-1], num)
    x_new = np.interp(d_uniform, perim, x)
    y_new = np.interp(d_uniform, perim, y)
    z_new = np.interp(d_uniform, perim, z)
    mean = np.array([x_new.mean(), y_new.mean(), z_new.mean()])
    return mean, x_new, y_new, z_new


def sort_sections(sections):
    """
    Reorder the list of Section objects so that all soma sections (sec_type == 1) come first,
    while preserving the original order within each group.
    """
    soma_sec = [sec for sec in sections if sec.sec_type == 1]
    other_sec = [sec for sec in sections if sec.sec_type != 1]
    return soma_sec + other_sec


def xy_intersect(bb1, bb2):
    """
    Check whether the bounding boxes bb1 and bb2 overlap in the XY plane.

    Each bounding box is (xmin, xmax, ymin, ymax).
    Returns True if there is any overlap, False otherwise.
    """
    xmin1, xmax1, ymin1, ymax1 = bb1
    xmin2, xmax2, ymin2, ymax2 = bb2
    return not (xmax1 < xmin2 or xmax2 < xmin1 or
                ymax1 < ymin2 or ymax2 < ymin1)


def merge_soma_sections(sections):
    """
    Merge overlapping soma (sec_type==1) sections, keeping only the main stack and 
    adding any overlapping somas to its contour_stack. Non-soma sections are preserved.
    """
    somas = [sec for sec in sections if sec.sec_type == 1]
    used = set()
    new_somas = []
    N = len(somas)
    i = 0
    while i < N:
        if i in used:
            i += 1
            continue
        master = somas[i]
        bb1 = master.bbox_xy
        for j in range(i + 1, N):
            if j in used:
                continue
            cand = somas[j]
            bb2 = cand.bbox_xy
            if xy_intersect(bb1, bb2):
                master.contour_stack.append(cand)
                used.add(j)
        new_somas.append(master)
        i += 1
    # Reassemble sections: only keep the main soma stacks and all other sections
    new_sec = [sec for sec in sections if sec.sec_type != 1]
    new_sec[:0] = new_somas  # Insert somas at the beginning
    return new_sec


def reindex_sections(sections):
    """
    After reordering or merging sections, update the sec_id and parent_id of all sections
    to ensure consistency. Returns the updated list.
    """
    old2new = {}
    for new_id, sec in enumerate(sections):
        old2new[sec.sec_id] = new_id
        sec.sec_id = new_id
    for sec in sections:
        if sec.parent_id is not None:
            sec.parent_id = old2new.get(sec.parent_id, None)
    return sections


def remove_duplicate_points(sections):
    """
    Remove duplicate points from each section. A duplicate is defined as a point with
    exactly the same (x, y, z, d) as a previously encountered point in the same section.
    Only the first occurrence is kept. Prints a warning for each removal.
    """
    for section in sections:
        unique_pts = []
        seen = set()
        for pt in section.points:
            key = (pt.x, pt.y, pt.z, pt.d)
            if key not in seen:
                unique_pts.append(pt)
                seen.add(key)
            else:
                print(
                    f"Warning: Section {section.sec_id} has duplicate point ({pt.x}, {pt.y}, {pt.z}, {pt.d}), removed.")
        section.points = unique_pts


def ensure_section_continuity(sections):
    """
    Ensure continuity between each section and its parent:
    If the last point of the parent section does not match the first point of the child section,
    insert a copy of the parent's last point at the beginning of the child section.
    This only applies to non-soma parent sections.
    """
    id2section = {sec.sec_id: sec for sec in sections}
    for sec in sections:
        if sec.parent_id is None:
            continue
        parent = id2section[sec.parent_id]
        if parent.sec_type == 1:  # skip soma
            continue
        if not sec.points or not parent.points:
            continue
        parent_last = parent.points[-1]
        child_first = sec.points[0]
        if parent_last != child_first:
            new_point = Point(
                x=parent_last.x,
                y=parent_last.y,
                z=parent_last.z,
                d=child_first.d,
                misc=child_first.misc,
                idx=None,
            )
            sec.points.insert(0, new_point)
            # print(f"Added continuity point from parent {parent.sec_id} to child {sec.sec_id}")


def validate_soma_stack(main_soma_section, tol=1e-6):
    """
    Check that the main soma section and its contour_stack meet the following criteria:
      1. All points within each layer (section) have identical z value (within tolerance);
      2. The z values across layers are strictly monotonic (increasing or decreasing).
    Raises ValueError if not satisfied.
    """
    stack = [main_soma_section] + list(getattr(main_soma_section, 'contour_stack', []))

    # Check all z values in each section are the same
    for idx, sec in enumerate(stack):
        z_vals = [p.z for p in sec.points]
        if not z_vals:
            raise ValueError(f"[SOMA CHECK] Section {idx} in soma stack is empty")
        z0 = z_vals[0]
        if not all(abs(z - z0) < tol for z in z_vals):
            raise ValueError(f"[SOMA CHECK] Contour {idx} z-values not constant: {z_vals}")

    # Check monotonicity of the stack in z
    z_stack = [sec.points[0].z for sec in stack if sec.points]
    dzs = [z_stack[i + 1] - z_stack[i] for i in range(len(z_stack) - 1)]
    if all(d > tol for d in dzs):
        return True  # strictly increasing
    if all(d < -tol for d in dzs):
        return True  # strictly decreasing
    raise ValueError(f"[SOMA CHECK] Contour stack z-values not monotonic: {z_stack}")


def validate_soma_stack_main(sections):
    """
    Validate all soma stacks in the given sections list.
    """
    for sec in sections:
        if sec.sec_type == 1:
            validate_soma_stack(sec)


def connect_to_soma(sections, buffer=0.5, verbose=True):
    """
    Automatically connect all dangling (parentless, non-soma) sections to the appropriate soma.
    - First, tries to match the section's root point to a "loose" bounding box (bbox) around each soma.
    - If not inside any soma bbox, connects to the center of the nearest soma.
    - Supports both contour_stack and regular point-cloud soma representations.

    Args:
        sections (list): List of Section objects (must include both soma and branches).
        buffer (float): Extra margin for bbox test.
        verbose (bool): If True, print connection info.

    Returns:
        unmatched (list): Dangling sections not inside any soma bbox, but auto-connected to the nearest soma.
    """
    # 1. Extract all soma sections
    soma_secs = [sec for sec in sections if sec.sec_type == 1]

    # 2. Precompute centers and bounding boxes for all somas
    soma_centers = []
    for soma_sec in soma_secs:
        if hasattr(soma_sec, "contour_stack") and soma_sec.contour_stack:
            center = soma_sec.stk_center  # Use contour stack center
            bbox_xy = soma_sec.stk_bbox_xy  # Use stack bbox
        else:
            center = soma_sec.center  # Use point cloud mean
            bbox_xy = soma_sec.bbox_xy
        soma_centers.append(center)
        soma_sec._bbox_xy = bbox_xy  # Optionally cache on object

    # 3. Gather all dangling non-soma sections
    dangling_secs = [sec for sec in sections if sec.parent_id is None and sec.sec_type != 1]
    unmatched = []

    # 4. Try to connect each dangling section to a soma
    for dangling_sec in dangling_secs:
        if not dangling_sec.points:
            continue
        x0, y0 = dangling_sec.points[0].x, dangling_sec.points[0].y
        found = False
        for i, soma_sec in enumerate(soma_secs):
            # Use appropriate bbox for this soma
            if hasattr(soma_sec, "contour_stack") and soma_sec.contour_stack:
                xmin, xmax, ymin, ymax = soma_sec.stk_bbox_xy
            else:
                xmin, xmax, ymin, ymax = soma_sec.bbox_xy
            loose_xmin, loose_xmax = xmin - buffer, xmax + buffer
            loose_ymin, loose_ymax = ymin - buffer, ymax + buffer
            # Check if root point falls inside the (loosened) bbox
            if loose_xmin <= x0 <= loose_xmax and loose_ymin <= y0 <= loose_ymax:
                if verbose:
                    print(f"{dangling_sec.sec_id} falls inside loose bbox of soma {soma_sec.sec_id}")
                dangling_sec.parent_id = soma_sec.sec_id
                dangling_sec.parent_x = 0.5  # Attach to soma center (for NEURON-style models)
                found = True
                break
        if not found:
            # Not inside any soma bbox; connect to nearest soma center
            min_dist = float('inf')
            min_idx = None
            z0 = dangling_sec.points[0].z
            for i, center in enumerate(soma_centers):
                cx, cy, cz = center
                dist = ((x0 - cx) ** 2 + (y0 - cy) ** 2 + (z0 - cz) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            nearest_soma = soma_secs[min_idx]
            if verbose:
                print(
                    f"!!! Section {dangling_sec.sec_id} root ({x0:.2f},{y0:.2f}) not inside any soma bbox; connected to nearest soma ({nearest_soma.sec_id})")
            dangling_sec.parent_id = nearest_soma.sec_id
            dangling_sec.parent_x = 0.5
            unmatched.append(dangling_sec)  # Store for possible manual inspection

    if verbose and unmatched:
        print(
            f"\n{len(unmatched)} dangling branches were not inside any soma bbox, but were auto-connected to the nearest soma.")
    return unmatched  # List of "rescued" branches for optional post-processing


def soma_axis_sampling(
    points,
    n_samples=21,
    arclength_resample=101,
):
    """
    Fit a closed soma contour, perform convex filtering and main axis interpolation,
    and return 21 spatial positions along the main axis and their diameter.

    Args:
        points: List of point objects with .x, .y attributes (2D contour, closed).
        n_samples: Number of axis points to sample (default 21).
        arclength_resample: How many arclength-resampled points for contour smoothing.
    Returns:
        XY_interp: (n_samples, 2) array of axis-sampled positions.
        diam_interp: (n_samples,) array of diameter at each axis sample.
    """
    # Step 1: Arclength uniform resampling and centroid
    mean, x_new, y_new, _ = contourcenter(points, num=arclength_resample)
    mean = mean[:-1]  ## no z value

    # Step 2: PCA for principal/minor axes
    pts = np.stack([x_new, y_new], axis=1)
    pts_centered = pts - mean
    cov = np.cov(pts_centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    minor = eigvecs[:, 0]
    if major[np.argmax(np.abs(major))] < 0:
        major = -major
    major = major / np.linalg.norm(major)
    minor = minor / np.linalg.norm(minor)

    # Step 3: Project all points onto axes
    d = (pts - mean) @ major
    rad = (pts - mean) @ minor

    # Step 4: Split contour into two convex sides and filter
    def rotate(arr, k):
        return np.concatenate([arr[k:], arr[:k]])

    def keep_strictly_monotonic(x, y, increasing=True, tol=1e-8):
        keep_idx = [0]
        for i in range(1, len(x)):
            if increasing:
                if x[i] > x[keep_idx[-1]] + tol:
                    keep_idx.append(i)
            else:
                if x[i] < x[keep_idx[-1]] - tol:
                    keep_idx.append(i)
        return x[keep_idx], y[keep_idx]

    imax = np.argmax(d)
    imin = np.argmin(d)
    d_rot = rotate(d, imax)
    rad_rot = rotate(rad, imax)
    pts_rot = rotate(pts, imax)
    imin_new = np.where(d_rot == d[imin])[0][0]

    d_side1 = d_rot[:imin_new][::-1]
    rad_side1 = rad_rot[:imin_new][::-1]
    d_side2 = d_rot[imin_new:]
    rad_side2 = rad_rot[imin_new:]

    inc1 = len(d_side1) > 1 and (d_side1[1] > d_side1[0])
    inc2 = len(d_side2) > 1 and (d_side2[1] > d_side2[0])
    d_side1_new, rad_side1_new = keep_strictly_monotonic(d_side1, rad_side1, increasing=inc1)
    d_side2_new, rad_side2_new = keep_strictly_monotonic(d_side2, rad_side2, increasing=inc2)

    # Step 5: Interpolate main axis (exclude endpoints)
    d_all = np.concatenate([d_side1_new, d_side2_new])
    d_all_sorted = np.sort(d_all)
    d_min = d_all_sorted[1]
    d_max = d_all_sorted[-2]
    d_interp = np.linspace(d_min, d_max, n_samples)
    XY_interp = mean[None, :] + d_interp[:, None] * major[None, :]

    # Step 6: Interpolate radii for both sides, then compute diameter
    f_rad1 = interp1d(d_side1_new, rad_side1_new, kind='linear', bounds_error=False,
                      fill_value=(rad_side1_new[0], rad_side1_new[-1]))
    f_rad2 = interp1d(d_side2_new, rad_side2_new, kind='linear', bounds_error=False,
                      fill_value=(rad_side2_new[0], rad_side2_new[-1]))
    rad1_interp = f_rad1(d_interp)
    rad2_interp = f_rad2(d_interp)
    diam_interp = np.abs(rad1_interp - rad2_interp)
    # Smooth endpoints 
    diam_interp[0] = (diam_interp[0] + diam_interp[1]) / 2
    diam_interp[-1] = (diam_interp[-1] + diam_interp[-2]) / 2

    return XY_interp, diam_interp


def approximate_contour_by_circle(points, num=101):
    """
    points: list of Point(x, y, z)
    num: int, number of points for arclength-uniform resampling

    Returns:
        center: (x, y, z) tuple, centroid of resampled contour
        avg_radius: float, averaged radius (robust hybrid)
    """
    n = len(points)
    if n < 2:
        raise ValueError("At least two points required")
    # Use arclength-resampled centroid (robust against uneven input points)
    mean, x_new, y_new, z_new = contourcenter(points, num=num)
    mean_radius = np.mean(
        np.sqrt((x_new - mean[0]) ** 2 + (y_new - mean[1]) ** 2 + (z_new - mean[2]) ** 2)
    )
    perim = np.sum(np.sqrt(np.diff(x_new) ** 2 + np.diff(y_new) ** 2 + np.diff(z_new) ** 2))
    perim_radius = perim / (2 * np.pi)
    diam = perim_radius + mean_radius
    return mean, diam


def contourstack2centroid(section, num=101):
    """
    Approximate each contour (main + contour_stack) as a circle,
    return all centers (x, y, z) and diameters as lists.
    
    Args:
        section: a Section object with .points (main contour) and .contour_stack (list of contour, each a list of Point)
        num: number of points for arclength-uniform resampling
        verbose: print area info if True

    Returns:
        xs, ys, zs: list of float, all centers for each layer
        diams: list of float, all diameters for each layer
    """
    xs, ys, zs, diams = [], [], [], []

    # 1. 主contour
    mean, diameter = approximate_contour_by_circle(section.points, num=num)
    xs.append(mean[0]);
    ys.append(mean[1]);
    zs.append(mean[2]);
    diams.append(diameter)

    # 2. 每个contour_stack
    for contour in getattr(section, "contour_stack", []):
        if contour:
            mean, diameter = approximate_contour_by_circle(contour, num=num)
            xs.append(mean[0]);
            ys.append(mean[1]);
            zs.append(mean[2]);
            diams.append(diameter)

    return xs, ys, zs, diams


def replace_soma_with_axis_sampling(sections, n_samples=21, **plot_kwargs):
    """
    For each soma section in `sections`, replace its points with axis-based samples:
    - If the soma section has only a single contour (no contour_stack), use `soma_axis_sampling`.
    - If the soma section is multi-layer (contour_stack not empty), use stack-based axis sampling (`contourstack2centroid`).
    - All other sections are left unchanged.

    Args:
        sections: list of Section objects (must have sec_type, points, optionally contour_stack).
        n_samples: for single-contour soma, number of axis-sampled points to generate.
        plot_kwargs: extra switches for soma_axis_sampling.

    Returns:
        sections: updated in place.
    """
    for sec in sections:
        if getattr(sec, 'sec_type', None) == 1:
            # Case 1: Single-layer soma (no contour_stack)
            if not getattr(sec, 'contour_stack', []):
                XY, diam = soma_axis_sampling(
                    sec.points,
                    n_samples=n_samples,
                    **plot_kwargs
                )
                sec.points = [
                    Point(
                        x=XY[i, 0], y=XY[i, 1],
                        z=sec.points[0].z if sec.points else 0.0,
                        d=float(diam[i]), misc=[], idx=i
                    )
                    for i in range(len(diam))
                ]
            # Case 2: Multi-layer soma (contour_stack present)
            else:
                xs, ys, zs, diams = contourstack2centroid(sec, num=n_samples)
                sec.points = [
                    Point(
                        x=xs[i], y=ys[i], z=zs[i],
                        d=float(diams[i]), misc=[], idx=i
                    )
                    for i in range(len(xs))
                ]


def read_asc(file):
    # Parse the tokens from the .asc file
    tokens = []
    with open(file) as f:
        for i, line in enumerate(f):
            tokens.extend(tokenize_asc_line(line, i + 1))
    tokens.append(Token('eof', '', i + 2))  # Add end-of-file token

    # Initialize parser and parse the token list
    parser = Parser(tokens)
    parser.parse()
    # Remove duplicate points from all sections
    remove_duplicate_points(parser.sections)
    # Sort, merge and reindex soma sections (root sections)
    parser.sections = sort_sections(parser.sections)
    parser.sections = merge_soma_sections(parser.sections)
    parser.sections = reindex_sections(parser.sections)
    # Ensure continuity of non-soma sections (branches)
    ensure_section_continuity(parser.sections)
    # Validate that the soma stack is consistent and makes sense
    validate_soma_stack_main(parser.sections)
    # Connect branch to the soma, within a certain buffer distance
    unmatched = connect_to_soma(parser.sections, buffer=0.5, verbose=False)
    if unmatched:
        raise RuntimeError("there are still unmatched sections after attempting to connect to soma")
    # Resample soma points along the main axis (e.g., to 21 samples)
    replace_soma_with_axis_sampling(parser.sections, n_samples=21)

    return parser.sections


def from_asc(filename: str | Path) -> Morphology:
    """
    Parse a Neurolucida ASC file and construct a Morphology object.

    This function reads a `.asc` file describing neuron morphology, parses its structure,
    processes soma and branch sections, and returns a Morphology object suitable for
    further analysis or simulation.

    Args:
        filename (str or Path): Path to the ASC file to be parsed. The file must have a `.asc` extension.

    Returns:
        Morphology: An instance of the Morphology class containing all sections and their connections.

    Processing steps:
        1. Validates the file extension.
        2. Parses the ASC file into a list of Section objects.
        3. Assigns unique names to each section based on type and order.
        4. Extracts positions and diameters for each section and stores them in a dictionary.
        5. Creates a Morphology object and adds all sections.
        6. Establishes parent-child connections between sections.
        7. Returns the fully constructed Morphology object.

    Raises:
        ValueError: If the file does not have a `.asc` extension.
        RuntimeError: If there are unmatched sections after attempting to connect to the soma.
    """

    # Check if the file has the correct extension
    _, postfix = os.path.splitext(filename)
    if postfix != '.asc':
        raise ValueError(f"File {filename} is not an ASC file.")

    # 1. Parse the ASC file into a list of Section objects
    sections = read_asc(filename)  # main returns a list of Section objects

    # 2. Build section_dicts using get_type_name for section names
    section_dicts = {}
    section_id_map = {}  # Map: sec_id -> section_name
    type_counters = {}
    for sec in sections:
        section_type = sec.sec_type
        type_name = get_type_name(section_type)

        # init counter
        if type_name not in type_counters:
            type_counters[type_name] = 0

        # index each type
        type_inner_id = type_counters[type_name]
        section_name = f"{type_name}_{type_inner_id}"
        type_counters[type_name] += 1

        section_id_map[sec.sec_id] = section_name

        # Collect positions as a (N, 3) numpy array: x, y, z  and diam as a  (N,)  numpy array: d 
        positions = np.column_stack([
            [p.x for p in sec.points],
            [p.y for p in sec.points],
            [p.z for p in sec.points],
        ])
        diams = np.array([p.d for p in sec.points])

        section_dicts[section_name] = {
            'positions': positions * u.um,
            'diams': diams * u.um,
            'nseg': 1,  # Default value
        }

    # morphology object
    morphology = Morphology()

    # 3. Add all sections
    morphology.add_multiple_sections(section_dicts)

    # 4. Prepare and add connection info
    connections = []
    for sec in sections:
        if sec.parent_id is not None:
            child_name = section_id_map[sec.sec_id]
            parent_name = section_id_map[sec.parent_id]
            parent_loc = sec.parent_x
            connections.append((child_name, parent_name, parent_loc))
    morphology.connect_sections(connections)
    return morphology
