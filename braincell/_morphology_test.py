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
import os.path

import brainunit as u

import braincell

class TestMorphologyConstruction:
    def test_single(self):
        # Instantiate a Morphology object
        morphology = braincell.Morphology()

        # Create individual sections using the creation methods
        morphology.add_cylinder_section(
            'soma', length=20 * u.um, diam=10 * u.um, nseg=1
        )  # Soma section
        morphology.add_cylinder_section(
            'axon', length=100 * u.um, diam=1 * u.um, nseg=2
        )  # Axon section
        morphology.add_point_section(
            'dendrite',
            positions=[[0, 0, 0], [100, 0, 0], [200, 0, 0]] * u.um,
            diams=[2, 3, 2] * u.um,
            nseg=3
        )  # Dendrite section with explicit points and diameters

        # Connect the sections: axon and dendrite connected to soma
        morphology.connect('axon', 'soma', parent_loc=1.0)      # Axon connects to soma at the end
        morphology.connect('dendrite', 'soma', parent_loc=1.0)  # Dendrite connects to soma at the end

        # Print a summary of the morphology
        print(morphology)

        # Print each section's name and diameters
        for sec in morphology.sections.values():
            print("name:", sec.name, 'diam:', sec.diams)

        # Initialize DHS (Dendritic Hierarchical Scheduling)
        morphology.dhs_init()

    def test_multiple(self):
        # Instantiate a Morphology object
        morphology = braincell.Morphology()

        # Define sections using a property dictionary
        section_dicts = {
            'soma':     {'length': 20 * u.um, 'diam': 10 * u.um, 'nseg': 1},
            'axon':     {'length': 100 * u.um, 'diam': 1 * u.um, 'nseg': 2},
            'dendrite': {
                'positions': [[0, 0, 0], [100, 0, 0], [200, 0, 0]] * u.um,
                'diams': [2, 3, 2] * u.um,
                'nseg': 3
            }
        }
        # Add all sections from the dictionary
        morphology.add_multiple_sections(section_dicts)

        # Define and apply connections between sections
        connection_list = [
            ('axon', 'soma', 1.0),        # Axon connects to soma at the end
            ('dendrite', 'axon', 1.0)     # Dendrite connects to axon at the end
        ]
        morphology.connect_sections(connection_list)

        # Print section information
        for sec in morphology.sections.values():
            print("name:", sec.name, 'nseg:', sec.nseg)

        # Print conductance matrix and area for the whole model
        print(morphology.conductance_matrix)
        print(morphology.area)

        # Initialize DHS
        morphology.dhs_init()

    def test_swc(self):
        # Load morphology from SWC file
        swc_file = os.path.join(os.path.dirname(__file__), "../dev/swc_file/io.swc")
        morphology = braincell.Morphology.from_swc(swc_file)
        print(morphology)
        # Initialize DHS
        morphology.dhs_init()

    def test_asc(self):
        # Load morphology from ASC file
        asc_file = os.path.join(os.path.dirname(__file__), "../dev/asc_file/golgi.asc")
        morphology = braincell.Morphology.from_asc(asc_file)
        print(morphology)
        # Initialize DHS
        morphology.dhs_init()
