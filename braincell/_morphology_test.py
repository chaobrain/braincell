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

import braincell
import brainunit as u


class TestMorphologyConstruction:
    def test1(self):
        # Instantiate the Morphology object
        morphology = braincell.Morphology()

        # Create individual sections using `create_section`
        morphology.add_cylinder_section('soma', length=20 * u.um, diam=10 * u.um, nseg=1)  # Soma section
        morphology.add_cylinder_section('axon', length=100 * u.um, diam=1 * u.um, nseg=2)  # Axon section
        morphology.add_point_section('dendrite', [[0, 0, 0, 2], [100, 0, 0, 3], [200, 0, 0, 2]] * u.um, nseg=3)  # Dendrite

        # Connect the sections (e.g., dendrite and axon connected to soma)
        morphology.connect('axon', 'soma', parent_loc=1.0)  # Axon connected at the end of soma
        morphology.connect('dendrite', 'soma', parent_loc=1)  # Dendrite connected at the start of soma

        # List all sections
        morphology.list_sections()

        for sec in morphology.sections.values():
            print("name:", sec.name, 'diam:', sec.diam)

        # List all segments
        print(morphology.segments)

        # Construct conductance matrix for the model
        print(morphology.conductance_matrix)

        morphology.list_sections()

        print(morphology)

    def test2(self):
        # Instantiate the Morphology object
        morphology = braincell.Morphology()

        # Create sections from a dictionary of properties
        section_dicts = {
            'soma': {'length': 20 * u.um, 'diam': 10 * u.um, 'nseg': 1},
            'axon': {'length': 100 * u.um, 'diam': 1 * u.um, 'nseg': 2},
            'dendrite': {'points': [[0, 0, 0, 2], [100, 0, 0, 3], [200, 0, 0, 2]] * u.um, 'nseg': 3}
        }
        morphology.add_multiple_sections(section_dicts)

        # Connect sections from list
        connection_list = [
            ('axon', 'soma', 1.0),
            ('dendrite', 'axon', 1.0)
        ]
        morphology.connect_sections(connection_list)

        # List all sections
        morphology.list_sections()

        for sec in morphology.sections.values():
            print("name:", sec.name, 'nseg:', sec.nseg)

        # List all segments
        print(morphology.segments)

        # Construct conductance matrix for the model
        print(morphology.conductance_matrix)
        print(morphology.area)

