# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Unit tests for :mod:`braincell._base_channel`.

The ion-channel family classes are defined here; ``braincell`` re-exports
them, and that public path is the one users import.
"""

import unittest


class BaseChannelExportTest(unittest.TestCase):
    def test_synapse_legacy_schema_requires_migration(self):
        from braincell import Synapse
        from braincell.mech import ParameterSpec

        class Legacy(Synapse):
            parameters = {"gain": ParameterSpec(1.0)}

        with self.assertRaisesRegex(TypeError, "explicit __init__"):
            Legacy(1)

    def test_synapse_signatures_forward_inheritance_and_required_fields(self):
        import brainunit as u
        import numpy as np
        from braincell import Synapse
        from braincell.synapse import ExpSyn

        class Extended(ExpSyn):
            def __init__(self, size, gain=1.0, **kwargs):
                super().__init__(size, **kwargs)
                self._init_parameters(gain=gain)

        self.assertEqual(set(Extended.parameter_info()), {"tau", "e", "gain"})
        self.assertEqual(float(Extended(1).gain), 1.0)

        class Required(Synapse):
            def __init__(self, size, reversal):
                super().__init__(size)
                self._init_parameters(reversal=reversal)

        node = Required(2, -70.0 * u.mV)
        np.testing.assert_allclose(node.reversal.to_decimal(u.mV), [-70.0, -70.0])
        node = ExpSyn(2, tau=lambda shape: np.full(shape, 3.0) * u.ms)
        np.testing.assert_allclose(node.tau.to_decimal(u.ms), [3.0, 3.0])

    def test_parameter_units_check_a_custom_channel_painted_on_a_cell(self):
        import brainunit as u
        import braincell
        from braincell.filter import AllRegion

        def build(with_units):
            class CustomLeak(braincell.Channel):
                root_type = braincell.HHTypedNeuron

                def __init__(self, size, g_max=None, name=None):
                    super().__init__(size, name=name)
                    self.g_max = g_max

                def current(self, V):
                    return self.g_max * (V - -65.0 * u.mV)

            if with_units:
                CustomLeak.parameter_units = {"g_max": u.mS / u.cm**2}
            branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um)
            cell = braincell.Cell(braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1))
            cell.paint(AllRegion(), braincell.mech.Channel(CustomLeak, g_max=0.1 * u.mV))
            return cell

        with self.assertWarnsRegex(FutureWarning, r"declared in .*CustomLeak.parameter_units"):
            cell = build(with_units=True)
        with self.assertRaisesRegex(ValueError, r"\(declared in .*CustomLeak.parameter_units\); got 0.1 mV"):
            cell.init_state()
        build(with_units=False).init_state()

    def test_parameter_units_merge_along_the_mro_and_name_real_parameters(self):
        import brainunit as u
        from braincell import Channel

        class Base(Channel):
            parameter_units = {"g_max": u.mS / u.cm**2, "tau": u.ms}

            def __init__(self, size, g_max=None, tau=None, name=None):
                super().__init__(size, name=name)

        class Child(Base):
            parameter_units = {"tau": u.second}

        info = Child.parameter_info()
        self.assertEqual(info["g_max"].declared_in, f"{Base.__qualname__}.parameter_units")
        self.assertEqual(info["tau"].declared_in, f"{Child.__qualname__}.parameter_units")
        self.assertEqual(info["tau"].prototype.unit, u.second)
        self.assertIsNone(info["tau"].default)
        self.assertIn("size", info)

        class Typo(Channel):
            parameter_units = {"gmax": u.mS / u.cm**2}

        with self.assertRaisesRegex(TypeError, r"Typo.parameter_units names unknown constructor parameters \['gmax'\]"):
            Typo.parameter_info()

    def test_public_namespace_reexports_this_module(self) -> None:
        import braincell
        import braincell._base_channel as channel_mod

        self.assertIs(braincell.IonChannel, channel_mod.IonChannel)
        self.assertIs(braincell.Channel, channel_mod.Channel)
        self.assertIs(braincell.Synapse, channel_mod.Synapse)
        self.assertIs(braincell.IonInfo, channel_mod.IonInfo)

    def test_channel_and_synapse_derive_from_ion_channel(self) -> None:
        from braincell._base_channel import Channel, IonChannel, Synapse

        self.assertTrue(issubclass(Channel, IonChannel))
        self.assertTrue(issubclass(Synapse, IonChannel))


class DirectConstructionUnitCheckTest(unittest.TestCase):
    """Channel and Ion check parameter units after their outermost ``__init__`` (MECH-28)."""

    def test_registry_classes_constructed_with_defaults_do_not_warn(self):
        import warnings

        from braincell.mech import get_registry

        counts = {}
        for category in ("channel", "ion"):
            for _, cls in get_registry().items(category):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    cls(1)
                counts[category] = counts.get(category, 0) + 1
        self.assertGreaterEqual(counts["channel"], 112)
        self.assertGreaterEqual(counts["ion"], 23)

    def test_runtime_nodes_built_by_a_cell_do_not_warn(self):
        import warnings

        import brainunit as u
        import braincell
        from braincell.filter import AllRegion

        branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um)
        cell = braincell.Cell(braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(2))
        cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak", g_max=0.2 * u.mS / u.cm**2))
        cell.paint(AllRegion(), braincell.mech.Ion("CalciumDetailed", tau=10.0 * u.ms))
        cell.paint(AllRegion(), braincell.mech.Channel("CaL_IS2008", g_max=lambda ctx: 1.0 * u.mS / u.cm**2))
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cell.init_state()

    def test_wrapping_keeps_the_constructor_signature(self):
        import inspect

        import brainunit as u
        from braincell import Channel
        from braincell.channel import IL
        from braincell.ion import SodiumFixed

        for cls in (IL, SodiumFixed):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(inspect.signature(cls.__init__), inspect.signature(cls.__init__.__wrapped__))
                self.assertEqual(cls.__init__.__name__, "__init__")
        self.assertEqual(list(inspect.signature(IL.__init__).parameters), ["self", "size", "g_max", "E", "name"])

        class Custom(Channel):
            parameter_units = {"g_max": u.mS / u.cm**2}

            def __init__(self, size, g_max=None, name=None):
                super().__init__(size, name=name)
                self.g_max = g_max

        self.assertEqual(str(inspect.signature(Custom.__init__)), "(self, size, g_max=None, name=None)")
        with self.assertWarnsRegex(FutureWarning, "declared in .*Custom.parameter_units\\); got 0.1 mV") as caught:
            Custom(1, g_max=0.1 * u.mV)
        self.assertEqual(caught.filename, __file__)

    def test_a_failed_constructor_does_not_disable_later_checks(self):
        from braincell.channel import IL

        class Failing(IL):
            def __init__(self, size, **kwargs):
                super().__init__(size, **kwargs)
                raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            Failing(1, g_max=0.1)  # no check: the constructor did not return
        with self.assertWarns(FutureWarning):
            IL(1, g_max=0.1)

    def test_synapses_are_not_wrapped(self):
        from braincell.synapse import ExpSyn

        self.assertFalse(hasattr(ExpSyn.__init__, "__wrapped__"))


if __name__ == "__main__":
    unittest.main()
