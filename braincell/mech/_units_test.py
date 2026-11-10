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

import os
import sys
import unittest
from unittest import mock
import warnings

import brainunit as u

import braincell
from braincell.mech import Channel
from braincell.mech._units import _PACKAGE_DIR, user_frame, warn_at_user_frame, warn_invalid_declared_parameters


def _from_package_file(source: str, namespace: dict) -> dict:
    """Run ``source`` as if it were a non-test braincell module."""
    exec(compile(source, os.path.join(_PACKAGE_DIR, "fake.py"), "exec"), namespace)
    return namespace


class UserFrameTest(unittest.TestCase):
    """MECH-26: the call-site rule shared with ``paint``/``place`` origins."""

    def test_package_frames_are_skipped_and_test_files_are_not(self) -> None:
        namespace = _from_package_file("found = user_frame(sys._getframe())", {"user_frame": user_frame, "sys": sys})
        frame, hops = namespace["found"]
        self.assertEqual((frame.f_code.co_filename, hops), (__file__, 1))
        self.assertEqual(user_frame(None), (None, 0))

    def test_warning_points_past_package_frames(self) -> None:
        with self.assertWarns(FutureWarning) as caught:
            _from_package_file("warn_at_user_frame('deep')", {"warn_at_user_frame": warn_at_user_frame})
        # The ``exec`` line in ``_from_package_file`` is the nearest user frame.
        exec_line = _from_package_file.__code__.co_firstlineno + 2
        self.assertEqual((caught.filename, caught.lineno), (__file__, exec_line))

    def test_without_frames_the_warning_points_at_the_caller(self) -> None:
        for getframe in (None, lambda depth: None):
            with self.subTest(getframe=getframe), mock.patch.object(sys, "_getframe", getframe):
                with self.assertWarns(FutureWarning) as caught:
                    warn_at_user_frame("no frames")
                self.assertEqual(caught.filename, __file__)


class _BadUnits(braincell.Channel):
    """``parameter_info()`` raises: ``parameter_units`` names no parameter."""

    parameter_units = {"missing": u.mV}

    def __init__(self, size, g_max=None, name=None):
        super().__init__(size, name=name)


class WarnInvalidDeclaredParametersTest(unittest.TestCase):
    def test_classes_without_metadata_are_left_to_init_state(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Channel(_BadUnits, g_max=0.1, typo=1.0)

    def test_unknown_name_without_a_close_match_has_no_suggestion(self) -> None:
        with self.assertWarns(FutureWarning) as caught:
            Channel("IL", zzz=1.0)
        self.assertEqual(
            str(caught.warning),
            "Channel 'IL' got an unexpected parameter 'zzz'. This will raise TypeError at declaration from braincell 0.4.",
        )

    def test_cell_supplied_arguments_are_unexpected(self) -> None:
        with self.assertWarnsRegex(FutureWarning, "unexpected parameter 'size'"):
            warn_invalid_declared_parameters("channel", "IL", None, {"size": 3})


if __name__ == "__main__":
    unittest.main()
