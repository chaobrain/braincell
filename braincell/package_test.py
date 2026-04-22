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

"""Package-level smoke tests for ``braincell``."""

import subprocess
import sys
import unittest


class BrainCellLazyNeuromorphoImportTest(unittest.TestCase):
    """LOW-03: importing braincell must not load io.neuromorpho eagerly."""

    def test_importing_braincell_does_not_load_neuromorpho(self) -> None:
        # Use a subprocess so we test a cold import — wiping sys.modules in the
        # live interpreter corrupts the brainstate registry for every test
        # that follows.
        script = (
            "import sys\n"
            "import braincell\n"
            "loaded = 'braincell.io.neuromorpho' in sys.modules\n"
            "print('loaded' if loaded else 'lazy')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "lazy", result.stderr)

    def test_attribute_access_still_works(self) -> None:
        import braincell

        fn = braincell.load_neuromorpho
        self.assertTrue(callable(fn))
        self.assertIn("braincell.io.neuromorpho", sys.modules)


if __name__ == "__main__":
    unittest.main()
