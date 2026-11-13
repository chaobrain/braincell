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

"""Physical constants for library code, held as NumPy scalars (OPT-17).

brainunit stores ``gas_constant`` and ``faraday_constant`` with a 0-d
``numpy.ndarray`` mantissa, one object for the whole process. jaxlib caches
the device form of a host ndarray per array object and returns it for as long
as it is alive, whatever ``jax_enable_x64`` is by then: once a float64 Cell's
compiled program captured the constant, a float32 Cell that multiplies by it
received a float64 buffer and failed with ``RuntimeProgramInputMismatch``
(the reverse order failed the same way). A ``numpy.float64`` scalar is not
cached per object and promotes exactly as the 0-d array does, so results are
unchanged bit for bit at both precisions.

Library code uses these names and never ``u.gas_constant`` or
``u.faraday_constant`` directly (``_constants_test`` scans for it).
"""

import brainunit as u
import numpy as np

__all__ = ["GAS_CONSTANT", "FARADAY_CONSTANT"]


def _numpy_scalar(constant: u.Quantity) -> u.Quantity:
    return u.Quantity(np.float64(constant.mantissa), unit=constant.unit)


#: Molar gas constant R, 8.31446261815324 J/(K mol) (CODATA 2018, exact).
GAS_CONSTANT = _numpy_scalar(u.gas_constant)

#: Faraday constant F, 96485.33212... C/mol (CODATA 2018, exact).
FARADAY_CONSTANT = _numpy_scalar(u.faraday_constant)
