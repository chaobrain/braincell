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

"""The axial operator of a compiled run: built once, before the steps, or given to the loop (CELL-50, CELL-52).

:meth:`Cell.run` and ``Network.run`` wrap their step loops with
:func:`axial_operator_before_steps`; the operator itself comes from
:mod:`braincell.quad._staggered`.
"""

import contextlib
import functools

from braincell.quad._staggered import axial_operator_per_run, installed_axial_operator


def axial_operator_before_steps(loop, cells, *, given: bool = False):
    """``loop``, building the axial operator of each of ``cells`` once, before the steps it traces.

    The geometry is the same at every step of a run; built inside the step,
    the operator would be built again at every step, and at every stage of
    the generic solvers (CELL-50). See ``axial_operator_per_run``.

    With ``given``, the operators are inputs instead: the returned function
    takes first one entry per cell, the leaves of
    :func:`~braincell.quad._staggered.host_axial_operator` (or ``None`` to
    build that cell's as above), and installs them with the layout the
    cell's runtime keeps, so the program builds no operator (CELL-52).
    """

    if given:

        @functools.wraps(loop)
        def with_given(operators, *args):
            with contextlib.ExitStack() as stack:
                for cell, leaves in zip(cells, operators, strict=True):
                    if leaves is None:
                        stack.enter_context(axial_operator_per_run(cell))
                    else:
                        built = cell._runtime.axial_operator_layout.join(leaves)
                        stack.enter_context(installed_axial_operator(cell, built))
                return loop(*args)

        return with_given

    @functools.wraps(loop)
    def built_once(*args):
        with contextlib.ExitStack() as stack:
            for cell in cells:
                stack.enter_context(axial_operator_per_run(cell))
            return loop(*args)

    return built_once
