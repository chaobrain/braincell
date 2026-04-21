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

"""Base class for every mechanism declaration in :mod:`braincell.mech`.

:class:`Mechanism` is a deliberately empty marker class that sits at the
top of the declaration-layer hierarchy. Both distributed (:class:`Density`)
and point-located (:class:`Point`) mechanism families inherit from it so
that consumer code can write ``isinstance(x, Mechanism)`` instead of
maintaining a parallel union of concrete types.

The class is intentionally free of abstract methods — the only
requirement it expresses is *being a mechanism*. Validation, equality,
and state live in the concrete subclasses.
"""

__all__ = ["Mechanism"]


class Mechanism:
    """Marker base class for every mechanism declaration.

    All declaration-layer mechanism types (:class:`~braincell.mech.Density`
    and its concrete subclasses :class:`~braincell.mech.Channel` /
    :class:`~braincell.mech.Ion`, and :class:`~braincell.mech.Point` and
    its concrete subclasses :class:`~braincell.mech.CurrentClamp` /
    :class:`~braincell.mech.SineClamp` / :class:`~braincell.mech.FunctionClamp`
    / :class:`~braincell.mech.Probe` / :class:`~braincell.mech.Synapse` /
    :class:`~braincell.mech.Junction`) inherit from this class.

    :class:`Mechanism` exists only to support
    ``isinstance(x, Mechanism)`` dispatch. It defines no fields, no
    methods, and no abstract contract — every concrete subclass owns its
    own data, constructors, and validation rules.

    See Also
    --------
    Density : Base for distributed channel and ion declarations.
    Point : Base for point-located declarations.
    """

    __slots__ = ()
