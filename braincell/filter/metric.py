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

"""Composable spatial metrics for morphology-backed callable contexts."""

__all__ = ["branch_x", "radius", "path_distance_from_soma", "position"]


def _first_attribute(context: object, names: tuple[str, ...], description: str) -> object:
    """Return the first of ``names`` that ``context`` exposes.

    The three context types (:class:`~braincell.filter.SamplingContext`,
    ``SynapseContext``, and ``CVContext``) name the same quantities
    differently -- ``CVContext`` spells ``branch_x`` as ``midpoint`` and
    ``radius`` as ``radius_mid`` -- so a metric accepts an ordered list of
    spellings and reports one ``TypeError`` when a context offers none.

    Parameters
    ----------
    context : object
        Callable-parameter context to read from.
    names : tuple of str
        Attribute names to try, in order of preference.
    description : str
        Noun phrase completing "does not expose ...", used in the error.

    Returns
    -------
    object
        The value of the first attribute present.

    Raises
    ------
    TypeError
        If ``context`` exposes none of ``names``.
    """
    for name in names:
        try:
            return getattr(context, name)
        except AttributeError:
            continue
    raise TypeError(f"{type(context).__name__} does not expose {description}.")


def branch_x(context: object) -> object:
    """Return the normalized branch coordinate represented by ``context``."""
    return _first_attribute(context, ("branch_x", "midpoint"), "a branch coordinate")


def radius(context: object) -> object:
    """Return radius at the location represented by ``context``."""
    return _first_attribute(context, ("radius", "radius_mid"), "a radius")


def path_distance_from_soma(context: object) -> object:
    """Return tree distance from the soma/root reference region."""
    return _first_attribute(context, ("path_distance_from_soma",), "soma-relative distance")


def position(context: object) -> object:
    """Return the morphology-local 3-D position represented by ``context``."""
    return _first_attribute(context, ("position",), "a 3-D position")
