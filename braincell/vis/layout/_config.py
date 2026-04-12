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

"""User-tunable knobs for the 2D layout engine.

Before the split of ``layout2d.py`` into :mod:`braincell.vis.layout`,
every layout parameter was a bare module-level constant buried in the
middle of the file. :class:`LayoutConfig` promotes the headline knobs
into a single frozen dataclass so callers can tune a visual without
forking the whole engine.

The goal here is not to expose *every* internal number — a handful of
low-level interpolation ratios (e.g. the 0.28/0.72 bands inside
``_stem_segment_angles_rad``) are left as hard-coded defaults because
they are not meaningful tuning targets. What :class:`LayoutConfig`
*does* surface is the set of parameters that determine the overall
shape of a layout:

* Per-family bend fractions and angular spans.
* Collision detection margins and retry limits.
* The scoring-function weights that govern how the stem family trades
  off physical overlap against angular fidelity.

The dataclass is frozen so a caller can hold on to a
:class:`LayoutConfig` and reuse it without worrying about mutation.
Passing ``layout_config=None`` through the entire layout stack means
"use :data:`DEFAULT_LAYOUT_CONFIG`", which recovers the pre-Phase-2
behaviour byte-for-byte.

"""


import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutConfig:
    """Configurable parameters for the 2D layout engine.

    Parameters
    ----------
    collision_margin_um : float
        Radius of the "too close" bubble around each placed branch.
        Any candidate whose closest approach to a previously-placed
        branch is below this threshold incurs a soft penalty of
        ``margin - distance`` per offending segment pair. Proper
        intersections incur a flat 1000.0 penalty regardless.
    collision_retry_limit : int
        Maximum number of candidate placements the stem-linear solver
        tries before accepting the best-so-far. Raising this makes
        dense morphologies cleaner at the cost of linear runtime.
    stem_collision_window : int
        How many of the most recently placed branches the stem-tree
        solver checks for collisions against. With the spatial-hash
        backend the cost per candidate is roughly linear in the
        number of *segments* actually within ``collision_cell_size_um``
        of the candidate; a larger window is therefore affordable but
        still bounded to avoid rescoring the whole tree at deep forks.
    collision_cell_size_um : float
        Grid cell size for the 2D spatial-hash collision backend in
        :mod:`_collision`. Should be a small multiple of the typical
        branch segment length; too small and the hash wastes memory
        with sparsely-populated cells, too large and queries degrade
        toward the O(n²) brute force.
    default_bend_fraction : float
        Fraction of a branch's total length used to bend from the
        attach tangent to the target angle in the stem-linear and
        legacy families.
    balloon_bend_fraction : float
        Same idea for the balloon family. Smaller values make each
        branch curve more sharply near its root.
    fan_bend_fraction : float
        Same idea for the fan family.
    radial_bend_fraction : float
        Same idea for the radial_360 family.
    fan_root_left_span_rad : float
        Angular span used for root children attached at the extreme
        left side of the soma.
    fan_root_middle_upper_span_rad : float
        Angular span used for the upper root sector for center-attached
        children.
    fan_root_middle_lower_span_rad : float
        Angular span used for the lower root sector for center-attached
        children.
    fan_root_right_span_rad : float
        Angular span used for root children attached away from the
        center/right side of the soma.
    fan_root_left_max_parent_x : float
        Maximum ``parent_x`` classified into the left sector.
    fan_root_middle_min_parent_x : float
        Inclusive lower bound of the center band.
    fan_root_middle_max_parent_x : float
        Inclusive upper bound of the center band.
    stem_root_full_span_rad : float
        Full angular span used for root children when ``root_layout``
        is not ``'type_split'`` (all children packed into one arc).
    stem_root_group_span_rad : float
        Angular span allocated *per group* (axon group or dendrite
        group) when ``root_layout='type_split'``. The two groups sit
        on opposite half-planes centered on ``±π/2``.
    balloon_root_span_rad : float
        Angular span used by the balloon family for the root fork
        when not splitting by type.
    balloon_child_span_rad : float
        Angular span used by the balloon family for each non-root
        fork, centered on the parent's tangent.
    balloon_type_split_span_rad : float
        Per-group span for the balloon family when
        ``root_layout='type_split'``.
    radial_root_span_rad : float
        Angular span used by the radial_360 family for the root fork
        (defaults to a full 2π).
    radial_child_span_rad : float
        Angular span used by the radial_360 family for each non-root
        fork.
    legacy_root_child_span_rad : float
        Angular span for root children in the legacy layout.
    stem_collision_weight : float
        Weight applied to the raw collision score when computing the
        total stem-tree score. With ``collision_weight=100`` a single
        intersection (score ``1000.0``) contributes ``100 000`` to the
        total, which dominates every other term — intentional, since
        intersections are always worse than a bad angle.
    stem_tail_delta_weight : float
        Weight on the tail-angle deviation from the desired target.
        Increase to make the engine more faithful to the target angle
        at the cost of possible overlap.
    stem_settle_delta_weight : float
        Weight on the "settle" angle deviation from the tail. Keeps
        the mid-branch from over-rotating between launch and tail.
    stem_overturn_weight : float
        Weight on turns wider than π/2 between launch and tail.
        Discourages hairpin turns.
    stem_trunk_tail_delta_weight : float
        Extra per-trunk penalty on tail deviation from the attach
        tangent. Keeps trunks visually continuous across a fork.
    stem_side_opening_weight : float
        Extra per-side penalty that rewards launches "opening away"
        from the desired tail by at least ``max(|desired|, 55°)``.
    """

    # --- Collision detection ---
    collision_margin_um: float = 2.0
    collision_retry_limit: int = 8
    stem_collision_window: int = 24
    collision_cell_size_um: float = 20.0

    # --- Bend fractions ---
    default_bend_fraction: float = 0.4
    balloon_bend_fraction: float = 0.22
    fan_bend_fraction: float = 0.24
    radial_bend_fraction: float = 0.25

    # --- Root layout spans (radians) ---
    stem_root_full_span_rad: float = math.radians(150.0)
    stem_root_group_span_rad: float = math.radians(120.0)
    balloon_root_span_rad: float = math.radians(180.0)
    balloon_child_span_rad: float = math.radians(120.0)
    balloon_type_split_span_rad: float = math.radians(110.0)
    fan_root_left_span_rad: float = math.radians(95.0)
    fan_root_middle_upper_span_rad: float = math.radians(70.0)
    fan_root_middle_lower_span_rad: float = math.radians(70.0)
    fan_root_right_span_rad: float = math.radians(95.0)
    radial_root_span_rad: float = 2.0 * math.pi
    radial_child_span_rad: float = math.radians(150.0)
    legacy_root_child_span_rad: float = math.radians(120.0)

    # --- Fan root parent_x bins ---
    fan_root_left_max_parent_x: float = 0.02
    fan_root_middle_min_parent_x: float = 0.35
    fan_root_middle_max_parent_x: float = 0.65

    # --- Stem tree-layout scoring weights ---
    #
    # The total score for a stem candidate is:
    #
    #     total = collision_weight      * collision_score
    #           + tail_delta_weight     * |tail - desired|
    #           + settle_delta_weight   * |settle - tail|
    #           + overturn_weight       * max(0, |launch-tail| - π/2)
    #           + (trunk) trunk_tail_delta_weight * |tail - attach|
    #           + (side)  side_opening_weight * max(0, opening_floor - |launch-attach|)
    #
    # where collision_score is a per-pair sum defined in _collision.py
    # (1000.0 per proper intersection, linear in "how close" for the
    # margin band).
    stem_collision_weight: float = 100.0
    stem_tail_delta_weight: float = 3.0
    stem_settle_delta_weight: float = 0.8
    stem_overturn_weight: float = 6.0
    stem_trunk_tail_delta_weight: float = 0.75
    stem_side_opening_weight: float = 2.0


DEFAULT_LAYOUT_CONFIG = LayoutConfig()
"""Process-wide default :class:`LayoutConfig`, used when a caller passes
``layout_config=None`` (i.e. the pre-Phase-2 behaviour)."""
