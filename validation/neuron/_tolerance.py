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

"""Error metrics and tiered tolerances for the NEURON comparison suites.

Metrics over every case and sample of one protocol: ``e_V``, the largest
absolute voltage difference (mV, current clamp); ``e_s``, the largest
absolute difference over all compared states; ``e_I``, the largest absolute
current difference over the largest NEURON current magnitude of the whole
protocol (voltage clamp), so that commands that barely activate a channel do
not amplify rounding.

Tier T0 (the class integrates the ``.mod`` equations, constants and initial
values) has the limits :data:`T0_LIMITS`. T1 is an explained difference of
class (c) or (d), with a per-configuration, per-protocol limit and a heading
of the suite's known-differences document. A configuration absent from a
suite's table is T0.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re

import numpy as np

T0_LIMITS = {"e_V": 1e-6, "e_I": 1e-10, "e_s": 1e-10}
"""T0 limit of each metric."""

CURRENT_CLAMP_PROTOCOLS = ("vinit_celsius", "dc", "ac")
VOLTAGE_CLAMP_PROTOCOLS = ("vc_steps", "vc_ramp")
PROTOCOL_METRICS = {
    **{name: ("e_V",) for name in CURRENT_CLAMP_PROTOCOLS},
    **{name: ("e_I", "e_s") for name in VOLTAGE_CLAMP_PROTOCOLS},
}
"""Metrics asserted for each protocol."""

DEVIATION_CLASSES = {
    "c": "numerical approximation in the reference implementation",
    "d": "intentional discretization or model difference",
}
"""Deviation classes that may carry a T1 limit; (a) and (b) are fixed instead."""


def _pairs(pairs: Iterable[tuple[object, object]]) -> list[tuple[np.ndarray, np.ndarray]]:
    resolved = []
    for braincell, neuron in pairs:
        braincell, neuron = np.asarray(braincell, dtype=float), np.asarray(neuron, dtype=float)
        if braincell.shape != neuron.shape:
            raise ValueError(f"trace shapes differ: {braincell.shape!r} vs {neuron.shape!r}.")
        resolved.append((braincell, neuron))
    if not resolved:
        raise ValueError("at least one (braincell, neuron) trace pair is required.")
    return resolved


def max_abs_error(pairs: Iterable[tuple[object, object]]) -> float:
    """Return ``e_V`` or ``e_s``: the largest absolute difference over all pairs."""
    return max(float(np.max(np.abs(b - n), initial=0.0)) for b, n in _pairs(pairs))


def relative_current_error(pairs: Iterable[tuple[object, object]]) -> float:
    """Return ``e_I``: the largest absolute difference over the largest ``|I_neuron|``.

    Raises
    ------
    ValueError
        If every NEURON current is zero.
    """
    resolved = _pairs(pairs)
    scale = max(float(np.max(np.abs(n), initial=0.0)) for _, n in resolved)
    if scale == 0.0:
        raise ValueError("the NEURON current is zero in every case; e_I is undefined.")
    return max(float(np.max(np.abs(b - n), initial=0.0)) for b, n in resolved) / scale


@dataclass(frozen=True)
class ToleranceEntry:
    """A T1 limit of ``config`` (path under ``configs/`` without ``.json``) under ``protocol``.

    ``limits`` raises the listed metrics above T0; the protocol's other
    metrics keep their T0 limit. ``section`` is the known-differences heading
    that explains the ``deviation_class`` (``"c"`` or ``"d"``).
    """

    config: str
    protocol: str
    limits: Mapping[str, float]
    deviation_class: str
    section: str
    tier: str = field(default="T1")

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOL_METRICS:
            raise ValueError(f"{self.config}: unknown protocol {self.protocol!r}.")
        if self.tier != "T1":
            raise ValueError(f"{self.config} {self.protocol}: only T1 entries are listed, got {self.tier!r}.")
        if self.deviation_class not in DEVIATION_CLASSES:
            raise ValueError(
                f"{self.config} {self.protocol}: deviation class must be one of "
                f"{sorted(DEVIATION_CLASSES)}, got {self.deviation_class!r}."
            )
        if not self.section.strip():
            raise ValueError(f"{self.config} {self.protocol}: a known-differences section is required.")
        if not self.limits:
            raise ValueError(f"{self.config} {self.protocol}: a T1 entry needs at least one limit.")
        for metric, limit in self.limits.items():
            if metric not in PROTOCOL_METRICS[self.protocol]:
                raise ValueError(f"{self.config} {self.protocol}: metric {metric!r} does not apply.")
            if not limit > T0_LIMITS[metric]:
                raise ValueError(f"{self.config} {self.protocol}: {metric} limit {limit!r} is not above T0.")


class ToleranceTable:
    """Look up the limit of a metric for a configuration and protocol."""

    def __init__(self, entries: Iterable[ToleranceEntry]):
        self._entries: dict[tuple[str, str], ToleranceEntry] = {}
        for entry in entries:
            key = (entry.config, entry.protocol)
            if key in self._entries:
                raise ValueError(f"duplicate tolerance entry for {key!r}.")
            self._entries[key] = entry

    @property
    def entries(self) -> tuple[ToleranceEntry, ...]:
        return tuple(self._entries.values())

    def tier(self, config: str, protocol: str) -> str:
        return "T1" if (config, protocol) in self._entries else "T0"

    def limit(self, config: str, protocol: str, metric: str) -> float:
        """Return the limit; T0 unless an entry raises it."""
        if metric not in PROTOCOL_METRICS.get(protocol, ()):
            raise ValueError(f"metric {metric!r} does not apply to protocol {protocol!r}.")
        entry = self._entries.get((config, protocol))
        if entry is not None and metric in entry.limits:
            return float(entry.limits[metric])
        return T0_LIMITS[metric]

    def missing_sections(self, document: str) -> list[str]:
        """Return the entry sections that are not headings of ``document`` (Markdown)."""
        headings = {match.strip() for match in re.findall(r"^#+\s+(.+?)\s*#*$", document, flags=re.MULTILINE)}
        return sorted({entry.section for entry in self._entries.values()} - headings)
