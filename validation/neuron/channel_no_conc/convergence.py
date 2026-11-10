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

"""``dt`` convergence and stiffness table of the channel_no_conc catalogue.

Measured by ``validation/neuron/_convergence.py``, without NEURON; the NEURON
limits are in ``tolerances.py`` and the two tables do not read each other.

- :data:`CONFIGS`: per configuration, the classification of the Cell's
  default path under ``vc_steps`` (or the relaxation protocol) and, for a
  convergent path, the limits of ``e_I`` and ``e_s`` at ``dt = 0.025 ms``:
  about 1.5-2x the measured value, rounded up to 1, 2 or 5 times a power of
  ten. An exact path is asserted at 1e-10 for every ``dt``.
- :data:`CLASSES`: per channel class, the decade of ``rho_max`` and whether
  the class is stiff at ``dt = 0.025 ms``.
- :data:`KNOWN_SINGULARITIES`: the classes whose gate values, derivatives or
  current are not finite somewhere on the 0.001 mV grid, and where. A fix
  removes its entry; any other non-finite value fails.

:data:`COVERAGE_SCOPE` lists the names the tables must cover; ``None``
covers every registered channel class or every file under ``configs/``.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from validation.neuron import _convergence as cv

CONFIGS_ROOT = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class ConfigEntry:
    """Classification of ``config`` (path under ``configs/`` without ``.json``) and its limits."""

    config: str
    classification: str
    protocol: str = cv.STEPS
    limits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol not in (cv.STEPS, cv.RELAXATION):
            raise ValueError(f"{self.config}: protocol must be {cv.STEPS!r} or {cv.RELAXATION!r}.")
        if self.classification == cv.EXACT:
            if self.limits:
                raise ValueError(f"{self.config}: an exact path is asserted at 1e-10 and takes no limits.")
            return
        if self.classification not in (cv.CONVERGENT, cv.ABOVE_GUIDE_LINE):
            raise ValueError(
                f"{self.config}: classification must be one of {cv.EXACT!r}, {cv.CONVERGENT!r} or "
                f"{cv.ABOVE_GUIDE_LINE!r}; a path that does not converge is fixed, not listed."
            )
        if not self.limits:
            raise ValueError(f"{self.config}: a convergent path needs its e_I/e_s limits.")
        for metric, limit in self.limits.items():
            if metric not in ("e_I", "e_s"):
                raise ValueError(f"{self.config}: unknown metric {metric!r}.")
            if not limit > cv.EXACT_LIMIT:
                raise ValueError(f"{self.config}: the {metric} limit {limit!r} is not above 1e-10.")


@dataclass(frozen=True)
class ClassEntry:
    """``floor(log10(rho_max))`` of channel class ``name`` and whether it is stiff at 0.025 ms."""

    name: str
    decade: int | None
    stiff: bool


CONFIGS = (
    ConfigEntry("ma20_goc/kv3p4_ma20_goc", cv.EXACT),
    # backward_euler x 1, measured 7.8e-2 / 3.2e-2 at 0.025 ms (order 0.90 / 0.95).
    ConfigEntry("ma20_goc/nav1p6_ma20_goc", cv.ABOVE_GUIDE_LINE, limits={"e_I": 2e-1, "e_s": 5e-2}),
)

CLASSES = (
    ClassEntry("Kv3p4_MA2020_GoC", 0, False),  # 6.88 /ms, m gate at +60 mV
    ClassEntry("Nav1p6_MA2020_GoC", 5, True),  # 1.96e5 /ms at +60 mV
)

KNOWN_SINGULARITIES = {
    "CaHT_Re1993": {"p.value": (-27.0,), "p.derivative": (-27.0,), "current": (-27.0,)},
    "CaHVA_SU2015_DCN": {"m.derivative": (-8.9,), "current": (0.0,)},
    "CaLVA_SU2015_DCN": {"current": (0.0,)},
    "Cav1p2_MA2020_GoC": {"m.derivative": (-8.124,)},
    "Cav1p2_MA2025_BC": {"m.derivative": (-8.124,)},
    "K_Kv_test": {"n.derivative": (25.0,)},
}
"""Removable 0/0 singularities, each to be replaced by its analytic limit."""

COVERAGE_SCOPE = {
    "configs": ("ma20_goc/kv3p4_ma20_goc", "ma20_goc/nav1p6_ma20_goc"),
    "classes": ("Kv3p4_MA2020_GoC", "Nav1p6_MA2020_GoC"),
}


def config_names() -> tuple[str, ...]:
    """Every configuration under ``configs/``, as a path without ``.json``."""
    return tuple(sorted(path.relative_to(CONFIGS_ROOT).with_suffix("").as_posix() for path in CONFIGS_ROOT.rglob("*.json")))


def class_names() -> tuple[str, ...]:
    """Every registered channel class."""
    from braincell.mech import get_registry

    return tuple(sorted(get_registry().names("channel")))


def coverage_problems(
    configs: Iterable[ConfigEntry] = CONFIGS,
    classes: Iterable[ClassEntry] = CLASSES,
    scope: Mapping[str, Iterable[str] | None] = COVERAGE_SCOPE,
) -> list[str]:
    """Return what the tables miss in ``scope`` and the names they list that do not exist."""
    problems = []
    for kind, entries, existing in (
        ("configs", [entry.config for entry in configs], config_names()),
        ("classes", [entry.name for entry in classes], class_names()),
    ):
        duplicates = sorted({name for name in entries if entries.count(name) > 1})
        problems += [f"{kind}: {name} is listed twice" for name in duplicates]
        problems += [f"{kind}: {name} does not exist" for name in entries if name not in existing]
        required = existing if scope[kind] is None else scope[kind]
        problems += [f"{kind}: {name} is not in the table" for name in required if name not in entries]
    return problems


def assert_covered(**kwargs) -> None:
    """Raise ``AssertionError`` listing every :func:`coverage_problems` item."""
    problems = coverage_problems(**kwargs)
    if problems:
        raise AssertionError("convergence table coverage:\n" + "\n".join(problems))
