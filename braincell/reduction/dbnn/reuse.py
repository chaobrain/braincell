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

"""Plan and validate process-local DBNN checkpoint reuse."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import math
from types import MappingProxyType
from typing import Any
import weakref

import brainunit as u
import numpy as np

from braincell.reduction.dbnn.dataset import TeacherSession, TeacherSpec, rasterize_events
from braincell.reduction.dbnn.functional import compute_metrics
from braincell.reduction.dbnn.layout import ChannelAlignment, ChannelLayout, align_channels
from braincell.reduction.dbnn.model import SpikeAlignmentEvidence
from braincell.reduction.dbnn.signature import CellStructuralSignature
from braincell.reduction.dbnn.stimulus import StimulusPlan, generate_multichannel_protocol
from braincell.reduction.dbnn.train import build_fit_mask


DEFAULT_REUSE_VALIDATION_TRACES = 8
DEFAULT_REUSE_VE_THRESHOLD = 0.8
_SEED_MODULUS = 2**31 - 1


def _time_ms(value: Any, *, name: str) -> float:
    if not hasattr(value, "to_decimal"):
        raise TypeError(f"{name} must be a brainunit time quantity.")
    try:
        scalar = float(np.asarray(value.to_decimal(u.ms)).reshape(()))
    except Exception as exc:
        raise TypeError(f"{name} must be a scalar quantity convertible to milliseconds.") from exc
    if not math.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


@dataclass(frozen=True)
class DBNNInputProfile:
    """Describe concrete candidate inputs and validation timing.

    Parameters
    ----------
    layout : ChannelLayout
        Candidate runtime channel layout.
    rate_hz : float or sequence of float
        Scalar or per-channel event rates in hertz.
    dt : brainunit.Quantity
        Positive simulation time step.
    validation_duration : brainunit.Quantity
        Positive validation duration containing a whole number of time steps.
    source_fingerprint : str
        Stable fingerprint of the detailed candidate declaration.
    dynamics_fingerprint : str or None, optional
        Stable source fingerprint excluding spike-readout ``V_th``. Defaults
        to ``source_fingerprint``.
    """

    layout: ChannelLayout
    rate_hz: float | Sequence[float]
    dt: Any
    validation_duration: Any
    source_fingerprint: str
    dynamics_fingerprint: str | None = None
    _dt_ms: float = field(init=False, repr=False)
    _duration_ms: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, ChannelLayout):
            raise TypeError(f"layout must be a ChannelLayout, got {type(self.layout).__name__!r}.")
        rates = np.asarray(self.rate_hz, dtype=float)
        if rates.ndim == 0:
            rates = np.full(self.layout.n_channels, float(rates))
        elif rates.shape != (self.layout.n_channels,):
            raise ValueError(f"rate_hz must be scalar or have shape {(self.layout.n_channels,)}, got {rates.shape}.")
        if not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("rate_hz must contain finite non-negative values.")
        dt_ms = _time_ms(self.dt, name="dt")
        if np.any(rates * dt_ms / 1000.0 > 1.0):
            raise ValueError("rate_hz * dt must define Bernoulli probabilities in [0, 1].")
        duration_ms = _time_ms(self.validation_duration, name="validation_duration")
        steps = round(duration_ms / dt_ms)
        if steps < 1 or not np.isclose(steps * dt_ms, duration_ms, rtol=1e-12, atol=1e-12):
            raise ValueError("validation_duration must be a positive integer multiple of dt.")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint.strip():
            raise ValueError("source_fingerprint must be a non-empty string.")
        object.__setattr__(self, "rate_hz", tuple(float(rate) for rate in rates))
        object.__setattr__(self, "source_fingerprint", self.source_fingerprint.strip())
        dynamics_fingerprint = (
            self.source_fingerprint if self.dynamics_fingerprint is None else self.dynamics_fingerprint
        )
        if not isinstance(dynamics_fingerprint, str) or not dynamics_fingerprint.strip():
            raise ValueError("dynamics_fingerprint must be a non-empty string or None.")
        object.__setattr__(self, "dynamics_fingerprint", dynamics_fingerprint.strip())
        object.__setattr__(self, "_dt_ms", dt_ms)
        object.__setattr__(self, "_duration_ms", duration_ms)

    @property
    def dt_ms(self) -> float:
        """Return the time step in milliseconds."""
        return self._dt_ms

    @property
    def validation_duration_ms(self) -> float:
        """Return the validation duration in milliseconds."""
        return self._duration_ms


@dataclass(frozen=True)
class ReuseValidationConfig:
    """Configure measured checkpoint reuse validation."""

    n_traces: int = DEFAULT_REUSE_VALIDATION_TRACES
    variance_explained_threshold: float = DEFAULT_REUSE_VE_THRESHOLD
    spike_threshold_mv: float = -20.0
    spike_window_pre_ms: float = 3.0
    spike_window_post_ms: float = 10.0
    spike_match_window_ms: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_traces", _positive_integer(self.n_traces, name="n_traces"))
        values = (
            self.variance_explained_threshold,
            self.spike_threshold_mv,
            self.spike_window_pre_ms,
            self.spike_window_post_ms,
            self.spike_match_window_ms,
        )
        if not np.isfinite(values).all():
            raise ValueError("Validation thresholds and windows must be finite.")
        if self.variance_explained_threshold < DEFAULT_REUSE_VE_THRESHOLD:
            raise ValueError("variance_explained_threshold cannot be lower than 0.8.")
        if self.spike_window_pre_ms < 0 or self.spike_window_post_ms < 0 or self.spike_match_window_ms < 0:
            raise ValueError("Validation spike windows must be non-negative.")


@dataclass(frozen=True)
class _ReuseCandidate:
    """Store an in-memory checkpoint and the declaration that fitted it."""

    candidate_id: str
    model: Any
    signature: CellStructuralSignature
    input_profile: DBNNInputProfile
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fit_seed_roots: tuple[int, ...] = ()
    fit_trace_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.signature, CellStructuralSignature):
            raise TypeError("signature must be a CellStructuralSignature.")
        if not isinstance(self.input_profile, DBNNInputProfile):
            raise TypeError("input_profile must be a DBNNInputProfile.")
        if not hasattr(self.model, "predict") or not callable(self.model.predict):
            raise TypeError("model must provide a callable predict(inputs) method.")
        model_layout = getattr(self.model, "layout", None)
        if not isinstance(model_layout, ChannelLayout) or model_layout.fingerprint != self.input_profile.layout.fingerprint:
            raise ValueError("model layout must match its input profile layout exactly.")
        input_alignment = getattr(self.model, "input_alignment", "target-step")
        if input_alignment != "target-step":
            raise ValueError("Reusable models must use target-step input alignment.")
        model_dt = getattr(self.model, "dt_ms", None)
        if model_dt is None or float(model_dt) != self.input_profile.dt_ms:
            raise ValueError("model dt must exactly match its input profile dt after unit conversion.")
        roots = tuple(int(root) for root in self.fit_seed_roots)
        if any(root < 0 or root >= _SEED_MODULUS for root in roots):
            raise ValueError(f"fit_seed_roots must lie in [0, {_SEED_MODULUS}).")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "fit_seed_roots", roots)
        trace_seeds = tuple(int(seed) for seed in self.fit_trace_seeds)
        if len(set(trace_seeds)) != len(trace_seeds) or any(
            seed < 0 or seed >= _SEED_MODULUS for seed in trace_seeds
        ):
            raise ValueError("fit_trace_seeds must be unique and lie in the supported seed range.")
        object.__setattr__(self, "fit_trace_seeds", trace_seeds)


@dataclass(frozen=True)
class ReuseValidationReport:
    """Report one compatibility or measured validation attempt."""

    candidate_id: str
    checkpoint_id: str
    alignment_method: str | None
    permutation: tuple[int, ...]
    seeds: tuple[int, ...]
    valid_count: int
    variance_explained: float
    accepted: bool
    reason: str
    spike_alignment_evidence: SpikeAlignmentEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id must be a non-empty string.")
        if self.alignment_method not in {None, "identity", "branch_position", "identifier_distance"}:
            raise ValueError("alignment_method is invalid.")
        permutation = tuple(int(index) for index in self.permutation)
        if self.alignment_method is not None:
            ChannelAlignment(self.alignment_method, permutation)
        seeds = tuple(int(seed) for seed in self.seeds)
        if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= _SEED_MODULUS for seed in seeds):
            raise ValueError("Validation seeds must be unique and lie in the supported seed range.")
        if isinstance(self.valid_count, bool) or not isinstance(self.valid_count, (int, np.integer)):
            raise TypeError("valid_count must be an integer.")
        if self.valid_count < 0:
            raise ValueError("valid_count must be non-negative.")
        variance_explained = float(self.variance_explained)
        if math.isinf(variance_explained):
            raise ValueError("variance_explained must be finite or NaN.")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be bool.")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string.")
        if self.spike_alignment_evidence is not None and not isinstance(
            self.spike_alignment_evidence, SpikeAlignmentEvidence
        ):
            raise TypeError("spike_alignment_evidence must be a SpikeAlignmentEvidence or None.")
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "valid_count", int(self.valid_count))
        object.__setattr__(self, "variance_explained", variance_explained)


@dataclass(frozen=True)
class ReusePlan:
    """Return the selected checkpoint, alignment, and audit report."""

    candidate: _ReuseCandidate
    alignment: ChannelAlignment
    attempts: tuple[ReuseValidationReport, ...]
    fitted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, _ReuseCandidate):
            raise TypeError("candidate must be an internal reuse candidate.")
        if not isinstance(self.alignment, ChannelAlignment):
            raise TypeError("alignment must be a ChannelAlignment.")
        attempts = tuple(self.attempts)
        if any(not isinstance(item, ReuseValidationReport) for item in attempts):
            raise TypeError("attempts must contain ReuseValidationReport objects.")
        if type(self.fitted) is not bool:
            raise TypeError("fitted must be bool.")
        object.__setattr__(self, "attempts", attempts)


def _new_seed_state(root: int, reserved: Mapping[str, Sequence[int]] | None = None) -> dict[str, Any]:
    if isinstance(root, bool) or not isinstance(root, (int, np.integer)):
        raise TypeError("root must be an integer.")
    state = {"root": int(root), "by_name": {}, "owners": {}}
    for name, seeds in (reserved or {}).items():
        _claim_seeds(state, name, seeds)
    return state


def _claim_seeds(state: dict[str, Any], name: str, seeds: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(name, str) or not name:
        raise ValueError("Seed allocation names must be non-empty strings.")
    if name in state["by_name"]:
        raise ValueError(f"Seed allocation name {name!r} is already present.")
    values = tuple(int(seed) for seed in seeds)
    if len(set(values)) != len(values):
        raise ValueError(f"Seed allocation {name!r} contains an overlap.")
    if any(seed < 0 or seed >= _SEED_MODULUS for seed in values):
        raise ValueError(f"Seeds must lie in [0, {_SEED_MODULUS}).")
    overlap = {seed: state["owners"][seed] for seed in values if seed in state["owners"]}
    if overlap:
        raise ValueError(f"Seed allocation {name!r} overlaps existing allocations: {overlap!r}.")
    state["by_name"][name] = values
    state["owners"].update({seed: name for seed in values})
    return values


def _derive_seeds(state: dict[str, Any], name: str, count: int = 1) -> tuple[int, ...]:
    count = _positive_integer(count, name="count")
    seeds = []
    nonce = 0
    while len(seeds) < count:
        payload = f"{state['root']}:{name}:{len(seeds)}:{nonce}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % _SEED_MODULUS
        if seed not in state["owners"] and seed not in seeds:
            seeds.append(seed)
        nonce += 1
    return _claim_seeds(state, name, seeds)


def _allocated_seeds(state: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(state["owners"]))


class _NetworkReuseCache:
    """Keep candidates weakly scoped to one live Network identity."""

    def __init__(self) -> None:
        self._networks: dict[
            int,
            tuple[weakref.ReferenceType[Any], dict[tuple[str, str], list[_ReuseCandidate]], set[int]],
        ] = {}

    def _groups(self, network: Any, *, create: bool) -> dict[tuple[str, str], list[_ReuseCandidate]]:
        identity = id(network)
        entry = self._networks.get(identity)
        if entry is not None and entry[0]() is network:
            return entry[1]
        if not create:
            return {}
        try:
            reference = weakref.ref(
                network,
                lambda current, identity=identity: self._networks.pop(identity, None)
                if self._networks.get(identity, (None,))[0] is current
                else None,
            )
        except TypeError as exc:
            raise TypeError("network must support weak references.") from exc
        groups: dict[tuple[str, str], list[_ReuseCandidate]] = {}
        self._networks[identity] = (reference, groups, set())
        return groups

    def used_seeds(self, network: Any) -> tuple[int, ...]:
        """Return all fitting and validation seeds consumed in this Network scope."""
        self._groups(network, create=False)
        entry = self._networks.get(id(network))
        return () if entry is None or entry[0]() is not network else tuple(sorted(entry[2]))

    def reserve_seeds(self, network: Any, seeds: Sequence[int]) -> None:
        """Reserve consumed seeds against every later planning request."""
        self._groups(network, create=True)
        entry = self._networks[id(network)]
        values = {int(seed) for seed in seeds}
        overlap = values.intersection(entry[2])
        if overlap:
            raise ValueError(f"Reduction seeds overlap data already consumed in this Network: {sorted(overlap)!r}.")
        entry[2].update(values)

    def add(self, network: Any, candidate: _ReuseCandidate) -> None:
        """Add a candidate under its Cell signature for one Network."""
        if not isinstance(candidate, _ReuseCandidate):
            raise TypeError("candidate must be a _ReuseCandidate.")
        groups = self._groups(network, create=True)
        group = groups.setdefault(candidate.signature.key, [])
        if any(item.candidate_id == candidate.candidate_id for item in group):
            raise ValueError(f"Candidate {candidate.candidate_id!r} is already cached for this Network.")
        group.append(candidate)

    def candidates(self, network: Any, signature: CellStructuralSignature) -> tuple[_ReuseCandidate, ...]:
        """Return insertion-ordered candidates for a Network and signature."""
        if not isinstance(signature, CellStructuralSignature):
            raise TypeError("signature must be a CellStructuralSignature.")
        groups = self._groups(network, create=False)
        return tuple(groups.get(signature.key, ()))

    def all_candidates(self, network: Any) -> tuple[_ReuseCandidate, ...]:
        """Return every candidate allocated in one Network scope."""
        groups = self._groups(network, create=False)
        return tuple(candidate for group in groups.values() for candidate in group)


def validate_candidate(
    candidate: _ReuseCandidate,
    target_profile: DBNNInputProfile,
    alignment: ChannelAlignment,
    *,
    seed: int | None = None,
    validation_plan: StimulusPlan | None = None,
    teacher: TeacherSpec,
    config: ReuseValidationConfig = ReuseValidationConfig(),
    teacher_factory: Callable[..., Any] = TeacherSession,
) -> ReuseValidationReport:
    """Measure one candidate against detailed target traces.

    Parameters
    ----------
    candidate : _ReuseCandidate
        Cached checkpoint to evaluate.
    target_profile : DBNNInputProfile
        Concrete target input distribution and channel layout.
    alignment : ChannelAlignment
        Mapping from target runtime channels to model channels.
    seed : int or None, optional
        Candidate-specific root used when ``validation_plan`` is omitted.
    validation_plan : StimulusPlan or None, optional
        Pre-generated candidate-specific plan.
    teacher : TeacherSpec
        Detailed target Cell factory and recording configuration.
    config : ReuseValidationConfig, optional
        Validation trace count, threshold, and mask semantics.
    teacher_factory : callable, optional
        Factory compatible with :class:`TeacherSession`.

    Returns
    -------
    ReuseValidationReport
        Immutable measured acceptance result.
    """
    if target_profile.dt_ms != candidate.input_profile.dt_ms:
        raise ValueError("Candidate and target dt must match exactly after unit conversion.")
    if isinstance(teacher, TeacherSpec):
        teacher_dt_ms = _time_ms(teacher.dt, name="teacher.dt")
        if teacher_dt_ms != target_profile.dt_ms:
            raise ValueError("Teacher and target profile dt must match exactly after unit conversion.")
        if teacher.source_fingerprint != target_profile.source_fingerprint:
            raise ValueError("Teacher and target profile source fingerprints must match.")
    if len(alignment.permutation) != target_profile.layout.n_channels:
        raise ValueError("Channel alignment size must match the target profile.")
    if (seed is None) == (validation_plan is None):
        raise ValueError("Provide exactly one of seed or validation_plan.")
    plan = (
        _make_validation_plan(candidate, target_profile, alignment, seed=int(seed), config=config)
        if validation_plan is None
        else validation_plan
    )
    if not isinstance(plan, StimulusPlan):
        raise TypeError("validation_plan must be a StimulusPlan.")
    if plan.layout_fingerprint != target_profile.layout.fingerprint or plan.n_traces != config.n_traces:
        raise ValueError("validation_plan is incompatible with the target profile or validation config.")
    session = teacher_factory(teacher, target_profile.layout, n_traces=config.n_traces)
    targets = np.asarray(session.run(plan).voltage_mv)
    candidate_inputs = np.asarray(rasterize_events(plan, dt_ms=target_profile.dt_ms, input_alignment="target-step"))
    model_inputs = candidate_inputs[:, alignment.permutation, :]
    prediction = candidate.model.predict(model_inputs)
    predictions = np.asarray(prediction["voltage"] if isinstance(prediction, Mapping) else prediction)
    mask = build_fit_mask(
        targets,
        target_profile.dt_ms,
        spike_threshold_mv=config.spike_threshold_mv,
        spike_window_pre_ms=config.spike_window_pre_ms,
        spike_window_post_ms=config.spike_window_post_ms,
    )
    metrics = compute_metrics(predictions, targets, mask)
    ve = float(np.asarray(metrics["variance_explained"]))
    valid_count = int(np.asarray(metrics["valid_count"]))
    accepted = math.isfinite(ve) and ve >= config.variance_explained_threshold
    reason = "accepted" if accepted else ("undefined variance explained" if not math.isfinite(ve) else "below threshold")
    evidence = None
    if isinstance(prediction, Mapping) and "spike" in prediction:
        spike = np.asarray(prediction["spike"])
        if spike.shape != targets.shape or not np.issubdtype(spike.dtype, np.bool_):
            raise ValueError("Reusable DBNN spike output must be Boolean and match teacher voltage shape.")
        raw_spikes = tuple(np.flatnonzero(row).astype(float) * target_profile.dt_ms for row in spike)
        evidence = SpikeAlignmentEvidence(
            teacher_voltage_mv=targets,
            raw_spike_times_ms=raw_spikes,
            dt_ms=target_profile.dt_ms,
            match_window_ms=config.spike_match_window_ms,
            layout_fingerprint=candidate.model.layout.fingerprint,
            dynamics_fingerprint=target_profile.dynamics_fingerprint,
            validation_seeds=plan.seeds,
        )
    return ReuseValidationReport(
        candidate_id=candidate.candidate_id,
        checkpoint_id=str(candidate.provenance.get("checkpoint_id", candidate.candidate_id)),
        alignment_method=alignment.method,
        permutation=alignment.permutation,
        seeds=plan.seeds,
        valid_count=valid_count,
        variance_explained=ve,
        accepted=accepted,
        reason=reason,
        spike_alignment_evidence=evidence,
    )


_DEFAULT_CACHE = _NetworkReuseCache()


def plan_reuse(
    network: Any,
    signature: CellStructuralSignature,
    input_profile: DBNNInputProfile,
    *,
    teacher: TeacherSpec | None = None,
    seed_root: int = 0,
    cache: _NetworkReuseCache = _DEFAULT_CACHE,
    config: ReuseValidationConfig = ReuseValidationConfig(),
    validation_callback: Callable[..., ReuseValidationReport] | None = None,
    fit_fn: Callable[..., _ReuseCandidate] | None = None,
) -> ReusePlan:
    """Try cached candidates in order, then optionally fit one replacement.

    The validation and fitting callables are injectable so orchestration tests
    need not initialize detailed Cells, train models, or compile JAX programs.

    Parameters
    ----------
    network : object
        Live Network identity that scopes the in-memory candidate cache.
    signature : CellStructuralSignature
        Target Cell classification key.
    input_profile : DBNNInputProfile
        Concrete target stimulus and timing profile.
    teacher : TeacherSpec or None, optional
        Detailed target factory required by default validation.
    seed_root : int, optional
        Root for named fit and reuse-validation seed allocations.
    cache : _NetworkReuseCache, optional
        Process-local weak candidate cache.
    config : ReuseValidationConfig, optional
        Measured validation settings.
    validation_callback : callable or None, optional
        Lightweight replacement for :func:`validate_candidate`.
    fit_fn : callable or None, optional
        Fitting callback used after every cached candidate fails.

    Returns
    -------
    ReusePlan
        Selected candidate, input alignment, and complete attempt report.

    Raises
    ------
    LookupError
        If no candidate is accepted and no fitting callback is supplied.
    """
    used_seeds = cache.used_seeds(network)
    seed_state = _new_seed_state(seed_root)
    candidates = cache.candidates(network, signature)
    for item in cache.all_candidates(network):
        _claim_seeds(seed_state, f"fit:{item.candidate_id}", item.fit_seed_roots)
        if item.fit_trace_seeds:
            _claim_seeds(seed_state, f"fit-traces:{item.candidate_id}", item.fit_trace_seeds)
    remaining_used = tuple(seed for seed in used_seeds if seed not in seed_state["owners"])
    if remaining_used:
        _claim_seeds(seed_state, "prior-network-data", remaining_used)
    attempts = []
    for candidate in candidates:
        if candidate.model.layout.fingerprint != candidate.input_profile.layout.fingerprint:
            attempts.append(_incompatible_report(candidate, "model layout changed after caching"))
            continue
        if candidate.input_profile.layout.n_channels != input_profile.layout.n_channels:
            attempts.append(_incompatible_report(candidate, "channel count mismatch"))
            continue
        if candidate.input_profile.dt_ms != input_profile.dt_ms:
            attempts.append(_incompatible_report(candidate, "dt mismatch"))
            continue
        try:
            alignment = align_channels(candidate.input_profile.layout, input_profile.layout)
        except (TypeError, ValueError) as exc:
            attempts.append(_incompatible_report(candidate, str(exc)))
            continue
        try:
            _validation_amplitudes(candidate, input_profile, alignment)
        except ValueError as exc:
            attempts.append(_incompatible_report(candidate, str(exc)))
            continue
        validation_seed = _derive_seeds(seed_state, f"reuse:{candidate.candidate_id}")[0]
        validation_plan = _make_validation_plan(
            candidate,
            input_profile,
            alignment,
            seed=validation_seed,
            config=config,
        )
        _claim_seeds(seed_state, f"reuse-traces:{candidate.candidate_id}", validation_plan.seeds)
        cache.reserve_seeds(network, (validation_seed, *validation_plan.seeds))
        callback = validate_candidate if validation_callback is None else validation_callback
        if callback is validate_candidate:
            if teacher is None:
                raise TypeError("teacher is required when using the default validation implementation.")
            report = callback(
                candidate,
                input_profile,
                alignment,
                validation_plan=validation_plan,
                teacher=teacher,
                config=config,
            )
        else:
            report = callback(
                candidate,
                input_profile,
                alignment,
                validation_plan=validation_plan,
                config=config,
            )
        if not isinstance(report, ReuseValidationReport):
            raise TypeError("validation_callback must return a ReuseValidationReport.")
        _validate_validation_report(report, candidate, alignment, validation_plan)
        measured_acceptance = (
            math.isfinite(report.variance_explained)
            and report.variance_explained >= config.variance_explained_threshold
        )
        if report.accepted and not measured_acceptance:
            reason = (
                "undefined variance explained"
                if not math.isfinite(report.variance_explained)
                else "below threshold"
            )
            report = replace(report, accepted=False, reason=reason)
        attempts.append(report)
        if report.accepted:
            return ReusePlan(candidate, alignment, tuple(attempts), False)

    if fit_fn is None:
        raise LookupError("No compatible DBNN reuse candidate passed validation and no fit_fn was supplied.")
    fit_roots = _derive_seeds(seed_state, "fit:new", 3)
    cache.reserve_seeds(network, fit_roots)
    fitted = fit_fn(
        signature=signature,
        input_profile=input_profile,
        fit_seed_roots=fit_roots,
        forbidden_seeds=_allocated_seeds(seed_state),
        reserve_seeds=lambda values: cache.reserve_seeds(network, values),
    )
    if not isinstance(fitted, _ReuseCandidate):
        raise TypeError("fit_fn must return a _ReuseCandidate.")
    if fitted.fit_seed_roots != fit_roots:
        raise ValueError("fit_fn must record the supplied fit_seed_roots on its ReuseCandidate.")
    if fitted.fit_trace_seeds:
        _claim_seeds(seed_state, "fit:new:traces", fitted.fit_trace_seeds)
    if fitted.signature.key != signature.key:
        raise ValueError("fit_fn returned a candidate for a different Cell signature.")
    if not _profiles_equal(fitted.input_profile, input_profile):
        raise ValueError("fit_fn returned a candidate incompatible with the target input profile.")
    alignment = align_channels(fitted.input_profile.layout, input_profile.layout)
    cache.add(network, fitted)
    return ReusePlan(fitted, alignment, tuple(attempts), True)


def _validation_amplitudes(
    candidate: _ReuseCandidate,
    target_profile: DBNNInputProfile,
    alignment: ChannelAlignment,
) -> np.ndarray:
    amplitudes = np.ones(target_profile.layout.n_channels, dtype=float)
    rates = np.asarray(target_profile.rate_hz)
    for model_channel, target_channel in enumerate(alignment.permutation):
        if rates[target_channel] == 0:
            continue
        target_low, target_high = target_profile.layout.training_range(target_channel)
        model_low, model_high = candidate.input_profile.layout.training_range(model_channel)
        low = max(target_low, model_low)
        high = min(target_high, model_high)
        if high <= 0 or low > high:
            raise ValueError(
                f"Target channel {target_channel} and model channel {model_channel} "
                "have no shared positive validation-amplitude range."
            )
        amplitude = max(low, min(1.0, high))
        if amplitude == 0:
            amplitude = high / 2.0
        amplitudes[target_channel] = amplitude
    return amplitudes


def _profiles_equal(left: DBNNInputProfile, right: DBNNInputProfile) -> bool:
    return (
        left.layout.fingerprint == right.layout.fingerprint
        and left.rate_hz == right.rate_hz
        and left.dt_ms == right.dt_ms
        and left.validation_duration_ms == right.validation_duration_ms
        and left.source_fingerprint == right.source_fingerprint
        and left.dynamics_fingerprint == right.dynamics_fingerprint
    )


def _make_validation_plan(
    candidate: _ReuseCandidate,
    target_profile: DBNNInputProfile,
    alignment: ChannelAlignment,
    *,
    seed: int,
    config: ReuseValidationConfig,
) -> StimulusPlan:
    return generate_multichannel_protocol(
        target_profile.layout,
        n_traces=config.n_traces,
        duration_ms=target_profile.validation_duration_ms,
        dt_ms=target_profile.dt_ms,
        rate_hz=np.asarray(target_profile.rate_hz),
        amplitude=_validation_amplitudes(candidate, target_profile, alignment),
        seed=seed,
        split=f"reuse-validation:{candidate.candidate_id}",
        ensure_channel_coverage=False,
    )


def _validate_validation_report(
    report: ReuseValidationReport,
    candidate: _ReuseCandidate,
    alignment: ChannelAlignment,
    validation_plan: StimulusPlan,
) -> None:
    expected_checkpoint = str(candidate.provenance.get("checkpoint_id", candidate.candidate_id))
    if report.candidate_id != candidate.candidate_id or report.checkpoint_id != expected_checkpoint:
        raise ValueError("Validation report candidate or checkpoint identity is incorrect.")
    if report.alignment_method != alignment.method or report.permutation != alignment.permutation:
        raise ValueError("Validation report channel alignment is incorrect.")
    if report.seeds != tuple(validation_plan.seeds):
        raise ValueError("Validation report seeds do not match the reserved validation plan.")


def _incompatible_report(candidate: _ReuseCandidate, reason: str) -> ReuseValidationReport:
    return ReuseValidationReport(
        candidate_id=candidate.candidate_id,
        checkpoint_id=str(candidate.provenance.get("checkpoint_id", candidate.candidate_id)),
        alignment_method=None,
        permutation=(),
        seeds=(),
        valid_count=0,
        variance_explained=float("nan"),
        accepted=False,
        reason=reason,
    )


__all__ = [
    "DEFAULT_REUSE_VALIDATION_TRACES",
    "DEFAULT_REUSE_VE_THRESHOLD",
    "DBNNInputProfile",
    "ReusePlan",
    "ReuseValidationConfig",
    "ReuseValidationReport",
    "plan_reuse",
    "validate_candidate",
]
