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

"""Tests for process-local DBNN checkpoint reuse planning."""

from dataclasses import FrozenInstanceError
import gc
import hashlib
import json
import unittest
import weakref

import brainunit as u
import numpy as np

from braincell.mech import Synapse
from braincell.reduction.dbnn.dataset import DatasetBatch
from braincell.reduction.dbnn.layout import ChannelAlignment, ChannelLayout
from braincell.reduction.dbnn.reuse import (
    DBNNInputProfile,
    ReusePlan,
    ReuseValidationConfig,
    ReuseValidationReport,
    _NetworkReuseCache,
    _ReuseCandidate,
    _claim_seeds,
    _derive_seeds,
    _new_seed_state,
    plan_reuse,
    validate_candidate,
)
from braincell.reduction.dbnn.signature import CellStructuralSignature


def _layout(ids=(0, 1)):
    return ChannelLayout(
        (Synapse("ExpSyn", name="E"),),
        tuple(ids),
        tuple(range(len(ids))),
        tuple(range(len(ids))),
        (0,) * len(ids),
    )


def _profile(*, layout=None, rates=20.0, dt=1.0 * u.ms, duration=4.0 * u.ms, source="target"):
    return DBNNInputProfile(layout or _layout(), rates, dt, duration, source)


def _signature(name="cell"):
    payload = json.dumps({"cell_type": name}, sort_keys=True, separators=(",", ":"))
    return CellStructuralSignature(name, payload, hashlib.sha256(payload.encode()).hexdigest())


class _Network:
    pass


class _EqualNetwork:
    __hash__ = None

    def __eq__(self, other):
        return isinstance(other, _EqualNetwork)


class _Model:
    def __init__(self, dt_ms=1.0, prediction=None, layout=None):
        self.dt_ms = dt_ms
        self.prediction = prediction
        self.last_inputs = None
        self.layout = layout or _layout()

    def predict(self, inputs):
        self.last_inputs = np.asarray(inputs)
        voltage = self.prediction(self.last_inputs) if callable(self.prediction) else self.prediction
        return {"voltage": voltage}


def _candidate(candidate_id="checkpoint", *, profile=None, model=None, roots=(11, 12, 13), trace_seeds=()):
    profile = profile or _profile(source="fit-source")
    return _ReuseCandidate(
        candidate_id,
        model or _Model(profile.dt_ms, layout=profile.layout),
        _signature(),
        profile,
        {"checkpoint_id": f"asset-{candidate_id}"},
        roots,
        trace_seeds,
    )


class InputProfileTest(unittest.TestCase):
    def test_normalizes_scalar_and_per_channel_rates(self):
        self.assertEqual(_profile(rates=3.0).rate_hz, (3.0, 3.0))
        self.assertEqual(_profile(rates=[1.0, 2.0]).rate_hz, (1.0, 2.0))

    def test_is_immutable_and_exposes_converted_timing(self):
        profile = _profile(dt=0.001 * u.second, duration=0.004 * u.second)
        self.assertEqual((profile.dt_ms, profile.validation_duration_ms), (1.0, 4.0))
        with self.assertRaises(FrozenInstanceError):
            profile.source_fingerprint = "changed"

    def test_rejects_invalid_layout_rate_units_duration_and_source(self):
        with self.assertRaises(TypeError):
            DBNNInputProfile(object(), 1.0, 1.0 * u.ms, 2.0 * u.ms, "x")
        for rates in ((1.0,), (1.0, np.nan), (-1.0, 2.0)):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                _profile(rates=rates)
        with self.assertRaises(TypeError):
            _profile(dt=1.0)
        with self.assertRaises(ValueError):
            _profile(duration=2.5 * u.ms)
        with self.assertRaises(ValueError):
            _profile(source=" ")
        with self.assertRaisesRegex(ValueError, "probabilities"):
            _profile(rates=1001.0)


class SeedAllocationTest(unittest.TestCase):
    def test_named_derivation_is_deterministic_and_disjoint(self):
        first = _new_seed_state(7, {"fit": (1, 2, 3)})
        second = _new_seed_state(7, {"fit": (1, 2, 3)})
        self.assertEqual(_derive_seeds(first, "candidate-a", 8), _derive_seeds(second, "candidate-a", 8))
        other = _derive_seeds(first, "candidate-b", 8)
        self.assertTrue(set(other).isdisjoint(first["by_name"]["candidate-a"]))
        self.assertTrue(set(other).isdisjoint(first["by_name"]["fit"]))

    def test_rejects_name_and_seed_overlaps(self):
        state = _new_seed_state(0)
        _claim_seeds(state, "data", (4, 5))
        with self.assertRaisesRegex(ValueError, "already present"):
            _claim_seeds(state, "data", (6,))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            _claim_seeds(state, "split", (5,))
        with self.assertRaisesRegex(ValueError, "contains an overlap"):
            _claim_seeds(state, "training", (7, 7))


class CacheTest(unittest.TestCase):
    def test_cache_is_network_and_signature_scoped(self):
        cache = _NetworkReuseCache()
        first, second = _Network(), _Network()
        candidate = _candidate()
        cache.add(first, candidate)
        self.assertEqual(cache.candidates(first, _signature()), (candidate,))
        self.assertEqual(cache.candidates(second, _signature()), ())
        self.assertEqual(cache.candidates(first, _signature("other")), ())

    def test_network_entry_is_weak(self):
        cache = _NetworkReuseCache()
        network = _Network()
        cache.add(network, _candidate())
        reference = weakref.ref(network)
        del network
        gc.collect()
        self.assertIsNone(reference())

    def test_cache_uses_identity_not_equality_or_hashing(self):
        cache = _NetworkReuseCache()
        first, second = _EqualNetwork(), _EqualNetwork()
        cache.add(first, _candidate())
        self.assertEqual(cache.candidates(second, _signature()), ())

    def test_validation_seeds_are_reserved_across_planning_calls(self):
        cache = _NetworkReuseCache()
        network = _Network()
        candidate = _candidate()
        cache.add(network, candidate)
        seen = []

        def validate(candidate, profile, alignment, **kwargs):
            plan = kwargs["validation_plan"]
            seen.append(plan.seeds[0])
            return ReuseValidationReport(
                candidate.candidate_id,
                candidate.provenance["checkpoint_id"],
                alignment.method,
                alignment.permutation,
                plan.seeds,
                10,
                0.9,
                True,
                "accepted",
            )

        for _ in range(2):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                seed_root=44,
                validation_callback=validate,
            )
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])


class ValidationTest(unittest.TestCase):
    def _validate(self, ve, *, threshold=0.8, constant=False):
        profile = _profile(rates=0.0)
        n_times = int(profile.validation_duration_ms / profile.dt_ms) + 1
        targets = np.zeros((8, n_times)) if constant else np.tile(np.arange(n_times), (8, 1)).astype(float)
        target_ss = np.sum(np.square(targets - np.mean(targets)))
        error = 0.0 if ve == 1.0 else np.sqrt((1.0 - ve) * target_ss / targets.size)
        predictions = targets + error
        model = _Model(prediction=predictions, layout=profile.layout)
        candidate = _candidate(profile=profile, model=model)

        class Session:
            def __init__(self, teacher, layout, *, n_traces):
                self.n_traces = n_traces

            def run(self, plan):
                return DatasetBatch(
                    plan,
                    targets,
                    tuple(np.array([]) for _ in range(plan.n_traces)),
                    np.arange(n_times),
                    {},
                )

        return validate_candidate(
            candidate,
            profile,
            ChannelAlignment("branch_position", (0, 1)),
            seed=99,
            teacher=object(),
            config=ReuseValidationConfig(
                variance_explained_threshold=threshold,
                spike_threshold_mv=100.0,
                spike_window_pre_ms=0.0,
                spike_window_post_ms=0.0,
            ),
            teacher_factory=Session,
        )

    def test_accepts_ve_at_and_above_threshold_and_rejects_below(self):
        for ve, accepted in ((0.799, False), (0.8, True), (0.9, True)):
            with self.subTest(ve=ve):
                report = self._validate(ve)
                self.assertEqual(report.accepted, accepted)
                self.assertAlmostEqual(report.variance_explained, ve, places=5)
                self.assertEqual(len(report.seeds), 8)

    def test_rejects_undefined_ve_and_reports_valid_count(self):
        report = self._validate(0.8, constant=True)
        self.assertFalse(report.accepted)
        self.assertTrue(np.isnan(report.variance_explained))
        self.assertEqual(report.valid_count, 40)
        self.assertEqual(report.reason, "undefined variance explained")

    def test_applies_candidate_to_model_permutation_on_target_step_raster(self):
        profile = _profile(rates=(1000.0, 1000.0))
        model = _Model(prediction=np.zeros((8, 5)), layout=profile.layout)
        candidate = _candidate(profile=profile, model=model)

        class Session:
            def __init__(self, teacher, layout, *, n_traces):
                pass

            def run(self, plan):
                return DatasetBatch(plan, np.zeros((8, 5)), tuple(np.array([]) for _ in range(8)), np.arange(5), {})

        report = validate_candidate(
            candidate,
            profile,
            ChannelAlignment("identifier_distance", (1, 0)),
            seed=1,
            teacher=object(),
            config=ReuseValidationConfig(spike_threshold_mv=100.0),
            teacher_factory=Session,
        )
        # Probability one gives both channels events without distorting profile coverage.
        self.assertEqual(report.permutation, (1, 0))
        self.assertEqual(model.last_inputs.shape, (8, 2, 5))
        self.assertTrue(np.any(model.last_inputs[:, 0, 1:]))

    def test_validation_retains_target_voltage_and_raw_model_spikes(self):
        profile = _profile(rates=0.0)
        targets = np.tile(np.asarray([-70.0, -60.0, -50.0, -40.0, -30.0]), (8, 1))

        class SpikeModel(_Model):
            def predict(self, inputs):
                self.last_inputs = np.asarray(inputs)
                spike = np.zeros(targets.shape, dtype=bool)
                spike[:, 2] = True
                return {"voltage": targets, "spike": spike}

        candidate = _candidate(profile=profile, model=SpikeModel(layout=profile.layout))

        class Session:
            def __init__(self, teacher, layout, *, n_traces):
                pass

            def run(self, plan):
                return DatasetBatch(
                    plan,
                    targets,
                    tuple(np.array([]) for _ in range(8)),
                    np.arange(5),
                    {},
                )

        report = validate_candidate(
            candidate,
            profile,
            ChannelAlignment("identity", (0, 1)),
            seed=4,
            teacher=object(),
            config=ReuseValidationConfig(spike_threshold_mv=100.0),
            teacher_factory=Session,
        )

        evidence = report.spike_alignment_evidence
        np.testing.assert_array_equal(evidence.teacher_voltage_mv, targets)
        np.testing.assert_array_equal(evidence.raw_spike_times_ms[0], [2.0])
        self.assertEqual(evidence.validation_seeds, report.seeds)

    def test_validation_amplitude_uses_target_and_checkpoint_range_intersection(self):
        target_layout = _layout()
        target_profile = _profile(layout=target_layout, rates=1000.0)
        model_layout = ChannelLayout(
            target_layout.specs,
            target_layout.synapse_ids,
            target_layout.placement_ids,
            target_layout.point_ids,
            target_layout.spec_indices,
            target_layout.reference_weights_us,
            ((0.25, 0.5),),
        )
        model_profile = _profile(layout=model_layout, rates=1000.0, source="fit")
        model = _Model(prediction=np.zeros((8, 5)), layout=model_layout)
        candidate = _candidate(profile=model_profile, model=model)

        class Session:
            def __init__(self, teacher, layout, *, n_traces):
                pass

            def run(self, plan):
                return DatasetBatch(plan, np.zeros((8, 5)), tuple(np.array([]) for _ in range(8)), np.arange(5), {})

        validate_candidate(
            candidate,
            target_profile,
            ChannelAlignment("branch_position", (0, 1)),
            seed=3,
            teacher=object(),
            config=ReuseValidationConfig(spike_threshold_mv=100.0),
            teacher_factory=Session,
        )
        self.assertLessEqual(float(np.max(model.last_inputs)), 0.5)


class PlanningTest(unittest.TestCase):
    def test_incompatible_amplitude_candidate_does_not_block_later_candidate(self):
        cache, network = _NetworkReuseCache(), _Network()
        target = _profile(rates=1000.0)
        base_layout = _layout()
        incompatible_layout = ChannelLayout(
            base_layout.specs,
            base_layout.synapse_ids,
            base_layout.placement_ids,
            base_layout.point_ids,
            base_layout.spec_indices,
            base_layout.reference_weights_us,
            ((2.0, 3.0),),
        )
        cache.add(network, _candidate("outside", profile=_profile(layout=incompatible_layout)))
        cache.add(network, _candidate("compatible", roots=(21, 22, 23)))

        def accept(candidate, profile, alignment, **kwargs):
            plan = kwargs["validation_plan"]
            return ReuseValidationReport(
                candidate.candidate_id,
                candidate.provenance["checkpoint_id"],
                alignment.method,
                alignment.permutation,
                plan.seeds,
                10,
                0.9,
                True,
                "accepted",
            )

        result = plan_reuse(
            network,
            _signature(),
            target,
            cache=cache,
            validation_callback=accept,
        )
        self.assertEqual(result.candidate.candidate_id, "compatible")
        self.assertIn("no shared positive", result.attempts[0].reason)

    def test_tries_candidates_reports_attempts_and_stops_on_acceptance(self):
        cache, network = _NetworkReuseCache(), _Network()
        first, second = _candidate("first"), _candidate("second", roots=(21, 22, 23))
        cache.add(network, first)
        cache.add(network, second)
        seen_seeds = []

        def validate(candidate, profile, alignment, *, validation_plan, config):
            seen_seeds.append(validation_plan.seeds[0])
            accepted = candidate.candidate_id == "second"
            return ReuseValidationReport(
                candidate.candidate_id,
                candidate.provenance["checkpoint_id"],
                alignment.method,
                alignment.permutation,
                validation_plan.seeds,
                10,
                0.9 if accepted else 0.5,
                accepted,
                "accepted" if accepted else "below threshold",
            )

        result = plan_reuse(
            network,
            _signature(),
            _profile(),
            cache=cache,
            seed_root=44,
            validation_callback=validate,
        )
        self.assertIsInstance(result, ReusePlan)
        self.assertEqual(result.candidate.candidate_id, "second")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(len(set(seen_seeds)), 2)
        self.assertTrue(set(seen_seeds).isdisjoint(first.fit_seed_roots + second.fit_seed_roots))

    def test_reports_incompatible_candidate_then_fits_and_caches(self):
        cache, network = _NetworkReuseCache(), _Network()
        incompatible_profile = _profile(dt=2.0 * u.ms, duration=4.0 * u.ms)
        incompatible = _candidate(
            profile=incompatible_profile,
            model=_Model(2.0, layout=incompatible_profile.layout),
        )
        cache.add(network, incompatible)

        def fit(**kwargs):
            return _candidate(
                "new",
                profile=kwargs["input_profile"],
                roots=kwargs["fit_seed_roots"],
            )

        result = plan_reuse(
            network,
            _signature(),
            _profile(),
            cache=cache,
            fit_fn=fit,
        )
        self.assertTrue(result.fitted)
        self.assertEqual(result.attempts[0].reason, "dt mismatch")
        self.assertEqual(len(cache.candidates(network, _signature())), 2)

    def test_errors_without_accepted_candidate_or_fit(self):
        with self.assertRaises(LookupError):
            plan_reuse(_Network(), _signature(), _profile(), cache=_NetworkReuseCache())

    def test_callback_cannot_accept_below_threshold_ve(self):
        cache, network = _NetworkReuseCache(), _Network()
        cache.add(network, _candidate())

        def invalid_acceptance(candidate, profile, alignment, **kwargs):
            plan = kwargs["validation_plan"]
            return ReuseValidationReport(
                candidate.candidate_id,
                candidate.provenance["checkpoint_id"],
                alignment.method,
                alignment.permutation,
                plan.seeds,
                10,
                0.79,
                True,
                "accepted",
            )

        with self.assertRaises(LookupError):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                validation_callback=invalid_acceptance,
            )

    def test_new_fit_trace_seeds_cannot_overlap_failed_reuse_validation(self):
        cache, network = _NetworkReuseCache(), _Network()
        cache.add(network, _candidate())
        validation_seed = []

        def reject(candidate, profile, alignment, **kwargs):
            plan = kwargs["validation_plan"]
            validation_seed.append(plan.seeds[0])
            return ReuseValidationReport(
                candidate.candidate_id,
                candidate.provenance["checkpoint_id"],
                alignment.method,
                alignment.permutation,
                plan.seeds,
                10,
                0.5,
                False,
                "below threshold",
            )

        def fit(**kwargs):
            return _candidate(
                "new",
                roots=kwargs["fit_seed_roots"],
                trace_seeds=(validation_seed[0],),
            )

        with self.assertRaisesRegex(ValueError, "overlaps"):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                validation_callback=reject,
                fit_fn=fit,
            )

    def test_report_dataclasses_are_immutable(self):
        permutation = []
        seeds = []
        report = ReuseValidationReport("a", "b", None, permutation, seeds, 0, np.nan, False, "no")
        permutation.append(1)
        seeds.append(2)
        self.assertEqual(report.permutation, ())
        self.assertEqual(report.seeds, ())
        with self.assertRaises(FrozenInstanceError):
            report.accepted = True

    def test_callback_report_must_match_reserved_candidate_plan_and_alignment(self):
        cache, network = _NetworkReuseCache(), _Network()
        cache.add(network, _candidate())

        def wrong(candidate, profile, alignment, **kwargs):
            plan = kwargs["validation_plan"]
            return ReuseValidationReport(
                "wrong",
                candidate.provenance["checkpoint_id"],
                alignment.method,
                alignment.permutation,
                plan.seeds,
                10,
                0.9,
                True,
                "accepted",
            )

        with self.assertRaisesRegex(ValueError, "identity"):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                validation_callback=wrong,
            )

    def test_validation_seeds_remain_reserved_when_callback_fails(self):
        cache, network = _NetworkReuseCache(), _Network()
        cache.add(network, _candidate())

        def fail(*args, **kwargs):
            raise RuntimeError("validation failed")

        with self.assertRaisesRegex(RuntimeError, "validation failed"):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                validation_callback=fail,
            )
        self.assertEqual(len(cache.used_seeds(network)), 9)

    def test_fit_roots_remain_reserved_when_fitting_fails(self):
        cache, network = _NetworkReuseCache(), _Network()
        captured = []

        def fail_fit(**kwargs):
            captured.extend(kwargs["fit_seed_roots"])
            raise RuntimeError("fit failed")

        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                fit_fn=fail_fit,
            )
        self.assertTrue(set(captured).issubset(cache.used_seeds(network)))

    def test_fitted_candidate_must_match_complete_target_profile(self):
        cache, network = _NetworkReuseCache(), _Network()

        def fit(**kwargs):
            return _candidate(
                "wrong-source",
                profile=_profile(source="different"),
                roots=kwargs["fit_seed_roots"],
            )

        with self.assertRaisesRegex(ValueError, "target input profile"):
            plan_reuse(
                network,
                _signature(),
                _profile(),
                cache=cache,
                fit_fn=fit,
            )
        self.assertEqual(cache.candidates(network, _signature()), ())


if __name__ == "__main__":
    unittest.main()
