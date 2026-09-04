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

"""Execute composable exact-gradient and derivative-free optimization stages."""

from __future__ import annotations

from dataclasses import dataclass
import time

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from examples.experimental.optim.gradients import (
    build_rollout_value_and_grad,
    build_trajectory_value_and_grad,
)
from examples.experimental.optim_parameter_fitting.config import ExperimentConfig
from examples.experimental.optim_parameter_fitting.datasets import DatasetBundle
from examples.experimental.optim_parameter_fitting.search import CandidateSet, SearchStageResult


@dataclass(frozen=True)
class SplitMetrics:
    """Store candidate-leading loss, protocol loss, and hard spike counts."""

    total_loss: np.ndarray
    raw_total_mse: np.ndarray
    protocol_loss: np.ndarray
    spike_counts: np.ndarray


@dataclass(frozen=True)
class GradientStageResult:
    """Store one complete gradient-stage state and update history."""

    name: str
    input_candidates: CandidateSet
    output_candidates: CandidateSet
    state_epoch: np.ndarray
    optimizer_z: np.ndarray
    physical_parameters: np.ndarray
    train_loss: np.ndarray
    validation_epoch: np.ndarray
    validation: SplitMetrics
    test_epoch: np.ndarray
    test: SplitMetrics
    gradient: np.ndarray
    gradient_norm: np.ndarray
    gradient_seconds: np.ndarray
    update_seconds: np.ndarray
    stage_seconds: float
    compile_seconds: float
    memory: dict[str, int]
    final_train: SplitMetrics
    forward_evaluations: int


@dataclass(frozen=True)
class PipelineResult:
    """Store ordered stage results and the final physical candidate set."""

    initial_candidates: CandidateSet
    final_candidates: CandidateSet
    stages: tuple[object, ...]
    dataset: DatasetBundle


class GradientEngine:
    """Compile one exact full-batch gradient kernel for all candidate lanes."""

    def __init__(self, config: ExperimentConfig, current_na, target_voltage_mv, *, method: str) -> None:
        self.config = config
        self.current = np.asarray(current_na, dtype=np.float64)
        self.target = jnp.asarray(target_voltage_mv)
        self.method = method
        self.cell = config.model.build_cell(self.current, trainable=True)
        self.times_ms = jnp.arange(self.target.shape[1], dtype=jnp.float64) * config.dataset.dt_ms
        num_steps = int(self.target.shape[1])
        protocol_weights = config.loss.prepare(self.target)

        def rollout_step(data):
            time_ms, target_mv = data
            voltage = self.cell.V.value.to_decimal(u.mV)
            local_loss = config.loss.local(voltage, target_mv, num_steps=num_steps, protocol_weights=protocol_weights)
            with brainstate.environ.context(t=time_ms * u.ms):
                self.cell.update()
            return local_loss

        engine = build_rollout_value_and_grad(self.cell, step=rollout_step, method=method)
        engine.prepare((self.times_ms[0], self.target[:, 0]))
        expected = config.model.parameter_space.names
        if engine.parameter_names != expected:
            raise RuntimeError(f"Unexpected parameter order {engine.parameter_names!r}, expected {expected!r}.")
        step_data = (self.times_ms, jnp.moveaxis(self.target, 0, 1))

        def one_candidate(z_roots):
            result = engine._rtrl(z_roots, step_data) if method == "rtrl" else engine._bptt(z_roots, step_data)
            gradient = jnp.stack(tuple(result.gradients[name] for name in expected))
            return result.loss, gradient

        self._function = jax.jit(jax.vmap(one_candidate))
        self._compiled = None

    def compile(self, z_roots) -> tuple[float, dict[str, int]]:
        """Compile once and return XLA memory accounting."""
        started = time.perf_counter()
        self._compiled = self._function.lower(z_roots).compile()
        compile_seconds = time.perf_counter() - started
        memory = self._compiled.memory_analysis()
        return compile_seconds, {
            "argument_bytes": int(memory.argument_size_in_bytes),
            "output_bytes": int(memory.output_size_in_bytes),
            "temporary_bytes": int(memory.temp_size_in_bytes),
            "alias_bytes": int(memory.alias_size_in_bytes),
        }

    def __call__(self, z_roots):
        """Return candidate loss and a dense ``(candidate,parameter)`` gradient."""
        function = self._compiled if self._compiled is not None else self._function
        loss, gradient = function(z_roots)
        _block_until_ready((loss, gradient))
        return loss, gradient


class ForwardEvaluator:
    """Compile forward-only voltage, loss, and spike metrics for one split."""

    def __init__(self, config: ExperimentConfig, current_na, target_voltage_mv) -> None:
        current = np.asarray(current_na, dtype=np.float64)
        target = jnp.asarray(target_voltage_mv)
        self.config = config
        self.target = target
        self.cell = config.model.build_cell(current, trainable=True)
        self.times_ms = jnp.arange(target.shape[1], dtype=jnp.float64) * config.dataset.dt_ms

        def observation_step(time_ms):
            voltage = self.cell.V.value.to_decimal(u.mV)
            with brainstate.environ.context(t=time_ms * u.ms):
                self.cell.update()
            return voltage

        engine = build_trajectory_value_and_grad(
            self.cell,
            step=observation_step,
            loss=lambda observation, _time: jnp.mean(observation * 0.0),
            method="rtrl",
        )
        engine.prepare(self.times_ms[0])
        expected = config.model.parameter_space.names
        if engine.parameter_names != expected:
            raise RuntimeError(f"Unexpected parameter order {engine.parameter_names!r}, expected {expected!r}.")

        def one_candidate(z_roots):
            time_leading = engine._observation_rollout(z_roots, self.times_ms)
            return jnp.moveaxis(time_leading, 0, 1)

        self._prediction = jax.jit(jax.vmap(one_candidate))

    def evaluate(self, physical) -> SplitMetrics:
        """Evaluate all candidate rows without differentiating the objective."""
        z_roots = self.config.model.parameter_space.z_roots(jnp.asarray(physical))
        prediction = self._prediction(z_roots)
        prediction.block_until_ready()
        total, raw, per_protocol = self.config.loss.evaluate(prediction, self.target)
        soma = prediction[..., 0]
        counts = jnp.sum((soma[..., :-1] < 0.0) & (soma[..., 1:] >= 0.0), axis=-1)
        _block_until_ready((total, per_protocol, counts))
        return SplitMetrics(np.asarray(total), np.asarray(raw), np.asarray(per_protocol), np.asarray(counts))

    def traces(self, physical) -> np.ndarray:
        """Return complete voltage traces for selected candidate rows."""
        z_roots = self.config.model.parameter_space.z_roots(jnp.asarray(physical))
        prediction = self._prediction(z_roots)
        prediction.block_until_ready()
        return np.asarray(prediction)


@dataclass(frozen=True)
class StageContext:
    """Expose bounded coordinates and train-only forward evaluation to search stages."""

    parameter_space: object
    train_evaluator: ForwardEvaluator


def initialize_candidates(config: ExperimentConfig) -> CandidateSet:
    """Draw all initial physical parameters once, independent of execution layout."""
    initialization = config.initialization
    random = brainstate.random.RandomState(initialization.seed)
    normalized = np.asarray(
        random.uniform(
            np.finfo(np.float64).eps,
            1.0 - np.finfo(np.float64).eps,
            size=(initialization.num_candidates, config.model.parameter_space.size),
        ),
        dtype=np.float64,
    )
    if initialization.target_relative_range is None:
        physical = np.asarray(config.model.parameter_space.normalized_to_physical(normalized), dtype=np.float64)
    else:
        lower_factor, upper_factor = initialization.target_relative_range
        target = jnp.asarray(config.model.parameter_space.target, dtype=jnp.float64)
        lower = lower_factor * target
        upper = upper_factor * target
        bounds_lower = jnp.asarray(config.model.parameter_space.lower)
        bounds_upper = jnp.asarray(config.model.parameter_space.upper)
        if bool(jnp.any(lower <= bounds_lower)) or bool(jnp.any(upper >= bounds_upper)):
            raise ValueError("Initialization support must lie strictly inside transform bounds.")
        physical = np.asarray(lower + (upper - lower) * jnp.asarray(normalized), dtype=np.float64)
    return CandidateSet(
        physical=physical,
        candidate_id=np.arange(initialization.num_candidates, dtype=np.int32),
        provenance=tuple("initialization" for _ in range(initialization.num_candidates)),
    )


def run_pipeline(config: ExperimentConfig, dataset: DatasetBundle) -> PipelineResult:
    """Execute the ordered optimization stages without implicit candidate chunking."""
    candidates = initialize_candidates(config)
    initial = candidates
    split_data = {split: dataset.subset(split) for split in ("train", "validation", "test")}
    evaluators = {
        split: ForwardEvaluator(config, current, target) for split, (current, target, _protocols) in split_data.items()
    }
    stage_results = []
    for stage in config.stages:
        if getattr(stage, "kind", None) == "gradient" and callable(getattr(stage, "build_optimizer", None)):
            result = run_gradient_stage(config, stage, candidates, split_data, evaluators)
            candidates = result.output_candidates
        elif getattr(stage, "kind", None) == "derivative_free" and callable(getattr(stage, "run", None)):
            result = stage.run(StageContext(config.model.parameter_space, evaluators["train"]), candidates)
            if not isinstance(result, SearchStageResult):
                raise TypeError("A derivative-free stage must return SearchStageResult.")
            candidates = result.candidates
        else:
            raise TypeError(f"Unsupported optimization stage {stage!r}.")
        stage_results.append(result)
    return PipelineResult(initial, candidates, tuple(stage_results), dataset)


def run_gradient_stage(
    config: ExperimentConfig,
    stage,
    candidates: CandidateSet,
    split_data,
    evaluators: dict[str, ForwardEvaluator],
) -> GradientStageResult:
    """Run one fresh-state gradient stage on every candidate in one compiled kernel."""
    train_current, train_target, _train_protocols = split_data["train"]
    engine = GradientEngine(config, train_current, train_target, method=stage.gradient_method)
    initial_z = np.asarray(config.model.parameter_space.physical_to_z(candidates.physical), dtype=np.float64)
    roots = tuple(brainstate.ParamState(jnp.asarray(initial_z[:, index])) for index in range(initial_z.shape[1]))
    parameter_states = {name: state for name, state in zip(config.model.parameter_space.names, roots)}
    optimizer = stage.build_optimizer(parameter_states)
    compile_seconds, memory = engine.compile(tuple(state.value for state in roots))

    state_epoch = [0]
    optimizer_history = [initial_z]
    physical_history = [np.asarray(candidates.physical)]
    train_loss = []
    gradients = []
    gradient_norm = []
    gradient_seconds = []
    update_seconds = []
    validation_epoch = [0]
    validation_history = [evaluators["validation"].evaluate(candidates.physical)]

    stage_started = time.perf_counter()
    for epoch in range(1, stage.epochs + 1):
        z_roots = tuple(state.value for state in roots)
        started = time.perf_counter()
        loss, gradient = engine(z_roots)
        gradient_seconds.append(time.perf_counter() - started)
        train_loss.append(np.asarray(loss))
        gradients.append(np.asarray(gradient))
        gradient_norm.append(np.linalg.norm(np.asarray(gradient), axis=1))
        mapping = {name: gradient[:, index] for index, name in enumerate(config.model.parameter_space.names)}
        started = time.perf_counter()
        optimizer.update(mapping)
        _block_until_ready(tuple(state.value for state in roots))
        update_seconds.append(time.perf_counter() - started)
        z = np.asarray(jnp.stack(tuple(state.value for state in roots), axis=1))
        physical = np.asarray(config.model.parameter_space.z_to_physical(z))
        state_epoch.append(epoch)
        optimizer_history.append(z)
        physical_history.append(physical)
        if epoch % stage.validation_every == 0:
            validation_epoch.append(epoch)
            validation_history.append(evaluators["validation"].evaluate(physical))

    final_physical = physical_history[-1]
    final_train = evaluators["train"].evaluate(final_physical)
    train_state_loss = np.concatenate((np.stack(train_loss), final_train.total_loss[None]), axis=0)
    final_validation = validation_history[-1]
    if validation_epoch[-1] != stage.epochs:
        validation_epoch.append(stage.epochs)
        final_validation = evaluators["validation"].evaluate(final_physical)
        validation_history.append(final_validation)
    final_test = evaluators["test"].evaluate(final_physical)
    stage_seconds = time.perf_counter() - stage_started
    output = candidates.replace(final_physical, stage_name=stage.name)
    validation = _stack_split_metrics(validation_history)
    forward_evaluations = (
        len(validation_history) * candidates.size * len(split_data["validation"][2])
        + candidates.size * len(split_data["train"][2])
        + candidates.size * len(split_data["test"][2])
    )
    return GradientStageResult(
        name=stage.name,
        input_candidates=candidates,
        output_candidates=output,
        state_epoch=np.asarray(state_epoch, dtype=np.int32),
        optimizer_z=np.stack(optimizer_history),
        physical_parameters=np.stack(physical_history),
        train_loss=train_state_loss,
        validation_epoch=np.asarray(validation_epoch, dtype=np.int32),
        validation=validation,
        test_epoch=np.asarray([stage.epochs], dtype=np.int32),
        test=_stack_split_metrics([final_test]),
        gradient=np.stack(gradients),
        gradient_norm=np.stack(gradient_norm),
        gradient_seconds=np.asarray(gradient_seconds),
        update_seconds=np.asarray(update_seconds),
        stage_seconds=stage_seconds,
        compile_seconds=compile_seconds,
        memory=memory,
        final_train=final_train,
        forward_evaluations=forward_evaluations,
    )


def _stack_split_metrics(values: list[SplitMetrics]) -> SplitMetrics:
    return SplitMetrics(
        total_loss=np.stack([item.total_loss for item in values]),
        raw_total_mse=np.stack([item.raw_total_mse for item in values]),
        protocol_loss=np.stack([item.protocol_loss for item in values]),
        spike_counts=np.stack([item.spike_counts for item in values]),
    )


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
