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

"""Gradient optimizer stages for the composable experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import braintools
import optax


@dataclass(frozen=True)
class AdamStage:
    """Run one exact-gradient plain Adam phase."""

    epochs: int = 180
    learning_rate: float = 0.01
    gradient_method: str = "rtrl"
    checkpoint_every: int = 10
    name: str = "adam"
    kind: str = "gradient"

    @property
    def validation_every(self) -> int:
        """Return the validation evaluation cadence."""
        return self.checkpoint_every

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.checkpoint_every < 1 or self.epochs % self.checkpoint_every:
            raise ValueError("epochs must be positive and divisible by checkpoint_every.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.gradient_method not in {"rtrl", "bptt"}:
            raise ValueError("gradient_method must be 'rtrl' or 'bptt'.")

    def build_optimizer(self, parameter_states):
        """Create a fresh plain Adam optimizer for one stage."""
        optimizer = braintools.optim.Adam(lr=self.learning_rate)
        optimizer.register_trainable_weights(parameter_states)
        return optimizer

    def describe(self) -> dict[str, object]:
        """Return serializable stage metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "optimizer": "Adam",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "gradient_method": self.gradient_method,
            "checkpoint_every": self.checkpoint_every,
            "gradient_clipping": None,
            "weight_decay": 0.0,
        }


def adam(*, validation_every: int | None = None, **kwargs) -> AdamStage:
    """Return a plain Adam gradient stage."""
    _apply_validation_alias(validation_every, kwargs)
    return AdamStage(**kwargs)


@dataclass(frozen=True)
class RpropStage:
    """Run one exact-gradient resilient-backpropagation phase."""

    epochs: int = 300
    learning_rate: float = 0.01
    gradient_method: str = "rtrl"
    checkpoint_every: int = 10
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-6, 50.0)
    name: str = "rprop"
    kind: str = "gradient"

    @property
    def validation_every(self) -> int:
        """Return the validation evaluation cadence."""
        return self.checkpoint_every

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.checkpoint_every < 1 or self.epochs % self.checkpoint_every:
            raise ValueError("epochs must be positive and divisible by checkpoint_every.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.gradient_method not in {"rtrl", "bptt"}:
            raise ValueError("gradient_method must be 'rtrl' or 'bptt'.")

    def build_optimizer(self, parameter_states):
        """Create fresh per-lane Rprop step-size state."""
        optimizer = braintools.optim.Rprop(
            lr=self.learning_rate,
            etas=self.etas,
            step_sizes=self.step_sizes,
        )
        optimizer.register_trainable_weights(parameter_states)
        return optimizer

    def describe(self) -> dict[str, object]:
        """Return serializable stage metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "optimizer": "Rprop",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "etas": list(self.etas),
            "step_sizes": list(self.step_sizes),
            "gradient_method": self.gradient_method,
            "checkpoint_every": self.checkpoint_every,
            "gradient_clipping": None,
        }


def rprop(*, validation_every: int | None = None, **kwargs) -> RpropStage:
    """Return a resilient-backpropagation gradient stage."""
    _apply_validation_alias(validation_every, kwargs)
    return RpropStage(**kwargs)


@dataclass(frozen=True)
class OptaxRpropStage:
    """Run Rprop with exactly one learning-rate application."""

    epochs: int = 300
    learning_rate: float = 1e-4
    gradient_method: str = "rtrl"
    checkpoint_every: int = 10
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-6, 50.0)
    name: str = "optax_rprop"
    kind: str = "gradient"

    @property
    def validation_every(self) -> int:
        """Return the validation evaluation cadence."""
        return self.checkpoint_every

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.checkpoint_every < 1 or self.epochs % self.checkpoint_every:
            raise ValueError("epochs must be positive and divisible by checkpoint_every.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.gradient_method not in {"rtrl", "bptt"}:
            raise ValueError("gradient_method must be 'rtrl' or 'bptt'.")

    def build_optimizer(self, parameter_states):
        """Create a custom-tx optimizer without wrapper schedule scaling."""
        tx = optax.rprop(
            learning_rate=self.learning_rate,
            eta_minus=self.etas[0],
            eta_plus=self.etas[1],
            min_step_size=self.step_sizes[0],
            max_step_size=self.step_sizes[1],
        )
        optimizer = braintools.optim.OptaxOptimizer(tx=tx, lr=self.learning_rate)
        optimizer.register_trainable_weights(parameter_states)
        return optimizer

    def describe(self) -> dict[str, object]:
        """Return serializable single-scale Rprop metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "optimizer": "OptaxRprop",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lr_application": "single_in_custom_tx",
            "etas": list(self.etas),
            "step_sizes": list(self.step_sizes),
            "gradient_method": self.gradient_method,
            "checkpoint_every": self.checkpoint_every,
            "gradient_clipping": None,
        }


def optax_rprop(*, validation_every: int | None = None, **kwargs) -> OptaxRpropStage:
    """Return a single-scale Optax Rprop gradient stage."""
    _apply_validation_alias(validation_every, kwargs)
    return OptaxRpropStage(**kwargs)


@dataclass(frozen=True)
class SGDStage:
    """Run one vanilla stochastic-gradient-descent phase."""

    epochs: int = 300
    learning_rate: float = 1e-4
    momentum: float = 0.0
    nesterov: bool = False
    gradient_method: str = "rtrl"
    checkpoint_every: int = 10
    name: str = "sgd"
    kind: str = "gradient"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.checkpoint_every < 1 or self.epochs % self.checkpoint_every:
            raise ValueError("epochs must be positive and divisible by checkpoint_every.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0,1).")
        if self.nesterov and self.momentum <= 0.0:
            raise ValueError("Nesterov SGD requires positive momentum.")
        if self.gradient_method not in {"rtrl", "bptt"}:
            raise ValueError("gradient_method must be 'rtrl' or 'bptt'.")

    @property
    def validation_every(self) -> int:
        """Return the validation evaluation cadence."""
        return self.checkpoint_every

    def build_optimizer(self, parameter_states):
        """Create fresh vanilla SGD state without momentum or regularization."""
        optimizer = braintools.optim.SGD(
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=0.0,
            nesterov=self.nesterov,
        )
        optimizer.register_trainable_weights(parameter_states)
        return optimizer

    def describe(self) -> dict[str, object]:
        """Return serializable vanilla SGD metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "optimizer": "SGD",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "weight_decay": 0.0,
            "gradient_method": self.gradient_method,
            "checkpoint_every": self.checkpoint_every,
            "gradient_clipping": None,
        }


def sgd(*, validation_every: int | None = None, **kwargs) -> SGDStage:
    """Return a vanilla SGD gradient stage."""
    _apply_validation_alias(validation_every, kwargs)
    return SGDStage(**kwargs)


def _apply_validation_alias(validation_every: int | None, kwargs: dict[str, object]) -> None:
    if validation_every is None:
        return
    if "checkpoint_every" in kwargs and kwargs["checkpoint_every"] != validation_every:
        raise ValueError("validation_every and legacy checkpoint_every disagree.")
    kwargs["checkpoint_every"] = validation_every
