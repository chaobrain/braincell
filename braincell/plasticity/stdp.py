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

"""Spike-timing-dependent plasticity rules."""

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base_rule import ConnectionRule, row_parameter
from braincell.mech import register_connection_rule

__all__ = [
    "PairSTDP",
]


@register_connection_rule("PairSTDP")
class PairSTDP(ConnectionRule):
    """All-to-all pair spike-timing-dependent plasticity with additive updates.

    The pair rule of [1]_ in the trace form of [2]_: each pre-post pair
    changes the weight by an amount that decays exponentially with the time
    between them. Each row keeps a presynaptic trace ``x_pre`` and a
    postsynaptic trace ``y_post``, both dimensionless, which decay exactly
    between event boundaries: ``x_pre' = -x_pre / tau_plus``,
    ``y_post' = -y_post / tau_minus``. At a boundary where the row has ``n_pre`` arrivals and its
    member ``n_post`` post events, every read sees the state just before the
    boundary::

        w      <- w + A_plus * x_pre * n_post - A_minus * y_post * n_pre
        w      <- clip(w, w_min, w_max)
        x_pre  <- x_pre + n_pre
        y_post <- y_post + n_post

    so a pre arrival ``dt_pre`` before a post event adds
    ``A_plus * exp(-dt_pre / tau_plus)`` and a post event ``dt_post`` before
    a pre arrival subtracts ``A_minus * exp(-dt_post / tau_minus)``; every
    earlier event of the other kind pairs. Events on the same boundary do not
    pair, whatever order they are processed in. A pre time is the arrival,
    after the row's delay; the event delivered there carries the weight
    before the update.

    Declare it with ``braincell.mech.ConnectionRule("PairSTDP", ...)``;
    ``connect`` builds the instance.

    Parameters
    ----------
    size : int
        Number of rows.
    weight_unit : brainunit.Unit
        Unit of the rows' weight.
    A_plus, A_minus : brainunit.Quantity
        Potentiation and depression amplitudes, in a unit compatible with the
        weight, scalar or one value per row.
    tau_plus, tau_minus : brainunit.Quantity
        Positive trace time constants, scalar or one value per row.
    w_min, w_max : brainunit.Quantity, optional
        Hard bounds applied after every update; an omitted side is unbounded.
        Both may be negative (an inhibitory current weight, for example).

    Attributes
    ----------
    signals : tuple of str
        ``("post",)``: the post events default to the target Cell's spikes.

    Raises
    ------
    TypeError
        If an amplitude or a time constant has no unit.
    ValueError
        If a unit is incompatible, a time constant is not positive,
        ``w_min > w_max``, or a shape is neither scalar nor ``(size,)``.

    Notes
    -----
    The state names are ``"w"`` (the weight), ``"x_pre"`` and ``"y_post"``;
    ``ConnectionView.rule_state(name)`` reads them, and
    ``ConnectionView.record(recording, name)`` records them during a run.
    The traces decay at the start of each update, so a stored trace is its
    value just after the latest boundary. An initial weight outside
    ``[w_min, w_max]`` is rejected (``ValueError``), not clipped. Nearest
    neighbour, triplet and weight-dependent variants are other rules.

    References
    ----------
    .. [1] Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian
       learning through spike-timing-dependent synaptic plasticity. Nature
       Neuroscience, 3(9), 919-926. doi:10.1038/78829
    .. [2] Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological
       models of synaptic plasticity based on spike timing. Biological
       Cybernetics, 98(6), 459-478. doi:10.1007/s00422-008-0233-1

    Examples
    --------
    A pre event at 0 ms and a post event at 10 ms potentiate the row by
    ``A_plus * exp(-10 / 20)`` (in the default float32):

    .. code-block:: python

        >>> import brainunit as u
        >>> import braincell
        >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        >>> cell = cell.place(braincell.mech.Synapse("ExpSyn", name="s"))
        >>> stdp = braincell.mech.ConnectionRule(
        ...     "PairSTDP",
        ...     A_plus=0.01 * u.uS,
        ...     A_minus=0.0105 * u.uS,
        ...     tau_plus=20.0 * u.ms,
        ...     tau_minus=20.0 * u.ms,
        ...     signals={"post": braincell.EventSequence.from_times(([10.0] * u.ms,))},
        ... )
        >>> _ = braincell.connect(
        ...     "c",
        ...     source=braincell.EventSequence.from_times(([0.0] * u.ms,)),
        ...     synapse=cell.synapses["s"],
        ...     weight=0.05 * u.uS,
        ...     plasticity=stdp,
        ... )
        >>> _ = cell.run(dt=0.1 * u.ms, duration=12.0 * u.ms)
        >>> dw = cell.connections["c"].rule_state("w") - 0.05 * u.uS
        >>> print(f"{float(dw.to_decimal(u.uS)[0]):.6f}")
        0.006065
    """

    __module__ = "braincell.plasticity"

    signals = ("post",)

    def __init__(
        self,
        size: int,
        weight_unit,
        *,
        A_plus,
        A_minus,
        tau_plus,
        tau_minus,
        w_min=None,
        w_max=None,
    ) -> None:
        super().__init__(size, weight_unit)
        rule = type(self).__name__
        self.A_plus = row_parameter(A_plus, unit=weight_unit, size=size, name="A_plus", rule=rule)
        self.A_minus = row_parameter(A_minus, unit=weight_unit, size=size, name="A_minus", rule=rule)
        self.tau_plus = row_parameter(tau_plus, unit=u.ms, size=size, name="tau_plus", rule=rule)
        self.tau_minus = row_parameter(tau_minus, unit=u.ms, size=size, name="tau_minus", rule=rule)
        for name in ("tau_plus", "tau_minus"):
            if np.any(np.asarray(getattr(self, name).to_decimal(u.ms)) <= 0.0):
                raise ValueError(f"{rule} parameter {name!r} must be > 0 ms; got {getattr(self, name)!r}.")
        self.w_min = (
            None if w_min is None else row_parameter(w_min, unit=weight_unit, size=size, name="w_min", rule=rule)
        )
        self.w_max = (
            None if w_max is None else row_parameter(w_max, unit=weight_unit, size=size, name="w_max", rule=rule)
        )
        if self.w_min is not None and self.w_max is not None:
            if np.any(np.asarray(self.w_min.to_decimal(weight_unit)) > np.asarray(self.w_max.to_decimal(weight_unit))):
                raise ValueError(f"{rule} requires w_min <= w_max; got w_min={self.w_min!r}, w_max={self.w_max!r}.")

    def init_state(self, weight) -> dict:
        """Return the weight and zero traces.

        Parameters
        ----------
        weight : brainunit.Quantity
            The rows' initial weight.

        Returns
        -------
        dict
            ``{"w": weight, "x_pre": 0, "y_post": 0}``, one entry per row.
        """
        zeros = jnp.zeros((self.size,), dtype=brainstate.environ.dftype())
        return {"w": weight, "x_pre": zeros, "y_post": zeros}

    def update(self, state: dict, *, pre, post, dt) -> dict:
        """Advance the traces and the weight over one boundary.

        Parameters
        ----------
        state : dict
            ``w``, ``x_pre`` and ``y_post`` just before the boundary.
        pre, post : jax.Array
            Arrivals of each row and post events of its member at the
            boundary.
        dt : brainunit.Quantity
            Time since the previous boundary.

        Returns
        -------
        dict
            The state after the boundary.
        """
        x_pre = state["x_pre"] * u.math.exp(-dt / self.tau_plus)
        y_post = state["y_post"] * u.math.exp(-dt / self.tau_minus)
        w = state["w"] + self.A_plus * x_pre * post - self.A_minus * y_post * pre
        if self.w_min is not None:
            w = u.math.maximum(w, self.w_min)
        if self.w_max is not None:
            w = u.math.minimum(w, self.w_max)
        return {"w": w, "x_pre": x_pre + pre, "y_post": y_post + post}

    def check_weight(self, weight) -> None:
        """Reject an initial weight outside ``[w_min, w_max]``.

        Parameters
        ----------
        weight : brainunit.Quantity
            The rows' weight.

        Raises
        ------
        ValueError
            If a row's weight is below ``w_min`` or above ``w_max``.
        """
        decimal = np.asarray(u.Quantity(weight).to_decimal(self.weight_unit), dtype=np.float64)
        for name, bound, outside in (("w_min", self.w_min, np.less), ("w_max", self.w_max, np.greater)):
            if bound is None:
                continue
            rows = np.flatnonzero(outside(decimal, np.asarray(bound.to_decimal(self.weight_unit))))
            if rows.size:
                raise ValueError(
                    f"{type(self).__name__} weight of {rows.size} row(s) (first: row {int(rows[0])}, "
                    f"{decimal[rows[0]]} {self.weight_unit}) is outside {name}={bound!r}; "
                    "the rule rejects a weight outside its bounds rather than clipping it."
                )
