"""Network run result containers."""

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NetworkRunResult:
    """Return value of :meth:`braincell.network.Network.run`.

    Attributes
    ----------
    time : brainunit.Quantity
        Step times spanning ``[start_t, start_t + duration)``.
    traces : dict
        ``population_name -> {probe_name: trace}`` mapping.
    spikes : dict
        ``population_name -> spike_trace`` mapping.
    """

    time: object
    traces: dict
    spikes: dict
