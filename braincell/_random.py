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

"""Integer seeds drawn from the BrainState default random stream.

What has no seed takes one integer here: ``Network()`` and
``filter.sample(...)`` / ``filter.RandomSamples(...)`` when they are built, a
``NetStim`` with noise when it is built, and a pairing rule in a direct
``braincell.connect`` when it is called. A single
``brainstate.random.seed_context(s)`` (or ``brainstate.random.seed(s)``)
then controls them together with initializers and callbacks. Everything they
derive afterwards is a pure function of that integer; ``Network``,
``sample`` and ``RandomSamples`` store and report it, so it can be recorded
and passed back as an explicit seed.

This module imports only BrainState and JAX: ``braincell.network`` imports it
while ``braincell`` is still initializing.
"""

import brainstate
import jax
import jax.numpy as jnp

__all__ = ["draw_seed"]

#: Drawn seeds are non-negative 31-bit integers, valid for every seed consumer
#: (NumPy, BrainState ``RandomState``, ``jax.random.key``) on every precision.
_SEED_MASK = 0x7FFF_FFFF


def draw_seed(what: str, *, remedy: str = "Build it outside the transformation, or pass an integer seed.") -> int:
    """Draw one integer seed in ``[0, 2**31)`` from the default random stream.

    The key comes from ``brainstate.random.split_key()``, so the draw advances
    the default stream (the active ``brainstate.random.seed_context``) by one
    split, like any BrainState call without ``key=``. The value is
    ``int(jax.random.bits(key, (), uint32)) & 0x7FFFFFFF`` and does not depend
    on the float precision.

    Parameters
    ----------
    what : str
        The call that draws, as the error message names it
        (``"Network(seed=None)"``).
    remedy : str, optional
        What the error message tells the user to do instead.

    Returns
    -------
    int
        The drawn seed.

    Raises
    ------
    TypeError
        If the default stream is being traced (the call runs inside
        ``brainstate.transform.jit``, ``for_loop`` or another BrainState
        transformation), where no concrete integer exists.
    """
    key = brainstate.random.split_key()
    if isinstance(key, jax.core.Tracer):
        raise TypeError(
            f"{what} draws its seed from the BrainState default random stream, which is traced "
            f"here (inside jit, a loop or another transformation). {remedy}"
        )
    return int(jax.random.bits(key, (), jnp.uint32)) & _SEED_MASK
