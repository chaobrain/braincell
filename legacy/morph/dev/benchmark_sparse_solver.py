# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import time

import brainevent
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import spsolve


def get_diag_indices(arr, k=0):
    idx = np.where(np.eye(arr.shape[0], arr.shape[1], k=k) == 1)
    return idx


def benchmark(length=1000):
    dense = u.Quantity(jnp.zeros([length, length]))
    dense[jnp.diag_indices(length, ndim=2)] = brainstate.random.rand(length)
    dense[get_diag_indices(dense, k=-1)] = brainstate.random.rand(length - 1)
    dense[get_diag_indices(dense, k=1)] = brainstate.random.rand(length - 1)
    dense = dense.mantissa

    csr = brainevent.CSR.fromdense(dense)

    @jax.jit
    def f1(x):
        return spsolve(csr.data, csr.indices, csr.indptr, x)

    @jax.jit
    def f2(x):
        return jnp.linalg.solve(dense, x)

    x = brainstate.random.rand(length)

    f1(x)
    f2(x)

    n_round = 10
    t0 = time.time()
    for _ in range(n_round):
        jax.block_until_ready(f1(x))
    t1 = time.time()
    sp_time = t1 - t0

    t0 = time.time()
    for _ in range(n_round):
        jax.block_until_ready(f2(x))
    t1 = time.time()
    dense_time = t1 - t0

    print(f'length = {length}, sparse time = {sp_time:.8f}s, dense time = {dense_time:.8f}s')


for length in [100, 1000, 2000, 5000, 10000]:
    benchmark(length)
