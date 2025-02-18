# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
import functools

import brainstate
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

from simple_dendrite_model_simulation import solve_explicit_solver

brainstate.environ.set(precision=64)


def visualize_a_simulate(params, f_current, show=True, title=''):
    saveat = u.math.arange(0., 100., 0.1) * u.ms
    ts, vs, _ = solve_explicit_solver(params, f_current, saveat)

    fig, gs = braintools.visualize.get_figure(1, 1, 3.0, 4.0)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(ts.to_decimal(u.ms), u.math.squeeze(vs.to_decimal(u.mV)))
    plt.xlabel('Time [ms]')
    plt.ylabel('Potential [mV]')
    if title:
        plt.title(title)
    if show:
        plt.show()


def fitting_example():
    t1 = 200 * u.ms

    # Step 1: generating input currents
    saveat = u.math.arange(0., t1 / u.ms, 0.2) * u.ms

    def f_current(t, i_current, *args):
        return jax.lax.switch(
            i_current,
            [
                lambda t: u.math.where(t < 50. * u.ms,
                                       0. * u.nA,
                                       u.math.where(t < 100. * u.ms, 0.5 * u.nA, 0. * u.nA)),
                lambda t: u.math.where(t < 60. * u.ms,
                                       0. * u.nA,
                                       u.math.where(t < 160. * u.ms, 0.2 * u.nA, 0. * u.nA)),
                lambda t: u.math.where(t < 80. * u.ms,
                                       0. * u.nA,
                                       u.math.where(t < 160. * u.ms, 1.0 * u.nA, 0. * u.nA)),
                lambda t: u.math.where(t < 100. * u.ms,
                                       0.2 * u.nA,
                                       u.math.where(t < 150. * u.ms, 0.1 * u.nA, 0.3 * u.nA)),
            ],  # suppose there are 4 input currents
            t
        )

    # Step 2: generating the target neuronal parameters
    target_params = np.asarray([0.12, 0.036])

    # Step 3: generating the target potentials

    def simulate_per_param(param):
        indices = np.arange(4)
        fun = functools.partial(solve_explicit_solver, param, f_current, saveat, t1)
        _, simulated_vs, _ = jax.vmap(fun)((indices,))  # [n_input, T, n_compartment]
        return simulated_vs

    target_vs = simulate_per_param(target_params)

    # Step 4: initialize a batch of parameters to optimize,
    #         these parameters are candidates to be optimized
    bounds = [
        np.asarray([0.05, 0.01]),
        np.asarray([0.2, 0.1])
    ]
    n_batch = 8
    param_to_optimize = brainstate.ParamState(brainstate.random.uniform(bounds[0], bounds[1], (n_batch, bounds[0].size)))

    # Step 5: define the loss function and optimizers

    # calculate the loss for each parameter based on
    # the mismatch between the 4 simulated and target potentials
    def loss_per_param(param, step=10):
        simulated_vs = simulate_per_param(param)  # [n_input, T, n_compartment]
        losses = braintools.metric.squared_error(simulated_vs.mantissa[..., ::step, 0], target_vs.mantissa[..., ::step, 0])
        return losses.mean()

    # calculate the gradients and loss for each parameter
    @jax.vmap
    @jax.jit
    def compute_grad(param):
        grads, loss = brainstate.augment.grad(loss_per_param, argnums=0, return_value=True)(param)
        return grads, loss

    # find the best loss and parameter in the batch
    @brainstate.compile.jit
    def best_loss_and_param(params, losses):
        i_best = u.math.argmin(losses)
        return losses[i_best], params[i_best]

    # define the optimizer
    optimizer = brainstate.optim.Adam(lr=1e-3)
    optimizer.register_trainable_weights({'param': param_to_optimize})

    # Step 6: training
    @brainstate.compile.jit
    def train_step_per_epoch():
        grads, losses = compute_grad(param_to_optimize.value)
        optimizer.update({'param': grads})
        return losses

    for i_epoch in range(1000):
        losses = train_step_per_epoch()
        best_loss, best_param = best_loss_and_param(param_to_optimize.value, losses)
        if best_loss < 1e-5:
            print(f'Epoch {i_epoch}, loss={losses.mean()}, best loss={best_loss}, best param={best_param}')
            break
        if i_epoch % 10 == 0:
            print(f'Epoch {i_epoch}, loss={losses.mean()}, best loss={best_loss}, best param={best_param}')

    # Step 7: visualize the results
    visualize_a_simulate(target_params, functools.partial(f_current, i_current=0), title='Target', show=False)
    visualize_a_simulate(best_param, functools.partial(f_current, i_current=0), title='Fitted', show=True)


if __name__ == '__main__':
    fitting_example()
