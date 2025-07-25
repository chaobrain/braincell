# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Li, Guoshi, Craig S. Henriquez, and Flavio Fröhlich. “Unified thalamic model generates
  multiple distinct oscillations with state-dependent entrainment by stimulation.”
  PLoS computational biology 13.10 (2017): e1005797.
"""

from typing import Dict, Callable

import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numba
import numpy as np

import brainstate
from SC05_thalamus_single_compartment_neurons import TRN, RTC, HTC, IN

brainstate.environ.set(dt=0.1 * u.ms)


class MgBlock(brainstate.nn.SynOut):
    def __init__(self, E=0. * u.mV):
        super(MgBlock, self).__init__()
        self.E = E

    def update(self, conductance, potential):
        return conductance * (self.E - potential) / (1 + u.math.exp(-(potential / u.mV + 25) / 12.5))


class ProbDist:
    def __init__(self, dist=2., prob=0.3, pre_ratio=1.0, include_self=False):
        self.dist = dist

        @numba.njit
        def _pos2ind(pos, size):
            idx = 0
            for i, p in enumerate(pos):
                idx += p * np.prod(size[i + 1:])
            return idx

        @numba.njit
        def _connect_2d(pre_pos, pre_size, post_size):
            all_post_ids = []
            all_pre_ids = []
            if np.random.random() < pre_ratio:
                normalized_pos = np.zeros(2)
                for i in range(2):
                    pre_len = pre_size[i]
                    post_len = post_size[i]
                    normalized_pos[i] = pre_pos[i] * post_len / pre_len
                for i in range(post_size[0]):
                    for j in range(post_size[1]):
                        post_pos = np.asarray((i, j))
                        d = np.sqrt(np.sum(np.square(pre_pos - post_pos)))
                        if d <= dist:
                            if d == 0. and not include_self:
                                continue
                            if np.random.random() <= prob:
                                all_post_ids.append(_pos2ind(post_pos, post_size))
                                all_pre_ids.append(_pos2ind(pre_pos, pre_size))
            return all_pre_ids, all_post_ids  # Return filled part of the arrays

        self.connect = _connect_2d

    def __call__(self, pre_size, post_size):
        pre_size = np.asarray([pre_size[0] ** 0.5, pre_size[0] ** 0.5], dtype=int)
        post_size = np.asarray([post_size[0] ** 0.5, post_size[0] ** 0.5], dtype=int)
        connected_pres = []
        connected_posts = []
        pre_ids = np.meshgrid(*(np.arange(p) for p in pre_size), indexing='ij')
        pre_ids = tuple(
            [
                (np.moveaxis(p, 0, 1).flatten())
                if p.ndim > 1 else p.flatten()
                for p in pre_ids
            ]
        )
        size = np.prod(pre_size)

        for i in range(size):
            pre_pos = np.asarray([p[i] for p in pre_ids])
            pres, posts = self.connect(pre_pos, pre_size, post_size)
            connected_pres.extend(pres)
            connected_posts.extend(posts)
        return np.asarray(connected_pres), np.asarray(connected_posts)


class Thalamus(brainstate.nn.Module):
    def __init__(
        self,
        g_input: Dict[str, float],
        g_KL: Dict[str, float],
        HTC_V_init: Callable = brainstate.init.Constant(-65. * u.mV),
        RTC_V_init: Callable = brainstate.init.Constant(-65. * u.mV),
        IN_V_init: Callable = brainstate.init.Constant(-70. * u.mV),
        RE_V_init: Callable = brainstate.init.Constant(-70. * u.mV),
    ):
        super(Thalamus, self).__init__()

        # populations
        self.HTC = HTC(7 * 7, gKL=g_KL['TC'], V_initializer=HTC_V_init)
        self.RTC = RTC(12 * 12, gKL=g_KL['TC'], V_initializer=RTC_V_init)
        self.RE = TRN(10 * 10, gKL=g_KL['RE'], V_initializer=IN_V_init)
        self.IN = IN(8 * 8, gKL=g_KL['IN'], V_initializer=RE_V_init)

        # noises
        self.noise2HTC = brainstate.nn.AlignPostProj(
            brainstate.nn.PoissonSpike(self.HTC.varshape, freqs=100 * u.Hz),
            comm=brainstate.nn.OneToOne(self.HTC.varshape, g_input['TC'], ),
            syn=brainstate.nn.Expon.desc(self.HTC.varshape, tau=5. * u.ms),
            out=brainstate.nn.COBA.desc(E=0. * u.mV),
            post=self.HTC,
        )
        self.noise2RTC = brainstate.nn.AlignPostProj(
            brainstate.nn.PoissonSpike(self.RTC.varshape, freqs=100 * u.Hz),
            comm=brainstate.nn.OneToOne(self.RTC.varshape, g_input['TC']),
            syn=brainstate.nn.Expon.desc(self.RTC.varshape, tau=5. * u.ms),
            out=brainstate.nn.COBA.desc(E=0. * u.mV),
            post=self.RTC,
        )
        self.noise2IN = brainstate.nn.AlignPostProj(
            brainstate.nn.PoissonSpike(self.IN.varshape, freqs=100 * u.Hz),
            comm=brainstate.nn.OneToOne(self.IN.varshape, g_input['IN']),
            syn=brainstate.nn.Expon.desc(self.IN.varshape, tau=5. * u.ms),
            out=brainstate.nn.COBA.desc(E=0. * u.mV),
            post=self.IN,
        )
        self.noise2RE = brainstate.nn.AlignPostProj(
            brainstate.nn.PoissonSpike(self.RE.varshape, freqs=100 * u.Hz),
            comm=brainstate.nn.OneToOne(self.RE.varshape, g_input['RE']),
            syn=brainstate.nn.Expon.desc(self.RE.varshape, tau=5. * u.ms),
            out=brainstate.nn.COBA.desc(E=0. * u.mV),
            post=self.RE,
        )

        # HTC cells were connected with gap junctions
        self.gj_HTC = brainstate.nn.SymmetryGapJunction(
            self.HTC, 'V', conn=ProbDist(dist=2., prob=0.3), weight=1e-2 * u.siemens
        )

        # HTC provides feedforward excitation to INs
        self.HTC2IN_ampa = brainstate.nn.CurrentProj(
            self.HTC.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.AMPA(self.HTC.varshape, alpha=0.94 / (u.ms * u.mM), beta=0.18 / u.ms)
            ).prefetch('g'),
            comm=brainstate.nn.FixedNumConn(self.HTC.varshape, self.IN.varshape, 0.3, 6e-3),
            out=brainstate.nn.COBA(E=0. * u.mV),
            post=self.IN,
        )
        self.HTC2IN_nmda = brainstate.nn.CurrentProj(
            self.HTC.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07),
            ).align_pre(
                brainstate.nn.AMPA(self.HTC.varshape, alpha=1.0 / (u.ms * u.mM), beta=0.0067 / u.ms)
            ).prefetch('g'),
            comm=brainstate.nn.FixedNumConn(self.HTC.varshape, self.IN.varshape, 0.3, 3e-3),
            out=MgBlock(),
            post=self.IN,
        )

        # INs delivered feedforward inhibition to RTC cells
        self.IN2RTC = brainstate.nn.CurrentProj(
            self.IN.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.GABAa(self.IN.varshape, alpha=10.5 / (u.ms * u.mM), beta=0.166 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.IN.varshape, self.RTC.varshape, 0.3, 3e-3),
            out=brainstate.nn.COBA(E=-80. * u.mV),
            post=self.RTC,
        )

        # 20% RTC cells electrically connected with HTC cells
        self.gj_RTC2HTC = brainstate.nn.SymmetryGapJunction(
            (self.RTC, self.HTC), 'V', conn=ProbDist(dist=2., prob=0.3, pre_ratio=0.2), weight=1 / 300 * u.mS
        )

        # Both HTC and RTC cells sent glutamatergic synapses to RE neurons, while
        # receiving GABAergic feedback inhibition from the RE population
        self.HTC2RE_ampa = brainstate.nn.CurrentProj(
            self.HTC.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.AMPA(self.HTC.varshape, alpha=0.94 / (u.ms * u.mM), beta=0.18 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.HTC.varshape, self.RE.varshape, 0.2, 4e-3),
            out=brainstate.nn.COBA(E=0. * u.mV),
            post=self.RE,
        )
        self.RTC2RE_ampa = brainstate.nn.CurrentProj(
            self.RTC.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.AMPA(self.RTC.varshape, alpha=0.94 / (u.ms * u.mM), beta=0.18 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.RTC.varshape, self.RE.varshape, 0.2, 4e-3),
            out=brainstate.nn.COBA(E=0. * u.mV),
            post=self.RE,
        )
        self.HTC2RE_nmda = brainstate.nn.CurrentProj(
            self.HTC.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.AMPA(self.HTC.varshape, alpha=1. / (u.ms * u.mM), beta=0.0067 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.HTC.varshape, self.RE.varshape, 0.2, 2e-3),
            out=MgBlock(),
            post=self.RE,
        )
        self.RTC2RE_nmda = brainstate.nn.CurrentProj(
            self.RTC.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.AMPA(self.RTC.varshape, alpha=1. / (u.ms * u.mM), beta=0.0067 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.RTC.varshape, self.RE.varshape, 0.2, 2e-3),
            out=MgBlock(),
            post=self.RE,
        )
        self.RE2HTC = brainstate.nn.CurrentProj(
            self.RE.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.GABAa(self.RE.varshape, alpha=10.5 / (u.ms * u.mM), beta=0.166 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.RE.varshape, self.HTC.varshape, 0.2, 3e-3),
            out=brainstate.nn.COBA(E=-80. * u.mV),
            post=self.HTC,
        )
        self.RE2RTC = brainstate.nn.CurrentProj(
            self.RE.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.GABAa(self.RE.varshape, alpha=10.5 / (u.ms * u.mM), beta=0.166 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.RE.varshape, self.RTC.varshape, 0.2, 3e-3),
            out=brainstate.nn.COBA(E=-80. * u.mV),
            post=self.RTC,
        )

        # RE neurons were connected with both gap junctions and GABAergic synapses
        self.gj_RE = brainstate.nn.SymmetryGapJunction(
            self.RE, 'V', conn=ProbDist(dist=2., prob=0.3, pre_ratio=0.2), weight=1 / 300 * u.mS
        )
        self.RE2RE = brainstate.nn.CurrentProj(
            self.RE.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.GABAa(self.RE.varshape, alpha=10.5 / (u.ms * u.mM), beta=0.166 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.RE.varshape, self.RE.varshape, 0.2, 1e-3),
            out=brainstate.nn.COBA(E=-70. * u.mV),
            post=self.RE,
        )

        # 10% RE neurons project GABAergic synapses to local interneurons
        # probability (0.05) was used for the RE->IN synapses according to experimental data
        self.RE2IN = brainstate.nn.CurrentProj(
            self.RE.align_pre(
                brainstate.nn.STD.desc(tau=700 * u.ms, U=0.07)
            ).align_pre(
                brainstate.nn.GABAa(self.RE.varshape, alpha=10.5 / (u.ms * u.mM), beta=0.166 / u.ms)
            ).prefetch_delay('g', 2 * u.ms),
            comm=brainstate.nn.FixedNumConn(self.RE.varshape, self.IN.varshape, 0.05, 1e-3, afferent_ratio=0.1),
            out=brainstate.nn.COBA(E=-80. * u.mV),
            post=self.IN,
        )

    def update(self, i, t, current):
        with brainstate.environ.context(t=t, i=i):
            self.noise2HTC()
            self.noise2RTC()
            self.noise2IN()
            self.noise2RE()

            self.HTC2IN_ampa()
            self.HTC2IN_nmda()

            self.IN2RTC()

            self.HTC2RE_ampa()
            self.RTC2RE_ampa()
            self.HTC2RE_nmda()
            self.RTC2RE_nmda()

            self.RE2HTC()
            self.RE2RTC()
            self.RE2RE()
            self.RE2IN()

            self.gj_HTC()
            self.gj_RTC2HTC()
            self.gj_RE()

            htc_spike = self.HTC(current)
            rtc_spike = self.RTC(current)
            re_spike = self.RE(current)
            in_spike = self.IN(current)

        return {
            'HTC.V': self.HTC.V.value,
            'RTC.V': self.RTC.V.value,
            'IN.V': self.IN.V.value,
            'RE.V': self.RE.V.value,
            'HTC.spike': htc_spike,
            'RTC.spike': rtc_spike,
            'RE.spike': re_spike,
            'IN.spike': in_spike,
        }


states = {
    'delta': dict(
        g_input={'IN': 1e-4 * u.mS, 'RE': 1e-4 * u.mS, 'TC': 1e-4 * u.mS},
        g_KL={'TC': 0.035 * u.mS / u.cm ** 2, 'RE': 0.03 * u.mS / u.cm ** 2, 'IN': 0.01 * u.mS / u.cm ** 2}
    ),
    'spindle': dict(
        g_input={'IN': 3e-4 * u.mS, 'RE': 3e-4 * u.mS, 'TC': 3e-4 * u.mS},
        g_KL={'TC': 0.01 * u.mS / u.cm ** 2, 'RE': 0.02 * u.mS / u.cm ** 2, 'IN': 0.015 * u.mS / u.cm ** 2}
    ),
    'alpha': dict(
        g_input={'IN': 1.5e-3 * u.mS, 'RE': 1.5e-3 * u.mS, 'TC': 1.5e-3 * u.mS},
        g_KL={'TC': 0. * u.mS / u.cm ** 2, 'RE': 0.01 * u.mS / u.cm ** 2, 'IN': 0.02 * u.mS / u.cm ** 2}
    ),
    'gamma': dict(
        g_input={'IN': 1.5e-3 * u.mS, 'RE': 1.5e-3 * u.mS, 'TC': 1.7e-2 * u.mS},
        g_KL={'TC': 0. * u.mS / u.cm ** 2, 'RE': 0.01 * u.mS / u.cm ** 2, 'IN': 0.02 * u.mS / u.cm ** 2}
    ),
}


def rhythm_const_input(amp, freq, length, duration, t_start=0., t_end=None, dt=None):
    if t_end is None:
        t_end = duration
    if length > duration:
        raise ValueError(f'Expected length <= duration, while we got {length} > {duration}')
    sec_length = 1 / freq
    values, durations = [0. * u.mA], [t_start]
    for t in u.math.arange(t_start, t_end, sec_length):
        values.append(amp)
        if t + length <= t_end:
            durations.append(length)
            values.append(0. * u.mA)
            if t + sec_length <= t_end:
                durations.append(sec_length - length)
            else:
                durations.append(t_end - t - length)
        else:
            durations.append(t_end - t)
    values.append(0. * u.mA)
    durations.append(duration - t_end)
    return braintools.input.section_input(values=values, durations=durations, dt=dt)


def line_plot(ax, xs, ys, ylabel=None, xlim=None):
    ax.plot(xs, ys)
    ax.set_xlabel('Time (ms)')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)


def raster_plot(
    ax, times, spikes, ylabel='Neuron Index', xlabel='Time [ms]',
    xlim=None, marker='.', markersize=2, color='k'
):
    elements = np.where(spikes > 0.)
    index = elements[1]
    time = times[elements[0]]
    ax.plot(time, index, marker + color, markersize=markersize)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)


def try_network(state='delta'):
    net = Thalamus(
        IN_V_init=brainstate.init.Constant(-70. * u.mV),
        RE_V_init=brainstate.init.Constant(-70. * u.mV),
        HTC_V_init=brainstate.init.Constant(-80. * u.mV),
        RTC_V_init=brainstate.init.Constant(-80. * u.mV),
        **states[state],
    )
    brainstate.nn.init_all_states(net)

    duration = 3e3 * u.ms  # 3 seconds
    currents = rhythm_const_input(
        2e-4 * u.mA,
        freq=4. * u.Hz,
        length=10. * u.ms,
        duration=duration,
        t_end=2e3 * u.ms,
        t_start=1e3 * u.ms
    )
    indices = np.arange(currents.shape[0])
    times = indices * brainstate.environ.get_dt()
    mon = brainstate.transform.for_loop(net.update, indices, times, currents, pbar=200)

    fig, gs = braintools.visualize.get_figure(5, 2, 2, 5)
    line_plot(fig.add_subplot(gs[0, :]), times, currents, ylabel='Current', xlim=(0, duration / u.ms))
    line_plot(fig.add_subplot(gs[1, 0]), times, mon.get('HTC.V'), ylabel='HTC', xlim=(0, duration / u.ms))
    line_plot(fig.add_subplot(gs[2, 0]), times, mon.get('RTC.V'), ylabel='RTC', xlim=(0, duration / u.ms))
    line_plot(fig.add_subplot(gs[3, 0]), times, mon.get('IN.V'), ylabel='IN', xlim=(0, duration / u.ms))
    line_plot(fig.add_subplot(gs[4, 0]), times, mon.get('RE.V'), ylabel='RE', xlim=(0, duration / u.ms))
    raster_plot(fig.add_subplot(gs[1, 1]), times, mon.get('HTC.spike'), xlim=(0, duration / u.ms))
    raster_plot(fig.add_subplot(gs[2, 1]), times, mon.get('RTC.spike'), xlim=(0, duration / u.ms))
    raster_plot(fig.add_subplot(gs[3, 1]), times, mon.get('IN.spike'), xlim=(0, duration / u.ms))
    raster_plot(fig.add_subplot(gs[4, 1]), times, mon.get('RE.spike'), xlim=(0, duration / u.ms))
    plt.suptitle(f'Thalamus Network State: {state}')
    plt.show()


if __name__ == '__main__':
    # try_network('delta')
    # try_network('spindle')
    try_network('alpha')
    try_network('gamma')
