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
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
import time
import jax.numpy as jnp
import brainstate
import brainunit as u
from utils import plot_traces, plot_spike_raster
brainstate.environ.set(precision=32, platform='gpu')

import braincell
import numpy as np
import braintools

def is_basal(idx):
    return (
        0 <= idx <= 3
        or 16 <= idx <= 17
        or 33 <= idx <= 41
        or idx == 84
        or 105 <= idx <= 150
    )


def is_apical(idx):
    return (
        4 <= idx <= 15
        or 18 <= idx <= 32
        or 42 <= idx <= 83
        or 85 <= idx <= 104
    )


def step_input(num, dur, amp):
    value = u.math.zeros((len(dur), num))
    for i in range(len(value)):
        value = value.at[i, 0].set(amp[i])
    return braintools.input.section(values=value, durations=dur * u.ms) * u.nA


def seg_ion_params(morphology):
    # segment index for each type
    index_soma = []
    index_axon = []
    index_dend_basal = []
    index_dend_apical = []

    for i, seg in enumerate(morphology.segments):
        name = str(seg.section_name)
        if name.startswith("soma"):
            index_soma.append(i)
        elif name.startswith("axon"):
            index_axon.append(i)
        elif name.startswith("dend_"):
            idx = int(name.split("_")[-1])
            if is_basal(idx):
                index_dend_basal.append(i)
            if is_apical(idx):
                index_dend_apical.append(i)

    n_compartments = len(morphology.segments)

    # conductance values
    conduct_values = 1e3 * np.array(
        [
            0.00499506303209, 0.01016375552607, 0.00247172479141, 0.00128859564935,
            3.690771983E-05, 0.0080938853146, 0.01226052748146, 0.01650689958385,
            0.00139885617712, 0.14927733727426, 0.00549507510519, 0.14910988921938,
            0.00406420380423, 0.01764345789036, 0.10177335775222, 0.0087689418803,
            3.407734319E-05, 0.0003371456442, 0.00030643090764, 0.17233663543619,
            0.00024381226198, 0.10008178886943, 0.00595046001148, 0.0115, 0.0091
        ]
    )

    # IL
    gl = np.ones(n_compartments)
    gl[index_soma] = 0.03
    gl[index_axon] = 0.001
    gl[index_axon[0:5]] = 0.03
    gl[index_dend_basal] = 0.03
    gl[index_dend_apical] = 0.03

    # IKv11_Ak2007
    gkv11 = np.zeros(n_compartments)
    gkv11[index_soma] = conduct_values[10]

    # IKv34_Ma2020
    gkv34 = np.zeros(n_compartments)
    gkv34[index_soma] = conduct_values[11]
    gkv34[index_axon[5:]] = 9.1

    # IKv43_Ma2020
    gkv43 = np.zeros(n_compartments)
    gkv43[index_soma] = conduct_values[12]

    # ICaGrc_Ma2020
    gcagrc = np.zeros(n_compartments)
    gcagrc[index_soma] = conduct_values[15]
    gcagrc[index_dend_basal] = conduct_values[8]
    gcagrc[index_axon[0:5]] = conduct_values[22]

    # ICav23_Ma2020
    gcav23 = np.zeros(n_compartments)
    gcav23[index_dend_apical] = conduct_values[3]

    # ICav31_Ma2020
    gcav31 = np.zeros(n_compartments)
    gcav31[index_soma] = conduct_values[16]
    gcav31[index_dend_apical] = conduct_values[4]

    # INa_Rsg
    gnarsg = np.zeros(n_compartments)
    gnarsg[index_soma] = conduct_values[9]
    gnarsg[index_dend_apical] = conduct_values[0]
    gnarsg[index_dend_basal] = conduct_values[5]
    gnarsg[index_axon[0:5]] = conduct_values[19]
    gnarsg[index_axon[5:]] = 11.5

    # Ih1_Ma2020
    gh1 = np.zeros(n_compartments)
    gh1[index_axon[0:5]] = conduct_values[17]

    # Ih2_Ma2020
    gh2 = np.zeros(n_compartments)
    gh2[index_axon[0:5]] = conduct_values[18]

    # IKca3_1_Ma2020
    gkca31 = np.zeros(n_compartments)
    gkca31[index_soma] = conduct_values[14]

    return gl, gh1, gh2, gkv11, gkv34, gkv43, gnarsg, gcagrc, gcav23, gcav31, gkca31


class Golgi(braincell.MultiCompartment):
    def __init__(
        self,
        popsize,
        morphology,
        E_L,
        gl,
        gh1,
        gh2,
        E_K,
        gkv11,
        gkv34,
        gkv43,
        E_Na,
        gnarsg,
        V_init=-65 * u.mV,
    ):
        super().__init__(
            popsize=popsize,
            morphology=morphology,
            V_th=0. * u.mV,
            V_initializer=braintools.init.Constant(V_init),
            spk_fun=braintools.surrogate.ReluGrad(),
            solver='staggered'
        )

        self.IL = braincell.channel.IL(self.varshape, E=E_L, g_max=gl * u.mS / (u.cm ** 2))
        self.Ih1 = braincell.channel.Ih1_Ma2020(self.varshape, E=-20. * u.mV, g_max=gh1 * u.mS / (u.cm ** 2))
        self.Ih2 = braincell.channel.Ih2_Ma2020(self.varshape, E=-20. * u.mV, g_max=gh2 * u.mS / (u.cm ** 2))

        self.k = braincell.ion.PotassiumFixed(self.varshape, E=E_K)
        self.k.add(IKv11=braincell.channel.IKv11_Ak2007(self.varshape, g_max=gkv11 * u.mS / (u.cm ** 2)))
        self.k.add(IKv34=braincell.channel.IKv34_Ma2020(self.varshape, g_max=gkv34 * u.mS / (u.cm ** 2)))
        self.k.add(IKv43=braincell.channel.IKv43_Ma2020(self.varshape, g_max=gkv43 * u.mS / (u.cm ** 2)))

        self.na = braincell.ion.SodiumFixed(self.varshape, E=E_Na)
        self.na.add(INa_Rsg=braincell.channel.INa_Rsg(self.varshape, g_max=gnarsg * u.mS / (u.cm ** 2), solver = 'rk4', compute_steps=5))
        #self.na.add(INa = braincell.channel.INa_HH1952(self.varshape, g_max=gnarsg * u.mS / (u.cm ** 2)))
        self.na.add(Syn_i = braincell.channel.SynExp(self.varshape,))
        self.na.add(Syn_e = braincell.channel.SynExp(self.varshape, E_syn = 0 * u.mV))

    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            get_spike = self.update(inp)
            return self.V.value, get_spike

morphology = braincell.Morphology.from_asc('golgi.asc')
morphology.set_passive_params()
gl, gh1, gh2, gkv11, gkv34, gkv43, gnarsg, gcagrc, gcav23, gcav31, gkca31 = seg_ion_params(morphology)

def make_cell(morphology, popsize=50):
    return Golgi(
        popsize= popsize,  
        morphology=morphology,
        E_L=-55. * u.mV,
        gl=gl,
        gh1=gh1,
        gh2=gh2,
        E_K=-80. * u.mV,
        gkv11=gkv11,
        gkv34=gkv34,
        gkv43=gkv43,
        E_Na=60. * u.mV,
        gnarsg=gnarsg,
        V_init=-65 * u.mV,
    )

def compute_g_post(pre_spike, size_post, conn):
    """
    Compute post-synaptic conductance map g_post from pre_spike using a sparse connection (conn).

    This function assumes the connectivity is already precomputed and packed as:
        conn = (pre_idx, post_idx, w)
    where:
        - pre_idx:  (num_syn,) flattened indices into pre_spike.reshape(-1)
        - post_idx: (num_syn,) flattened indices into g_post_flat
        - w:        (num_syn,) synaptic weights

    Flattening convention:
        flat_idx = cell_id * nseg + seg_id

    What it does (one step, no synaptic dynamics):
        1) pre_spike_flat = pre_spike.reshape(-1)
        2) g_syn = pre_spike_flat[pre_idx] * w
        3) scatter-add g_syn into g_post_flat at post_idx
        4) reshape back to (ncell_post, nseg_post)

    Example:
        size_post = (S=2, K=5)  -> post_flat has length 10
        conn:
            pre_idx  = [6, 8]
            post_idx = [3, 9]
            w        = [0.5, 1.2]
        If pre_spike_flat[6]=1 and pre_spike_flat[8]=0, then:
            g_syn = [0.5, 0.0]
            g_post_flat[3] += 0.5
            g_post_flat[9] += 0.0

    Notes:
        - Uses `.add`, so multiple synapses targeting the same post_idx will accumulate.
        - If you want "last one wins" behavior, replace `.add` with `.set`.
    """
    pre_idx, post_idx, w = conn
    ncell_post, nseg_post = size_post

    # Flatten pre spikes so pre_idx can index into a 1D array
    pre_spike_flat = pre_spike.reshape(-1)

    # Per-synapse conductance contribution (instantaneous)
    g_syn = pre_spike_flat[pre_idx] * w

    # Scatter-add onto post (flattened), then reshape back
    n_post_flat = ncell_post * nseg_post
    g_post_flat = jnp.zeros((n_post_flat,), dtype=g_syn.dtype)
    g_post_flat = g_post_flat.at[post_idx].add(g_syn)

    return g_post_flat.reshape(ncell_post, nseg_post)


def generate_random_synapses_flat(
    size_pre,
    size_post,
    N_syn,
    weight=1.0,
    pre_seg_list=None,
):
    """
    Generate `N_syn` random synapses with unique (pre_flat, post_flat) pairs.

    Flattening convention:
        flat_idx = cell_id * nseg + seg_id

    Args:
        size_pre:  (ncell_pre, nseg_pre)
        size_post: (ncell_post, nseg_post)
        N_syn: number of synapses to sample (unique pairs)
        weight: scalar weight assigned to every sampled synapse (can be extended to per-synapse later)
        pre_seg_list: optional list of allowed pre segment indices.
                      Example: [0] means "only soma segment" for all pre cells.

    Returns:
        conn tuple (pre_flat_idx, post_flat_idx, w):
            - pre_flat_idx:  (N_syn,) int, flattened indices into pre side
            - post_flat_idx: (N_syn,) int, flattened indices into post side
            - w:             (N_syn,) float, weights

    Example:
        Pre:  N=3 cells, M=4 segs -> size_pre=(3,4) => pre_flat in [0..11]
        Post: S=2 cells, K=5 segs -> size_post=(2,5) => post_flat in [0..9]
        pre_seg_list=[0] restricts pre_flat_allowed to:
            [0*4+0, 1*4+0, 2*4+0] = [0, 4, 8]
        Then each synapse is a unique pair (pre_flat in {0,4,8}, post_flat in [0..9]).

    Note:
        This function samples *unique* pairs from the Cartesian product:
            allowed_pre_flat × all_post_flat
        So it guarantees no duplicate (pre_flat, post_flat) connections.
    """
    ncell_pre, nseg_pre = size_pre
    ncell_post, nseg_post = size_post

    # 1) Build allowed pre_flat indices
    if pre_seg_list is None:
        pre_flat_allowed = np.arange(ncell_pre * nseg_pre)
    else:
        pre_seg_list = np.asarray(pre_seg_list, dtype=int)
        if np.any((pre_seg_list < 0) | (pre_seg_list >= nseg_pre)):
            raise ValueError("pre_seg_list contains out-of-range segment index.")
        # For each pre cell, allow only the specified segments
        pre_flat_allowed = (np.arange(ncell_pre)[:, None] * nseg_pre + pre_seg_list[None, :]).ravel()

    # 2) Total possible unique pairs = (#allowed_pre) * (#all_post)
    total_post = ncell_post * nseg_post
    total_pairs = pre_flat_allowed.size * total_post
    if N_syn > total_pairs:
        raise ValueError(f"N_syn={N_syn} exceeds maximum possible unique pairs={total_pairs}")

    # 3) Sample unique pairs from the Cartesian product (no replacement)
    # Pair index encoding:
    #   pair = pre_choice * total_post + post_flat
    pair = np.random.choice(total_pairs, size=N_syn, replace=False)
    pre_choice = pair // total_post
    post_flat_idx = pair % total_post
    pre_flat_idx = pre_flat_allowed[pre_choice]

    # 4) Attach weights (currently constant per synapse)
    w = np.full((N_syn,), weight)

    return (pre_flat_idx, post_flat_idx, w)

size_bg  = (50, 50)
cell_1 = make_cell(morphology, popsize=50) 
cell_2 = make_cell(morphology, popsize=50) 
size_1 = cell_1.in_size
size_2 = cell_2.in_size

conn_1_2     = generate_random_synapses_flat(size_1,  size_2,  2000, pre_seg_list=[0])
conn_2_1     = generate_random_synapses_flat(size_2,  size_1,  2000, pre_seg_list=[0])
conn_bg_to_1 = generate_random_synapses_flat(size_bg, size_1,  5000, pre_seg_list=[0])
conn_bg_to_2 = generate_random_synapses_flat(size_bg, size_2,  5000, pre_seg_list=[0])

brainstate.environ.set(dt=0.01 * u.ms)
I = step_input(num=size_1[1], dur=[100, 0, 0], amp=[0, 0, 0])

def random_spike(key, size_bg, p=0.01):
    return (jax.random.uniform(key, size_bg) < p).astype(jnp.float32)

def inject_g(post_cell, chan_name, gain, pre_spike, conn):
    g = compute_g_post(pre_spike, post_cell.in_size, conn)
    post_cell.na.channels[chan_name].g.value += gain * g

def net_step_run(t, inp, key_t=None):
    v_1, spike_1 = cell_1.step_run(t, inp)
    v_2, spike_2 = cell_2.step_run(t, inp)

    bg_spike_1 = random_spike(key=key_t,         size_bg =  size_bg, p=0.002)
    bg_spike_2 = random_spike(key=key_t * 2 + 1, size_bg =  size_bg, p=0.001)

    inject_g(cell_1, "Syn_e", 1.0, bg_spike_1, conn_bg_to_1)
    inject_g(cell_1, "Syn_i", 1.5, spike_2,    conn_2_1)

    inject_g(cell_2, "Syn_e", 1.0, bg_spike_2, conn_bg_to_2)
    inject_g(cell_2, "Syn_i", 1.5, spike_1,    conn_1_2)

    return v_1, spike_1, v_2, spike_2

@brainstate.transform.jit
def simulate(I):
    T = I.shape[0]
    master_key = jax.random.PRNGKey(0)
    keys = jax.random.split(master_key, T)  
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()
    cell_1.init_state()
    cell_1.reset_state()
    cell_2.init_state()
    cell_2.reset_state()
    v_1, spike_1, v_2, spike_2 = brainstate.transform.for_loop(net_step_run, times, I, keys)  
    return times.to_decimal(u.ms), v_1[:,:,0].to_decimal(u.mV), spike_1, v_2[:,:,0].to_decimal(u.mV), spike_2

t0 = time.time()
t, v_1, spike_1, v_2, spike_2 = jax.block_until_ready(simulate(I)) #, v_2, spike_2
t1 = time.time()

t2 = time.time()
t, v_1, spike_1, v_2, spike_2 = jax.block_until_ready(simulate(I))
t3 = time.time()

print(f'First run time = {t1 - t0} s, second run time = {t3 - t2} s')

plot_traces(v_2, n=2, t=t)
plot_spike_raster(spike_2)