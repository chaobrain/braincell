import warnings

import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
try:
    from neuron import h, gui
except ImportError:
    warnings.warn('NEURON is not installed. NEURON-related functions will not work.', ImportWarning)


# NEURON run
def NeuronRun(cell, stim, tstop, dt, v_init):
    # create record vector
    t_vec = h.Vector()
    t_vec.record(h._ref_t)
    v_vecs = []

    spike_times = h.Vector()
    nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0])
    nc.threshold = 0
    nc.record(spike_times)

    for sec in h.allsec():
        for seg in sec:
            v_vec = h.Vector()
            v_vec.record(seg._ref_v)
            v_vecs.append(v_vec)

    # simulation
    h.celsius = 22
    h.dt = dt
    h.v_init = v_init
    h.finitialize(v_init)

    h.continuerun(tstop)
    return t_vec, v_vecs, spike_times


def step_stim(cell, delay, dur, amp):
    stim = h.IClamp(cell.soma[0](0.5))
    stim.delay = delay
    stim.dur = dur
    stim.amp = amp
    return stim


# Braincell run 
@brainstate.transform.jit(static_argnums=(0,))
def BraincellRun(cell, I, dt):
    # time
    brainstate.environ.set(dt=dt * u.ms)
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()
    # init and reset
    cell.init_state()
    cell.reset_state()
    # run
    vs = brainstate.compile.for_loop(cell.step_run, times, I) #vs = 

    return times.to_decimal(u.ms), vs.to_decimal(u.mV)


# different input func
def step_input(num, dur, amp, dt):
    brainstate.environ.set(dt=dt * u.ms)
    value = u.math.zeros((len(dur), num))
    for i in range(len(value)):
        value = value.at[i, 0].set(amp[i])

    I = braintools.input.section_input(values=value, durations=dur * u.ms) * u.nA
    return I


# Passive parameters
def sec_passive_params(
    morph,
    nseg_length=40 * u.um,
    Ra_soma=122, cm_soma=1,
    Ra_dend=122, cm_dend=2.5,
    Ra_axon=122, cm_axon=1
):
    """
    Set nseg, Ra, and cm for all sections in a morphology.

    Parameters
    ----------
    morph : Morphology
        Morphology object with .sections (dict).
    nseg_length : float or Quantity
        Target segment length (default: 40 um).
    Ra_soma, cm_soma, Ra_dend, cm_dend, Ra_axon, cm_axon : float
        Ra and cm values for each section type.

    Notes
    -----
    Section type is identified by the name substring: 'soma', 'dend', or 'axon'.
    """
    for k, v in morph.sections.items():
        # Update nseg based on section length
        v.nseg = int(1 + 2 * np.floor(v.L / nseg_length))

        # Set Ra and cm by section type
        if 'soma' in k:
            v.Ra = Ra_soma
            v.cm = cm_soma
        elif 'dend' in k:
            v.Ra = Ra_dend
            v.cm = cm_dend
        elif 'axon' in k:
            v.Ra = Ra_axon
            v.cm = cm_axon


# Biophysical param

# index for ion channel 
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

    # conductvalues 
    conductvalues = 1e3 * np.array([
        0.00499506303209, 0.01016375552607, 0.00247172479141, 0.00128859564935,
        3.690771983E-05, 0.0080938853146, 0.01226052748146, 0.01650689958385,
        0.00139885617712, 0.14927733727426, 0.00549507510519, 0.14910988921938,
        0.00406420380423, 0.01764345789036, 0.10177335775222, 0.0087689418803,
        3.407734319E-05, 0.0003371456442, 0.00030643090764, 0.17233663543619,
        0.00024381226198, 0.10008178886943, 0.00595046001148, 0.0115, 0.0091
    ])

    # IL 
    gl = np.ones(n_compartments)
    gl[index_soma] = 0.03
    gl[index_axon] = 0.001
    gl[index_axon[0:5]] = 0.03
    gl[index_dend_basal] = 0.03
    gl[index_dend_apical] = 0.03

    # IKv11_Ak2007
    gkv11 = np.zeros(n_compartments)
    gkv11[index_soma] = conductvalues[10]

    # IKv34_Ma2020  
    gkv34 = np.zeros(n_compartments)
    gkv34[index_soma] = conductvalues[11]
    gkv34[index_axon[5:]] = 9.1

    # IKv43_Ma2020
    gkv43 = np.zeros(n_compartments)
    gkv43[index_soma] = conductvalues[12]

    # ICaGrc_Ma2020
    gcagrc = np.zeros(n_compartments)
    gcagrc[index_soma] = conductvalues[15]
    gcagrc[index_dend_basal] = conductvalues[8]
    gcagrc[index_axon[0:5]] = conductvalues[22]

    # ICav23_Ma2020
    gcav23 = np.zeros(n_compartments)
    gcav23[index_dend_apical] = conductvalues[3]

    # ICav31_Ma2020 
    gcav31 = np.zeros(n_compartments)
    gcav31[index_soma] = conductvalues[16]
    gcav31[index_dend_apical] = conductvalues[4]

    # INa_Rsg
    gnarsg = np.zeros(n_compartments)
    gnarsg[index_soma] = conductvalues[9]
    gnarsg[index_dend_apical] = conductvalues[0]
    gnarsg[index_dend_basal] = conductvalues[5]
    gnarsg[index_axon[0:5]] = conductvalues[19]
    gnarsg[index_axon[5:]] = 11.5

    # Ih1_Ma2020 
    gh1 = np.zeros(n_compartments)
    gh1[index_axon[0:5]] = conductvalues[17]

    # Ih2_Ma2020 
    gh2 = np.zeros(n_compartments)
    gh2[index_axon[0:5]] = conductvalues[18]

    # IKca3_1_Ma2020 
    gkca31 = np.zeros(n_compartments)
    gkca31[index_soma] = conductvalues[14]

    return gl, gh1, gh2, gkv11, gkv34, gkv43, gnarsg, gcagrc, gcav23, gcav31, gkca31


def plot_voltage_traces(
    t_vec,
    v_vecs,
    indices=None,
    title="Voltage Traces",
    xlabel="Time (ms)",
    ylabel="Voltage (mV)",
    legend=True,
    color_map="tab10",
    figsize=(7, 4),
    grid=True
):
    """
    Plot voltage traces for multiple sections.

    Parameters
    ----------
    t_vec : array-like
        Time vector.
    v_vecs : array-like (2D, shape: [n_section, n_time])
        Voltage traces, one per section.
    indices : list or range or None
        Which sections to plot. Default: all.
    title, xlabel, ylabel : str
        Plot labels.
    legend : bool
        Whether to show legend.
    color_map : str
        Matplotlib colormap for lines.
    figsize : tuple
        Figure size.
    grid : bool
        Whether to show grid.
    """
    if indices is None:
        indices = range(len(v_vecs))
    colors = plt.get_cmap(color_map)
    plt.figure(figsize=figsize)
    for idx, i in enumerate(indices):
        plt.plot(t_vec, v_vecs[i], label=f"sec_{i}", color=colors(idx % colors.N), linewidth=2)
    if legend:
        plt.legend(frameon=False, fontsize=11)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, weight='bold')
    if grid:
        plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_voltage_comparison(
    t_vec, v_vecs,
    t_ref, v_refs,
    indices=None,
    label1="NEURON",
    label2="Braincell",
    color1="tab:green",
    color2="tab:orange",
    ref_linestyle=(0, (4, 4)),
    title="Voltage Comparison",
    xlabel="Time (ms)",
    ylabel="Voltage (mV)",
    figsize=(7, 4),
    legend=True,
    grid=True
):
    """
    Plot voltage comparison curves for multiple sections.

    Parameters
    ----------
    t_vec : array-like
        Time vector for the first set (e.g., NEURON).
    v_vecs : array-like (2D: [n_section, n_time])
        First voltage set, shape [n_section, n_time].
    t_ref : array-like
        Time vector for reference set (e.g., Braincell).
    v_refs : array-like (2D/3D)
        Reference voltage, shape [n_time, n_repeat, n_section] or [n_section, n_time]
    indices : list/range/None
        Which section indices to plot (default: all).
    label1, label2 : str
        Legend labels.
    color1, color2 : str
        Line colors.
    ref_linestyle : tuple or str
        Reference line style.
    title, xlabel, ylabel : str
        Plot labels.
    figsize : tuple
        Figure size.
    legend : bool
        Show legend.
    grid : bool
        Show grid.
    """
    if indices is None:
        indices = range(len(v_vecs))
    plt.figure(figsize=figsize)
    for idx, i in enumerate(indices):
        plt.plot(t_vec, v_vecs[i], color=color1, label=f"{label1}_{i}", linewidth=2)
        # 适配v_refs为3D (n_time, n_repeat, n_section)或2D
        if v_refs.ndim == 3:
            plt.plot(t_ref, v_refs[:, 0, i], linestyle=ref_linestyle, color=color2, label=f"{label2}_{i}", linewidth=2)
        else:
            plt.plot(t_ref, v_refs[i], linestyle=ref_linestyle, color=color2, label=f"{label2}_{i}", linewidth=2)
    if legend:
        plt.legend(frameon=False, fontsize=11)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, weight='bold')
    if grid:
        plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

# dt v.s error 

# num_dt = 5
# dt_list = [0.005 *2 ** i for i in range(num_dt)]
# error_1_list = []
# error_2_list = []
# error_inf_list = []
# v_list = [] 

# # neuron
# cell = Golgi_morpho_1(el=-55, gl=1, ghcn1=1, ghcn2=1, ena=50, gna=0, ek=-80, gkv11=1, gkv34=1, gkv43=1)
# stim = step_stim(cell, delay=0, dur=100, amp=0.0)

# t_vec, v_vecs, _  = NeuronRun(cell=cell, stim=stim, tstop=100, dt=0.005, v_init=-65)
# t_vec = np.array(t_vec);t_vec = t_vec[:-1000]
# v_vecs = np.array(v_vecs);v_vecs = v_vecs[:,:-1000]

# v_list.append(v_vecs[0])

# for i in range(num_dt):

#     El = -55; Gl=gl;Ek=-80; Ena=60
#     V_init = -65*np.ones(n_compartments)
#     V_init[0] = -65

#     cell = Golgi(size=size, el=El, gl=gl, gh1=gh1, gh2=gh2, ek=Ek, gkv11=gkv11, gkv34=gkv34, gkv43=gkv43, ena=Ena, gnarsg=gnarsg, 
#                 Gl=Gl, El = El, V_init=V_init, solver = 'staggered')

#     cell.add_multiple_sections(mor_info)
#     cell.connect_sections(mor_connection) 
#     print(cell.cm * cell.area)

#     I = step_input(num=n_compartments, dur=[100,0,0], amp=[0.0,0.,0], dt=dt_list[i])
#     t1, v1 = BraincellRun(neu=cell, I=I, dt=dt_list[i])
#     v1 = np.roll(v1, 1, axis=0)
#     v1[0] = V_init

#     # interpolation for the same length
#     f = interp1d(t1, v1[:,:,0].reshape(-1), kind='linear')
#     v_list.append(f(t_vec))
#     # error
#     error_1_list.append(np.linalg.norm(v_vecs[0] -f(t_vec),1)/len(v_vecs[0]))
#     error_2_list.append(np.linalg.norm(v_vecs[0] -f(t_vec),2)/len(v_vecs[0]))
#     #error_inf_list.append(np.linalg.norm(v_vecs[0] -f(t_vec),np.inf))

# # log_e
# log_dt = np.log10(dt_list)
# log_error_1 = np.log10(error_1_list)
# #log_error_2 = np.log10(error_2_list)
# #log_error_inf = np.log10(error_inf_list)

# # ref line
# x0 = log_dt[0]
# y0 = log_error_1[0]
# b = y0 - 1 * x0  
# reference_line_y = 1 * log_dt + b
# plt.plot(log_dt, reference_line_y, linestyle='--', color='red', label='Slope = 1')

# # plot
# plt.plot(log_dt, log_error_1, marker='o', label='1-norm error')
# #plt.plot(log_dt, log_error_2, marker='s', label='2-norm error')
# #plt.plot(log_dt, log_error_inf, marker='^', label='Inf-norm error')

# #plt.title('Errors vs dt(CN+Rk4)')
# plt.xlabel('log(dt)')
# plt.ylabel('log(Error)')
# plt.grid(True)
# plt.legend()

# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# # dt compare
# plt.figure(figsize=(8, 6))
# for i in range(num_dt+1):
#     if i == 0:
#         plt.plot(t_vec,v_list[i], linewidth=3,color = 'black', label = 'NEURON(dt=5*1e-3)')
#     elif i<=9:
#         plt.plot(t_vec,v_list[i], alpha = 0.9, label = f'dt={dt_list[i-1]}ms')
#     plt.legend()
# plt.xlabel('Time (ms)')
# plt.ylabel('Voltage (mV)')
# #plt.title('Voltage')
# plt.grid(True)
# plt.show()
