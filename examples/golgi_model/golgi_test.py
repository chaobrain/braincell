# %%
import os
import sys

current_dir = os.path.dirname(os.path.abspath('.'))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
dendritex_path = os.path.join(project_root, 'braincell')
sys.path.insert(0, dendritex_path)

import time
import brainstate 
import braintools 
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np
import braincell as dx
import jax.numpy as jnp

brainstate .environ.set(precision=64)
#jax.config.update("jax_disable_jit", True)

# %%
loaded_params = np.load('golgi_morphology.npz')

connection = loaded_params['connection']
L = loaded_params['L']  # um
diam = loaded_params['diam']  # um
Ra = loaded_params['Ra']  # ohm * cm
cm = loaded_params['cm']  # uF / cm ** 2

n_neuron = 1
n_compartments = len(L)
size = (n_neuron, n_compartments)

index_soma = loaded_params['index_soma']
index_axon = loaded_params['index_axon']
index_dend_basal = loaded_params['index_dend_basal']
index_dend_apical = loaded_params['index_dend_apical']

## conductvalues 
conductvalues = 1e3 * np.array([

    0.00499506303209, 0.01016375552607, 0.00247172479141, 0.00128859564935,
    3.690771983E-05, 0.0080938853146, 0.01226052748146, 0.01650689958385,
    0.00139885617712, 0.14927733727426, 0.00549507510519, 0.14910988921938,
    0.00406420380423, 0.01764345789036, 0.10177335775222, 0.0087689418803,
    3.407734319E-05, 0.0003371456442, 0.00030643090764, 0.17233663543619,
    0.00024381226198, 0.10008178886943, 0.00595046001148, 0.0115, 0.0091
])

## IL 
gl = np.ones(n_compartments)
gl[index_soma] = 0.03
gl[index_axon] = 0.001
gl[index_axon[0]] = 0.03
gl[index_dend_basal] = 0.03
gl[index_dend_apical] = 0.03

## IKv11_Ak2007
gkv11 = np.zeros(n_compartments)
gkv11[index_soma] = conductvalues[10]

## IKv34_Ma2020  
gkv34 = np.zeros(n_compartments)
gkv34[index_soma] = conductvalues[11]
gkv34[index_axon[1:]] = 9.1

## IKv43_Ma2020
gkv43 = np.zeros(n_compartments)
gkv43[index_soma] = conductvalues[12]

## ICaGrc_Ma2020
gcagrc = np.zeros(n_compartments)
gcagrc[index_soma] = conductvalues[15]
gcagrc[index_dend_basal] = conductvalues[8]
gcagrc[index_axon[0]] = conductvalues[22]

## ICav23_Ma2020
gcav23 = np.zeros(n_compartments)
gcav23[index_dend_apical] = conductvalues[3]

## ICav31_Ma2020 
gcav31 = np.zeros(n_compartments)
gcav31[index_soma] = conductvalues[16]
gcav31[index_dend_apical] = conductvalues[4]

## INa_Rsg
gnarsg = np.zeros(n_compartments)
gnarsg[index_soma] = conductvalues[9]
gnarsg[index_dend_apical] = conductvalues[0]
gnarsg[index_dend_basal] = conductvalues[5]
gnarsg[index_axon[0]] = conductvalues[19]
gnarsg[index_axon[1:]] = 11.5

## Ih1_Ma2020 
gh1 = np.zeros(n_compartments)
gh1[index_axon[0]] = conductvalues[17]

## Ih2_Ma2020 
gh2 = np.zeros(n_compartments)
gh2[index_axon[0]] = conductvalues[18]

## IKca3_1_Ma2020 
gkca31 = np.zeros(n_compartments)
gkca31[index_soma] = conductvalues[14]

# %%
# single ion test
'''
connection = ((1, 2), (2, 3))
gl = np.zeros(n_compartments)
g_test = np.zeros(n_compartments)
g_test[index_soma] = 2.5e-4
'''

# %%
class Golgi(dx.neuron.MultiCompartment):
    def __init__(self, size, connection, Ra, cm, diam, L, gl, gh2, solver = 'exp_euler'):
        super().__init__(
            size=size,
            connection=connection,
            Ra=Ra * u.ohm * u.cm,
            cm=cm * u.uF / u.cm ** 2,
            diam=diam * u.um,
            L=L * u.um,
            V_th=20. * u.mV,
            V_initializer=brainstate.init.Constant(-55 * u.mV),
            spk_fun=brainstate.surrogate.ReluGrad(),
            solver = solver 
        )
        #self.IL = dx.channel.IL(self.varshape, E=-55. * u.mV, g_max=gl * u.mS / (u.cm ** 2))
        #self.Ih1 = dx.channel.Ih1_Ma2020(self.varshape, E= -20. * u.mV, g_max= 0 * u.mS / (u.cm ** 2))
        #self.Ih2 = dx.channel.Ih2_Ma2020(self.varshape, E= -20. * u.mV, g_max= 0 * u.mS / (u.cm ** 2))
        self.k = dx.ion.PotassiumFixed(self.varshape, E=-80. * u.mV)
        self.k.add(IK = dx.channel.IKM_Grc_Ma2020(self.varshape, g_max= 0 * u.mS / (u.cm ** 2)))
        #self.ca = dx.ion.CalciumFixed(self.size, E=137.* u.mV, C =5e-5 * u.mM)
        #self.ca.add_elem(dx.channel.ICav31_Ma2020(self.size, g_max=g_test * (u.cm / u.second)))
        #self.kca = dx.MixIons(self.k, self.ca)
        #self.kca.add_elem(dx.channel.IKca1_1_Ma2020(self.size, g_max=g_test * u.mS / (u.cm ** 2)))
    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            self.update(inp)
            return self.V.value

# %%
def try_trn_neuron():
    import jax
    #jax.config.update("jax_disable_jit", True)
    brainstate.environ.set(dt=0.001 * u.ms)

    I = braintools.input.section_input(values=[0.000001, 0, 0], durations=[20 * u.ms, 1 * u.ms, 1 * u.ms]) * u.uA
    
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()

    neu = Golgi(size, connection, Ra, cm, diam, L, gl, gh1, solver='splitting')  # [n_neuron,]
    neu.init_state()

    t0 = time.time()
    vs = brainstate.compile.for_loop(neu.step_run, times, I)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.4f} s")

    print(vs.shape)
    #print(times.shape)
    #jnp.save('times.npy', times.to_decimal(u.ms))
    #jnp.save('vs.npy', vs.to_decimal(u.mV))
    plt.plot(times.to_decimal(u.ms), u.math.squeeze(vs.to_decimal(u.mV)))
    #plt.savefig('neuron_plot.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    try_trn_neuron()

