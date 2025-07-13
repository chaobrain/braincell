import os

import numpy as np
from neuron import h


class Golgi_morpho_1():
    def __init__(self, el, gl, ghcn1, ghcn2, ena, gna, ek, gkv11, gkv34, gkv43):

        # path and loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        asc_path = os.path.join(current_dir, 'pair-140514-C2-1_split_1.asc')
        txt_path = os.path.join(current_dir, 'Optimization_result.txt')
        h.load_file('stdlib.hoc')
        h.load_file('import3d.hoc')
        # morpho
        cell = h.Import3d_Neurolucida3()
        cell.input(asc_path)
        i3d = h.Import3d_GUI(cell, 0)
        i3d.instantiate(self)

        sec_test = cell.sections.object(0)
        print('!!!!!!!', sec_test)
        # 获取点数量
        npt = int(sec_test.raw.ncol())

        # 获取点的x,y,z
        xs = [sec_test.raw.x[0][i] for i in range(npt)]
        ys = [sec_test.raw.x[1][i] for i in range(npt)]
        zs = [sec_test.raw.x[2][i] for i in range(npt)]

        # 打印所有点
        for i in range(npt):
            print(f"{i}: ({xs[i]:.6f}, {ys[i]:.6f}, {zs[i]:.6f})")
        # conductvalues
        conductvalues = np.genfromtxt(txt_path)

        # Soma

        self.soma[0].nseg = 1 + (2 * int(self.soma[0].L / 40))
        self.soma[0].Ra = 122
        self.soma[0].cm = 1

        if gl != 0:
            self.soma[0].insert('Leak')
            self.soma[0].gmax_Leak = 0.00003 * gl
            self.soma[0].e_Leak = el  # -55
        if gkv11 != 0:
            self.soma[0].insert('Kv1_1')
            self.soma[0].gbar_Kv1_1 = conductvalues[10] * gkv11
            self.soma[0].ek = ek  # -80
        if gkv34 != 0:
            self.soma[0].insert('Kv3_4')
            self.soma[0].gkbar_Kv3_4 = conductvalues[11] * gkv34
            self.soma[0].ek = ek
        if gkv43 != 0:
            self.soma[0].insert('Kv4_3')
            self.soma[0].gkbar_Kv4_3 = conductvalues[12] * gkv43
            self.soma[0].ek = ek
        if gna != 0:
            self.soma[0].insert('Nav1_6')
            self.soma[0].gbar_Nav1_6 = conductvalues[9] * gna
            self.soma[0].ena = ena

        '''
        self.soma[0].insert('Kca1_1')
        self.soma[0].gbar_Kca1_1 = conductvalues[13]
	  
        self.soma[0].insert('Kca3_1')
        self.soma[0].gkbar_Kca3_1 = conductvalues[14]
        
        self.soma[0].insert('GRC_CA')
        self.soma[0].gcabar_GRC_CA = conductvalues[15]
	
        self.soma[0].insert('Cav3_1')
        self.soma[0].pcabar_Cav3_1 = conductvalues[16]
        
        
        self.soma[0].insert('cdp5StCmod')
        self.soma[0].TotalPump_cdp5StCmod = 1e-7
        
        self.soma[0].eca = 137
        '''

        self.whatami = "golgi2020"

        # dend #to be redone

        self.dendbasal = []
        self.dendapical = []

        for en_index, d_sec in enumerate(self.dend):
            if en_index >= 0 and en_index <= 3 or en_index >= 16 and en_index <= 17 or en_index >= 33 and en_index <= 41 or en_index == 84 or en_index >= 105 and en_index <= 150:
                self.dendbasal.append(d_sec)

            if en_index >= 4 and en_index <= 15 or en_index >= 18 and en_index <= 32 or en_index >= 42 and en_index <= 83 or en_index >= 85 and en_index <= 104:
                self.dendapical.append(d_sec)

                # Dend apical	    
        for r in self.dendapical:
            r.nseg = 1 + (2 * int(r.L / 40))
            r.Ra = 122
            r.cm = 2.5

            if gl != 0:
                r.insert('Leak')
                r.gmax_Leak = 0.00003 * gl
                r.e_Leak = el
            if gna != 0:
                r.insert('Nav1_6')
                r.gbar_Nav1_6 = conductvalues[0] * gna
                r.ena = ena
            '''
            r.insert('Kca1_1')
            r.gbar_Kca1_1 = conductvalues[1]
            
            r.insert('Kca2_2')
            r.gkbar_Kca2_2 = conductvalues[2]
            r.ek = -80
            
            r.insert('Cav2_3')
            r.gcabar_Cav2_3 = conductvalues[3]
            
            r.insert('Cav3_1')
            r.pcabar_Cav3_1 = conductvalues[4]
            
            r.insert('cdp5StCmod')
            r.TotalPump_cdp5StCmod = 5e-9
            
            r.push()
            r.eca = 137
            h.pop_section()  
            '''

            # Dend basal	    
        for i in self.dendbasal:
            i.nseg = 1 + (2 * int(i.L / 40))
            i.Ra = 122
            i.cm = 2.5
            if gl != 0:
                i.insert('Leak')
                i.gmax_Leak = 0.00003 * gl
                i.e_Leak = el
            if gna != 0:
                i.insert('Nav1_6')
                i.gbar_Nav1_6 = conductvalues[5] * gna
                i.ena = ena
            '''
            i.insert('Kca1_1')
            i.gbar_Kca1_1 = conductvalues[6]
            
            i.insert('Kca2_2')
            i.gkbar_Kca2_2 = conductvalues[7]
            i.ek = -80

            i.insert('GRC_CA')
            i.gcabar_GRC_CA = conductvalues[8]
            
            
            i.insert('cdp5StCmod')
            i.TotalPump_cdp5StCmod = 2e-9
            
            i.push()
            i.eca = 137
            h.pop_section()   
            '''

        # axon
        for i, d in enumerate(self.axon):
            if i == 0:
                # AIS
                self.axon[i].nseg = 1 + (2 * int(self.axon[i].L / 40))
                self.axon[i].Ra = 122
                self.axon[i].cm = 1

                if gl != 0:
                    self.axon[i].insert('Leak')
                    self.axon[i].gmax_Leak = 0.00003 * gl
                    self.axon[i].e_Leak = el

                if ghcn1 != 0:
                    self.axon[i].insert('HCN1')
                    self.axon[i].gbar_HCN1 = conductvalues[17] * ghcn1

                if ghcn2 != 0:
                    self.axon[i].insert('HCN2')
                    self.axon[i].gbar_HCN2 = conductvalues[18] * ghcn2

                if gna != 0:
                    self.axon[i].insert('Nav1_6')
                    self.axon[i].gbar_Nav1_6 = conductvalues[19] * gna
                    self.axon[i].ena = ena

                '''
                self.axon[i].insert('GRC_KM')
                self.axon[i].gkbar_GRC_KM = conductvalues[20]
                
                self.axon[i].insert('Kca1_1')
                self.axon[i].gbar_Kca1_1 = conductvalues[21]               

                self.axon[i].insert('GRC_CA')
                self.axon[i].gcabar_GRC_CA = conductvalues[22]

                self.axon[i].ek = -80                 
                self.axon[i].insert('cdp5StCmod')	
                self.axon[i].TotalPump_cdp5StCmod = 1e-8
                
                self.axon[i].push()
                self.axon[i].eca = 137
                h.pop_section() 
                '''

            elif i >= 1:
                # axon
                self.axon[i].nseg = 1 + (2 * int(self.axon[i].L / 40))
                self.axon[i].cm = 1
                self.axon[i].Ra = 122

                if gl != 0:
                    self.axon[i].insert('Leak')
                    self.axon[i].e_Leak = el
                    self.axon[i].gmax_Leak = 0.000001 * gl
                if gkv34 != 0:
                    self.axon[i].insert('Kv3_4')
                    self.axon[i].gkbar_Kv3_4 = 0.0091 * gkv34
                    self.axon[i].ek = ek

                if gna != 0:
                    self.axon[i].insert('Nav1_6')
                    self.axon[i].gbar_Nav1_6 = 0.0115 * gna
                    self.axon[i].ena = ena
                '''
                self.axon[i].insert('cdp5StCmod')	    
                self.axon[i].TotalPump_cdp5StCmod = 1e-8
                
                self.axon[i].push()
                self.axon[i].eca = 137
                h.pop_section()   
                '''

        # Code to record everything.

        self.nc_spike = h.NetCon(self.soma[0](1)._ref_v, None, -20, 0.1, 1, sec=self.soma[0])

        self.time_vector = h.Vector()
        self.time_vector.record(h._ref_t)

        self.vm = h.Vector()
        self.vm.record(self.soma[0](0.5)._ref_v)

#     def createsyn(self, pf_n, mf_n, aa_n):	
# #PF       
#         self.L_PF = []
#         self.dend_pf = []

#         for sec_index, sec_sec in enumerate(self.dend):
#             if sec_index >= 4 and sec_index <= 15 or sec_index >= 18 and sec_index <= 32 or sec_index >= 42 and sec_index <= 83 or sec_index >= 85 and sec_index <= 104:
#                 self.dend_pf.append(sec_sec)

#         #To increase the number of synpases for each seaction
#         self.dend_pf = self.dend_pf *1

#         print('self.dend_pf', len(self.dend_pf))

# #PF location
#         for i in range(0, pf_n):
#             self.L_PF.append(Synapse_py3('PF',self,self.dend_pf[i])) 

#         print('pf_list_list', self.L_PF)

# #MOSSY        
#         self.L_MF = []
#         self.L_MF_NMDA_B = []
#         self.dend_mf = []
#         self.dend_aa = []

#         for sec_index, sec_sec in enumerate(self.dend):
#             if sec_index >= 108 and sec_index <= 112 or sec_index >= 114 and sec_index <= 121 or sec_index >= 128 and sec_index <= 129 or sec_index >= 131 and sec_index <= 132 or sec_index >= 135 and sec_index <= 140 or sec_index >= 144 and sec_index <= 145 or sec_index >= 147 and sec_index <= 150:
#                 self.dend_mf.append(sec_sec)   
#                 self.dend_aa.append(sec_sec) 

#         print('self.dend_mf', len(self.dend_mf))

# #MF location      
#         for i in range(0, mf_n):
#             self.L_MF.append(Synapse_py3('MF',self,self.dend_mf[i])) 
#             self.L_MF_NMDA_B.append(Synapse_py3('MF_nmda_B',self,self.dend_mf[i])) 

# #AA 
#         self.L_AA = [] 
#         self.L_AA_NMDA_B = []

#         for i in range(0, aa_n):
#                 self.L_AA.append(Synapse_py3('AA',self,self.dend_aa[i])) 
#                 self.L_AA_NMDA_B.append(Synapse_py3('MF_nmda_B',self,self.dend_aa[i]))
