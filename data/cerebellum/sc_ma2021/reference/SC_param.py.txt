SC_param=dict()


SC_param['Ra'] = 110
SC_param['ena'] = 60

#Soma

SC_param['Eleak'] = -52 # -48
SC_param['leak_soma'] = 0.00023 #0.00003 # 0.000008

SC_param['Nav1.1_soma'] = 0.2
SC_param['Cav3.2_soma'] = 0.00163912063769
SC_param['Cav3.3_soma'] = 0.00001615552993
SC_param['Kir2.3_soma'] = 0.00001093425575
SC_param['Kv3.4_soma'] = 0.015
SC_param['Kv4.3_soma'] = 0.00404228168138
SC_param['Kca1.1_soma'] =  0.00518036298671
SC_param['Kv1.1_soma'] = 0.00107430134923
SC_param['Kca2.2_soma'] = 0.00054166094878
SC_param['Cav2.1_soma'] = 0.0005 # 0.00038
SC_param['HCN1_soma'] = 0.00058451678362

#Dendrites proximal

SC_param['Cav2.1_dendprox'] = 0.0008 # 0.0005
SC_param['Cav3.2_dendprox'] = 0.00070661092763
SC_param['Cav3.3_dendprox'] = 0.00001526216781
SC_param['Kca1.1_dendprox'] = 0.00499205404769
SC_param['Kca2.2_dendprox'] = 0.00000326194117
SC_param['Kv1.1_dendprox'] = 0.00906810561650
SC_param['Kv4.3_dendprox'] = 0.00264204713540
SC_param['leak_dendprox'] = 0.000008

##Dendrites distal

SC_param['Cav2.1_denddist'] = 0.00025
SC_param['Kca1.1_denddist'] = 0.00226329455766
SC_param['Kca2.2_denddist'] = 0.00001079984416
SC_param['Kv1.1_denddist'] = 0.00237825442906
SC_param['leak_denddist'] = 0.000008

#AIS

SC_param['Nav1.6_ais'] = 0.3
SC_param['Kv3.4_ais'] = 0.03351450571128
SC_param['Kv1.1_ais'] = 0.00492841685426
SC_param['HCN1_ais'] = 0.00099184971498
SC_param['leak_ais'] = 0.000008
SC_param['km_ais'] = 7.513731954E-05

#Axon

SC_param['Nav1.6_axon'] =  0.00835931586458
SC_param['Kv3.4_axon'] = 0.01153520393521
SC_param['Kv1.1_axon'] = 0.00271359229578
SC_param['HCN1_axon'] = 0.00070017344082
SC_param['leak_axon'] = 0.000008

#gap
#SC_param['apcond'] = 1000 #100 #- 0 is for no gaps and 50 is the current setting.
