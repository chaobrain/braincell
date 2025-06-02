#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _Cav12_reg(void);
extern void _Cav13_reg(void);
extern void _Cav2_3_reg(void);
extern void _Cav3_1_reg(void);
extern void _cdp5StCmod_reg(void);
extern void _GOLGI_Ampa_mossy_det_vi_reg(void);
extern void _GOLGI_Ampa_pf_aa_det_vi_reg(void);
extern void _GRC_CA_reg(void);
extern void _GRC_KM_reg(void);
extern void _Hcn1_reg(void);
extern void _Hcn2_reg(void);
extern void _Kca11_reg(void);
extern void _Kca22_reg(void);
extern void _Kca31_reg(void);
extern void _Kv11_reg(void);
extern void _Kv34_reg(void);
extern void _Kv43_reg(void);
extern void _Leak_reg(void);
extern void _Nav16_reg(void);
extern void _PC_NMDA_NR2B_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Cav12.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Cav13.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Cav2_3.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Cav3_1.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/cdp5StCmod.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/GOLGI_Ampa_mossy_det_vi.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/GOLGI_Ampa_pf_aa_det_vi.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/GRC_CA.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/GRC_KM.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Hcn1.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Hcn2.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca11.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca22.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca31.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kv11.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kv34.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kv43.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Leak.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Nav16.mod\"");
    fprintf(stderr, " \"/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/PC_NMDA_NR2B.mod\"");
    fprintf(stderr, "\n");
  }
  _Cav12_reg();
  _Cav13_reg();
  _Cav2_3_reg();
  _Cav3_1_reg();
  _cdp5StCmod_reg();
  _GOLGI_Ampa_mossy_det_vi_reg();
  _GOLGI_Ampa_pf_aa_det_vi_reg();
  _GRC_CA_reg();
  _GRC_KM_reg();
  _Hcn1_reg();
  _Hcn2_reg();
  _Kca11_reg();
  _Kca22_reg();
  _Kca31_reg();
  _Kv11_reg();
  _Kv34_reg();
  _Kv43_reg();
  _Leak_reg();
  _Nav16_reg();
  _PC_NMDA_NR2B_reg();
}

#if defined(__cplusplus)
}
#endif
