/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__cdp5StCmod
#define _nrn_initial _nrn_initial__cdp5StCmod
#define nrn_cur _nrn_cur__cdp5StCmod
#define _nrn_current _nrn_current__cdp5StCmod
#define nrn_jacob _nrn_jacob__cdp5StCmod
#define nrn_state _nrn_state__cdp5StCmod
#define _net_receive _net_receive__cdp5StCmod 
#define factors factors__cdp5StCmod 
#define state state__cdp5StCmod 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define Nannuli _p[0]
#define Nannuli_columnindex 0
#define Buffnull2 _p[1]
#define Buffnull2_columnindex 1
#define rf3 _p[2]
#define rf3_columnindex 2
#define rf4 _p[3]
#define rf4_columnindex 3
#define TotalPump _p[4]
#define TotalPump_columnindex 4
#define ica_pmp _p[5]
#define ica_pmp_columnindex 5
#define vrat _p[6]
#define vrat_columnindex 6
#define icazz _p[7]
#define icazz_columnindex 7
#define ca _p[8]
#define ca_columnindex 8
#define mg _p[9]
#define mg_columnindex 9
#define Buff1 _p[10]
#define Buff1_columnindex 10
#define Buff1_ca _p[11]
#define Buff1_ca_columnindex 11
#define Buff2 _p[12]
#define Buff2_columnindex 12
#define Buff2_ca _p[13]
#define Buff2_ca_columnindex 13
#define BTC _p[14]
#define BTC_columnindex 14
#define BTC_ca _p[15]
#define BTC_ca_columnindex 15
#define DMNPE _p[16]
#define DMNPE_columnindex 16
#define DMNPE_ca _p[17]
#define DMNPE_ca_columnindex 17
#define PV _p[18]
#define PV_columnindex 18
#define PV_ca _p[19]
#define PV_ca_columnindex 19
#define PV_mg _p[20]
#define PV_mg_columnindex 20
#define CAM0 _p[21]
#define CAM0_columnindex 21
#define CAM1C _p[22]
#define CAM1C_columnindex 22
#define CAM2C _p[23]
#define CAM2C_columnindex 23
#define CAM1N2C _p[24]
#define CAM1N2C_columnindex 24
#define CAM1N _p[25]
#define CAM1N_columnindex 25
#define CAM2N _p[26]
#define CAM2N_columnindex 26
#define CAM2N1C _p[27]
#define CAM2N1C_columnindex 27
#define CAM1C1N _p[28]
#define CAM1C1N_columnindex 28
#define CAM4 _p[29]
#define CAM4_columnindex 29
#define pump _p[30]
#define pump_columnindex 30
#define pumpca _p[31]
#define pumpca_columnindex 31
#define nrvci _p[32]
#define nrvci_columnindex 32
#define ica _p[33]
#define ica_columnindex 33
#define parea _p[34]
#define parea_columnindex 34
#define parea2 _p[35]
#define parea2_columnindex 35
#define cai _p[36]
#define cai_columnindex 36
#define mgi _p[37]
#define mgi_columnindex 37
#define Dca _p[38]
#define Dca_columnindex 38
#define Dmg _p[39]
#define Dmg_columnindex 39
#define DBuff1 _p[40]
#define DBuff1_columnindex 40
#define DBuff1_ca _p[41]
#define DBuff1_ca_columnindex 41
#define DBuff2 _p[42]
#define DBuff2_columnindex 42
#define DBuff2_ca _p[43]
#define DBuff2_ca_columnindex 43
#define DBTC _p[44]
#define DBTC_columnindex 44
#define DBTC_ca _p[45]
#define DBTC_ca_columnindex 45
#define DDMNPE _p[46]
#define DDMNPE_columnindex 46
#define DDMNPE_ca _p[47]
#define DDMNPE_ca_columnindex 47
#define DPV _p[48]
#define DPV_columnindex 48
#define DPV_ca _p[49]
#define DPV_ca_columnindex 49
#define DPV_mg _p[50]
#define DPV_mg_columnindex 50
#define DCAM0 _p[51]
#define DCAM0_columnindex 51
#define DCAM1C _p[52]
#define DCAM1C_columnindex 52
#define DCAM2C _p[53]
#define DCAM2C_columnindex 53
#define DCAM1N2C _p[54]
#define DCAM1N2C_columnindex 54
#define DCAM1N _p[55]
#define DCAM1N_columnindex 55
#define DCAM2N _p[56]
#define DCAM2N_columnindex 56
#define DCAM2N1C _p[57]
#define DCAM2N1C_columnindex 57
#define DCAM1C1N _p[58]
#define DCAM1C1N_columnindex 58
#define DCAM4 _p[59]
#define DCAM4_columnindex 59
#define Dpump _p[60]
#define Dpump_columnindex 60
#define Dpumpca _p[61]
#define Dpumpca_columnindex 61
#define v _p[62]
#define v_columnindex 62
#define _g _p[63]
#define _g_columnindex 63
#define _ion_cao	*_ppvar[0]._pval
#define _ion_cai	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _style_ca	*((int*)_ppvar[3]._pvoid)
#define _ion_nrvci	*_ppvar[4]._pval
#define diam	*_ppvar[5]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_factors(void);
 static void _hoc_kdm(void);
 static void _hoc_kdc(void);
 static void _hoc_ssPVmg(void);
 static void _hoc_ssPVca(void);
 static void _hoc_ssPV(void);
 static void _hoc_ssDMNPEca(void);
 static void _hoc_ssDMNPE(void);
 static void _hoc_ssBTCca(void);
 static void _hoc_ssBTC(void);
 static void _hoc_ssBuff2ca(void);
 static void _hoc_ssBuff2(void);
 static void _hoc_ssBuff1ca(void);
 static void _hoc_ssBuff1(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_cdp5StCmod", _hoc_setdata,
 "factors_cdp5StCmod", _hoc_factors,
 "kdm_cdp5StCmod", _hoc_kdm,
 "kdc_cdp5StCmod", _hoc_kdc,
 "ssPVmg_cdp5StCmod", _hoc_ssPVmg,
 "ssPVca_cdp5StCmod", _hoc_ssPVca,
 "ssPV_cdp5StCmod", _hoc_ssPV,
 "ssDMNPEca_cdp5StCmod", _hoc_ssDMNPEca,
 "ssDMNPE_cdp5StCmod", _hoc_ssDMNPE,
 "ssBTCca_cdp5StCmod", _hoc_ssBTCca,
 "ssBTC_cdp5StCmod", _hoc_ssBTC,
 "ssBuff2ca_cdp5StCmod", _hoc_ssBuff2ca,
 "ssBuff2_cdp5StCmod", _hoc_ssBuff2,
 "ssBuff1ca_cdp5StCmod", _hoc_ssBuff1ca,
 "ssBuff1_cdp5StCmod", _hoc_ssBuff1,
 0, 0
};
#define kdm kdm_cdp5StCmod
#define kdc kdc_cdp5StCmod
#define ssPVmg ssPVmg_cdp5StCmod
#define ssPVca ssPVca_cdp5StCmod
#define ssPV ssPV_cdp5StCmod
#define ssDMNPEca ssDMNPEca_cdp5StCmod
#define ssDMNPE ssDMNPE_cdp5StCmod
#define ssBTCca ssBTCca_cdp5StCmod
#define ssBTC ssBTC_cdp5StCmod
#define ssBuff2ca ssBuff2ca_cdp5StCmod
#define ssBuff2 ssBuff2_cdp5StCmod
#define ssBuff1ca ssBuff1ca_cdp5StCmod
#define ssBuff1 ssBuff1_cdp5StCmod
 extern double kdm( _threadargsproto_ );
 extern double kdc( _threadargsproto_ );
 extern double ssPVmg( _threadargsprotocomma_ double , double );
 extern double ssPVca( _threadargsprotocomma_ double , double );
 extern double ssPV( _threadargsprotocomma_ double , double );
 extern double ssDMNPEca( _threadargsproto_ );
 extern double ssDMNPE( _threadargsproto_ );
 extern double ssBTCca( _threadargsproto_ );
 extern double ssBTC( _threadargsproto_ );
 extern double ssBuff2ca( _threadargsproto_ );
 extern double ssBuff2( _threadargsproto_ );
 extern double ssBuff1ca( _threadargsproto_ );
 extern double ssBuff1( _threadargsproto_ );
 #define _zfactors_done _thread[2]._pval[0]
 #define _zdsq _thread[2]._pval[1]
 #define _zdsqvol _thread[2]._pval[2]
 /* declare global and static user variables */
#define BTCnull BTCnull_cdp5StCmod
 double BTCnull = 0;
#define Buffnull1 Buffnull1_cdp5StCmod
 double Buffnull1 = 0;
#define CAM_start CAM_start_cdp5StCmod
 double CAM_start = 0.03;
#define DMNPEnull DMNPEnull_cdp5StCmod
 double DMNPEnull = 0;
#define K2Non K2Non_cdp5StCmod
 double K2Non = 175;
#define K2Noff K2Noff_cdp5StCmod
 double K2Noff = 0.75;
#define Kd2N Kd2N_cdp5StCmod
 double Kd2N = 0.00615;
#define K1Non K1Non_cdp5StCmod
 double K1Non = 142.5;
#define K1Noff K1Noff_cdp5StCmod
 double K1Noff = 2.5;
#define Kd1N Kd1N_cdp5StCmod
 double Kd1N = 0.0275;
#define K2Con K2Con_cdp5StCmod
 double K2Con = 15;
#define K2Coff K2Coff_cdp5StCmod
 double K2Coff = 0.00925;
#define Kd2C Kd2C_cdp5StCmod
 double Kd2C = 0.00105;
#define K1Con K1Con_cdp5StCmod
 double K1Con = 5.4;
#define K1Coff K1Coff_cdp5StCmod
 double K1Coff = 0.04;
#define Kd1C Kd1C_cdp5StCmod
 double Kd1C = 0.00965;
#define PVnull PVnull_cdp5StCmod
 double PVnull = 0.08;
#define b2 b2_cdp5StCmod
 double b2 = 0.08;
#define b1 b1_cdp5StCmod
 double b1 = 5.33;
#define c2 c2_cdp5StCmod
 double c2 = 0.000107;
#define c1 c1_cdp5StCmod
 double c1 = 5.63;
#define cainull cainull_cdp5StCmod
 double cainull = 4.5e-05;
#define kpmp3 kpmp3_cdp5StCmod
 double kpmp3 = 7.255e-05;
#define kpmp2 kpmp2_cdp5StCmod
 double kpmp2 = 1.75e-05;
#define kpmp1 kpmp1_cdp5StCmod
 double kpmp1 = 0.003;
#define m2 m2_cdp5StCmod
 double m2 = 0.00095;
#define m1 m1_cdp5StCmod
 double m1 = 107;
#define mginull mginull_cdp5StCmod
 double mginull = 0.59;
#define p2 p2_cdp5StCmod
 double p2 = 0.025;
#define p1 p1_cdp5StCmod
 double p1 = 0.8;
#define rf2 rf2_cdp5StCmod
 double rf2 = 0.0397469;
#define rf1 rf1_cdp5StCmod
 double rf1 = 0.0134329;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "cainull_cdp5StCmod", "mM",
 "mginull_cdp5StCmod", "mM",
 "Buffnull1_cdp5StCmod", "mM",
 "rf1_cdp5StCmod", "/ms",
 "rf2_cdp5StCmod", "/ms",
 "BTCnull_cdp5StCmod", "mM",
 "b1_cdp5StCmod", "/ms",
 "b2_cdp5StCmod", "/ms",
 "DMNPEnull_cdp5StCmod", "mM",
 "c1_cdp5StCmod", "/ms",
 "c2_cdp5StCmod", "/ms",
 "PVnull_cdp5StCmod", "mM",
 "m1_cdp5StCmod", "/ms",
 "m2_cdp5StCmod", "/ms",
 "p1_cdp5StCmod", "/ms",
 "p2_cdp5StCmod", "/ms",
 "CAM_start_cdp5StCmod", "mM",
 "Kd1C_cdp5StCmod", "mM",
 "K1Coff_cdp5StCmod", "/ms",
 "K1Con_cdp5StCmod", "/mM",
 "Kd2C_cdp5StCmod", "mM",
 "K2Coff_cdp5StCmod", "/ms",
 "K2Con_cdp5StCmod", "/mM",
 "Kd1N_cdp5StCmod", "uM",
 "K1Noff_cdp5StCmod", "/ms",
 "K1Non_cdp5StCmod", "/mM",
 "Kd2N_cdp5StCmod", "mM",
 "K2Noff_cdp5StCmod", "/ms",
 "K2Non_cdp5StCmod", "/mM",
 "kpmp1_cdp5StCmod", "/mM-ms",
 "kpmp2_cdp5StCmod", "/ms",
 "kpmp3_cdp5StCmod", "/ms",
 "Nannuli_cdp5StCmod", "1",
 "Buffnull2_cdp5StCmod", "mM",
 "rf3_cdp5StCmod", "/ms",
 "rf4_cdp5StCmod", "/ms",
 "TotalPump_cdp5StCmod", "mol/cm2",
 "ca_cdp5StCmod", "mM",
 "mg_cdp5StCmod", "mM",
 "Buff1_cdp5StCmod", "mM",
 "Buff1_ca_cdp5StCmod", "mM",
 "Buff2_cdp5StCmod", "mM",
 "Buff2_ca_cdp5StCmod", "mM",
 "BTC_cdp5StCmod", "mM",
 "BTC_ca_cdp5StCmod", "mM",
 "DMNPE_cdp5StCmod", "mM",
 "DMNPE_ca_cdp5StCmod", "mM",
 "PV_cdp5StCmod", "mM",
 "PV_ca_cdp5StCmod", "mM",
 "PV_mg_cdp5StCmod", "mM",
 "CAM0_cdp5StCmod", "mM",
 "CAM1C_cdp5StCmod", "mM",
 "CAM2C_cdp5StCmod", "mM",
 "CAM1N2C_cdp5StCmod", "mM",
 "CAM1N_cdp5StCmod", "mM",
 "CAM2N_cdp5StCmod", "mM",
 "CAM2N1C_cdp5StCmod", "mM",
 "CAM1C1N_cdp5StCmod", "mM",
 "CAM4_cdp5StCmod", "mM",
 "pump_cdp5StCmod", "mol/cm2",
 "pumpca_cdp5StCmod", "mol/cm2",
 "ica_pmp_cdp5StCmod", "mA/cm2",
 "vrat_cdp5StCmod", "1",
 "icazz_cdp5StCmod", "nA",
 0,0
};
 static double BTC_ca0 = 0;
 static double BTC0 = 0;
 static double Buff2_ca0 = 0;
 static double Buff20 = 0;
 static double Buff1_ca0 = 0;
 static double Buff10 = 0;
 static double CAM40 = 0;
 static double CAM1C1N0 = 0;
 static double CAM2N1C0 = 0;
 static double CAM2N0 = 0;
 static double CAM1N0 = 0;
 static double CAM1N2C0 = 0;
 static double CAM2C0 = 0;
 static double CAM1C0 = 0;
 static double CAM00 = 0;
 static double DMNPE_ca0 = 0;
 static double DMNPE0 = 0;
 static double PV_mg0 = 0;
 static double PV_ca0 = 0;
 static double PV0 = 0;
 static double ca0 = 0;
 static double delta_t = 0.01;
 static double mg0 = 0;
 static double pumpca0 = 0;
 static double pump0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "cainull_cdp5StCmod", &cainull_cdp5StCmod,
 "mginull_cdp5StCmod", &mginull_cdp5StCmod,
 "Buffnull1_cdp5StCmod", &Buffnull1_cdp5StCmod,
 "rf1_cdp5StCmod", &rf1_cdp5StCmod,
 "rf2_cdp5StCmod", &rf2_cdp5StCmod,
 "BTCnull_cdp5StCmod", &BTCnull_cdp5StCmod,
 "b1_cdp5StCmod", &b1_cdp5StCmod,
 "b2_cdp5StCmod", &b2_cdp5StCmod,
 "DMNPEnull_cdp5StCmod", &DMNPEnull_cdp5StCmod,
 "c1_cdp5StCmod", &c1_cdp5StCmod,
 "c2_cdp5StCmod", &c2_cdp5StCmod,
 "PVnull_cdp5StCmod", &PVnull_cdp5StCmod,
 "m1_cdp5StCmod", &m1_cdp5StCmod,
 "m2_cdp5StCmod", &m2_cdp5StCmod,
 "p1_cdp5StCmod", &p1_cdp5StCmod,
 "p2_cdp5StCmod", &p2_cdp5StCmod,
 "CAM_start_cdp5StCmod", &CAM_start_cdp5StCmod,
 "Kd1C_cdp5StCmod", &Kd1C_cdp5StCmod,
 "K1Coff_cdp5StCmod", &K1Coff_cdp5StCmod,
 "K1Con_cdp5StCmod", &K1Con_cdp5StCmod,
 "Kd2C_cdp5StCmod", &Kd2C_cdp5StCmod,
 "K2Coff_cdp5StCmod", &K2Coff_cdp5StCmod,
 "K2Con_cdp5StCmod", &K2Con_cdp5StCmod,
 "Kd1N_cdp5StCmod", &Kd1N_cdp5StCmod,
 "K1Noff_cdp5StCmod", &K1Noff_cdp5StCmod,
 "K1Non_cdp5StCmod", &K1Non_cdp5StCmod,
 "Kd2N_cdp5StCmod", &Kd2N_cdp5StCmod,
 "K2Noff_cdp5StCmod", &K2Noff_cdp5StCmod,
 "K2Non_cdp5StCmod", &K2Non_cdp5StCmod,
 "kpmp1_cdp5StCmod", &kpmp1_cdp5StCmod,
 "kpmp2_cdp5StCmod", &kpmp2_cdp5StCmod,
 "kpmp3_cdp5StCmod", &kpmp3_cdp5StCmod,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_synonym(int, double**, Datum**);
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"cdp5StCmod",
 "Nannuli_cdp5StCmod",
 "Buffnull2_cdp5StCmod",
 "rf3_cdp5StCmod",
 "rf4_cdp5StCmod",
 "TotalPump_cdp5StCmod",
 0,
 "ica_pmp_cdp5StCmod",
 "vrat_cdp5StCmod",
 "icazz_cdp5StCmod",
 0,
 "ca_cdp5StCmod",
 "mg_cdp5StCmod",
 "Buff1_cdp5StCmod",
 "Buff1_ca_cdp5StCmod",
 "Buff2_cdp5StCmod",
 "Buff2_ca_cdp5StCmod",
 "BTC_cdp5StCmod",
 "BTC_ca_cdp5StCmod",
 "DMNPE_cdp5StCmod",
 "DMNPE_ca_cdp5StCmod",
 "PV_cdp5StCmod",
 "PV_ca_cdp5StCmod",
 "PV_mg_cdp5StCmod",
 "CAM0_cdp5StCmod",
 "CAM1C_cdp5StCmod",
 "CAM2C_cdp5StCmod",
 "CAM1N2C_cdp5StCmod",
 "CAM1N_cdp5StCmod",
 "CAM2N_cdp5StCmod",
 "CAM2N1C_cdp5StCmod",
 "CAM1C1N_cdp5StCmod",
 "CAM4_cdp5StCmod",
 "pump_cdp5StCmod",
 "pumpca_cdp5StCmod",
 0,
 0};
 static Symbol* _morphology_sym;
 static Symbol* _ca_sym;
 static Symbol* _nrvc_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 64, _prop);
 	/*initialize range parameters*/
 	Nannuli = 10.9495;
 	Buffnull2 = 60.9091;
 	rf3 = 0.1435;
 	rf4 = 0.0014;
 	TotalPump = 1e-09;
 	_prop->param = _p;
 	_prop->param_size = 64;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_morphology_sym);
 	_ppvar[5]._pval = &prop_ion->param[0]; /* diam */
 prop_ion = need_memb(_ca_sym);
 nrn_check_conc_write(_prop, prop_ion, 1);
 nrn_promote(prop_ion, 3, 0);
 	_ppvar[0]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[1]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pvoid = (void*)(&(prop_ion->dparam[0]._i)); /* iontype for ca */
 prop_ion = need_memb(_nrvc_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[4]._pval = &prop_ion->param[1]; /* nrvci */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 "ca_cdp5StCmod", 0.001,
 "mg_cdp5StCmod", 1e-06,
 "pump_cdp5StCmod", 1e-15,
 "pumpca_cdp5StCmod", 1e-15,
 0,0
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _cdp5StCmod_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("nrvc", 1.0);
 	_morphology_sym = hoc_lookup("morphology");
 	_ca_sym = hoc_lookup("ca_ion");
 	_nrvc_sym = hoc_lookup("nrvc_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 4);
  _extcall_thread = (Datum*)ecalloc(3, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 64, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "#ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "nrvc_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
  hoc_register_dparam_semantics(_mechtype, 5, "diam");
 	nrn_writes_conc(_mechtype, 0);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_synonym(_mechtype, _ode_synonym);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 cdp5StCmod /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/cdp5StCmod.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.34c0c8b92a9b7p+3, 9.64853}; /* 9.64853321233100125 */
 
#define PI _nrnunit_PI[_nrnunit_use_legacy_]
static double _nrnunit_PI[2] = {0x1.921fb54442d18p+1, 3.14159}; /* 3.14159265358979312 */
 static double cao = 2;
 /*Top LOCAL _zfactors_done */
 /*Top LOCAL _zdsq , _zdsqvol */
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int factors(_threadargsproto_);
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  0
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[24], _dlist1[24]; static double *_temp1;
 static int state();
 
static int  factors ( _threadargsproto_ ) {
   double _lr , _ldr2 ;
 _lr = 1.0 / 2.0 ;
   _ldr2 = _lr / ( Nannuli - 1.0 ) / 2.0 ;
   vrat = PI * ( _lr - _ldr2 / 2.0 ) * 2.0 * _ldr2 ;
   _lr = _lr - _ldr2 ;
    return 0; }
 
static void _hoc_factors(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 factors ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
static int state (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<24;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
}  
_RHS1(1) *= ( diam * diam * vrat) ;
_MATELM1(1, 1) *= ( diam * diam * vrat); 
_RHS1(2) *= ( diam * diam * vrat) ;
_MATELM1(2, 2) *= ( diam * diam * vrat); 
_RHS1(3) *= ( diam * diam * vrat) ;
_MATELM1(3, 3) *= ( diam * diam * vrat); 
_RHS1(4) *= ( diam * diam * vrat) ;
_MATELM1(4, 4) *= ( diam * diam * vrat); 
_RHS1(5) *= ( diam * diam * vrat) ;
_MATELM1(5, 5) *= ( diam * diam * vrat); 
_RHS1(6) *= ( diam * diam * vrat) ;
_MATELM1(6, 6) *= ( diam * diam * vrat); 
_RHS1(16) *= ( diam * diam * vrat) ;
_MATELM1(16, 16) *= ( diam * diam * vrat); 
_RHS1(17) *= ( diam * diam * vrat) ;
_MATELM1(17, 17) *= ( diam * diam * vrat); 
_RHS1(18) *= ( diam * diam * vrat) ;
_MATELM1(18, 18) *= ( diam * diam * vrat); 
_RHS1(19) *= ( diam * diam * vrat) ;
_MATELM1(19, 19) *= ( diam * diam * vrat); 
_RHS1(20) *= ( diam * diam * vrat) ;
_MATELM1(20, 20) *= ( diam * diam * vrat); 
_RHS1(21) *= ( diam * diam * vrat) ;
_MATELM1(21, 21) *= ( diam * diam * vrat); 
_RHS1(22) *= ( diam * diam * vrat) ;
_MATELM1(22, 22) *= ( diam * diam * vrat); 
_RHS1(23) *= ( ( 1e10 ) * parea) ;
_MATELM1(23, 23) *= ( ( 1e10 ) * parea);  }
 /* COMPARTMENT diam * diam * vrat {
     ca mg Buff1 Buff1_ca Buff2 Buff2_ca BTC BTC_ca DMNPE DMNPE_ca PV PV_ca PV_mg }
   */
 /* COMPARTMENT ( 1e10 ) * parea {
     pump pumpca }
   */
 /* ~ ca + pump <-> pumpca ( kpmp1 * parea * ( 1e10 ) , kpmp2 * parea * ( 1e10 ) )*/
 f_flux =  kpmp1 * parea * ( 1e10 ) * pump * ca ;
 b_flux =  kpmp2 * parea * ( 1e10 ) * pumpca ;
 _RHS1( 23) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 
 _term =  kpmp1 * parea * ( 1e10 ) * ca ;
 _MATELM1( 23 ,23)  += _term;
 _MATELM1( 21 ,23)  += _term;
 _term =  kpmp1 * parea * ( 1e10 ) * pump ;
 _MATELM1( 23 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _term =  kpmp2 * parea * ( 1e10 ) ;
 _MATELM1( 23 ,0)  -= _term;
 _MATELM1( 21 ,0)  -= _term;
 /*REACTION*/
  /* ~ pumpca <-> pump ( kpmp3 * parea * ( 1e10 ) , 0.0 )*/
 f_flux =  kpmp3 * parea * ( 1e10 ) * pumpca ;
 b_flux =  0.0 * pump ;
 _RHS1( 23) += (f_flux - b_flux);
 
 _term =  kpmp3 * parea * ( 1e10 ) ;
 _MATELM1( 23 ,0)  -= _term;
 _term =  0.0 ;
 _MATELM1( 23 ,23)  += _term;
 /*REACTION*/
   /* pump + pumpca = TotalPump * parea * ( 1e10 ) */
 _RHS1(0) =  TotalPump * parea * ( 1e10 );
 _MATELM1(0, 0) = 1 * ( ( 1e10 ) * parea);
 _RHS1(0) -= pumpca * ( ( 1e10 ) * parea) ;
 _MATELM1(0, 23) = 1 * ( ( 1e10 ) * parea);
 _RHS1(0) -= pump * ( ( 1e10 ) * parea) ;
 /*CONSERVATION*/
 ica_pmp = 2.0 * FARADAY * ( f_flux - b_flux ) / parea ;
   /* ~ ca < < ( - ica * PI * diam / ( 2.0 * FARADAY ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 21) += (b_flux =   ( - ica * PI * diam / ( 2.0 * FARADAY ) ) );
 /*FLUX*/
  _zdsq = diam * diam ;
   _zdsqvol = _zdsq * vrat ;
   /* ~ ca + Buff1 <-> Buff1_ca ( rf1 * _zdsqvol , rf2 * _zdsqvol )*/
 f_flux =  rf1 * _zdsqvol * Buff1 * ca ;
 b_flux =  rf2 * _zdsqvol * Buff1_ca ;
 _RHS1( 6) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 5) += (f_flux - b_flux);
 
 _term =  rf1 * _zdsqvol * ca ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 21 ,6)  += _term;
 _MATELM1( 5 ,6)  -= _term;
 _term =  rf1 * _zdsqvol * Buff1 ;
 _MATELM1( 6 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 5 ,21)  -= _term;
 _term =  rf2 * _zdsqvol ;
 _MATELM1( 6 ,5)  -= _term;
 _MATELM1( 21 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ ca + Buff2 <-> Buff2_ca ( rf3 * _zdsqvol , rf4 * _zdsqvol )*/
 f_flux =  rf3 * _zdsqvol * Buff2 * ca ;
 b_flux =  rf4 * _zdsqvol * Buff2_ca ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  rf3 * _zdsqvol * ca ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 21 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  rf3 * _zdsqvol * Buff2 ;
 _MATELM1( 4 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 3 ,21)  -= _term;
 _term =  rf4 * _zdsqvol ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 21 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ ca + BTC <-> BTC_ca ( b1 * _zdsqvol , b2 * _zdsqvol )*/
 f_flux =  b1 * _zdsqvol * BTC * ca ;
 b_flux =  b2 * _zdsqvol * BTC_ca ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  b1 * _zdsqvol * ca ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 21 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  b1 * _zdsqvol * BTC ;
 _MATELM1( 2 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 1 ,21)  -= _term;
 _term =  b2 * _zdsqvol ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 21 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ ca + DMNPE <-> DMNPE_ca ( c1 * _zdsqvol , c2 * _zdsqvol )*/
 f_flux =  c1 * _zdsqvol * DMNPE * ca ;
 b_flux =  c2 * _zdsqvol * DMNPE_ca ;
 _RHS1( 17) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 16) += (f_flux - b_flux);
 
 _term =  c1 * _zdsqvol * ca ;
 _MATELM1( 17 ,17)  += _term;
 _MATELM1( 21 ,17)  += _term;
 _MATELM1( 16 ,17)  -= _term;
 _term =  c1 * _zdsqvol * DMNPE ;
 _MATELM1( 17 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 16 ,21)  -= _term;
 _term =  c2 * _zdsqvol ;
 _MATELM1( 17 ,16)  -= _term;
 _MATELM1( 21 ,16)  -= _term;
 _MATELM1( 16 ,16)  += _term;
 /*REACTION*/
  /* ~ ca + PV <-> PV_ca ( m1 * _zdsqvol , m2 * _zdsqvol )*/
 f_flux =  m1 * _zdsqvol * PV * ca ;
 b_flux =  m2 * _zdsqvol * PV_ca ;
 _RHS1( 20) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 19) += (f_flux - b_flux);
 
 _term =  m1 * _zdsqvol * ca ;
 _MATELM1( 20 ,20)  += _term;
 _MATELM1( 21 ,20)  += _term;
 _MATELM1( 19 ,20)  -= _term;
 _term =  m1 * _zdsqvol * PV ;
 _MATELM1( 20 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 19 ,21)  -= _term;
 _term =  m2 * _zdsqvol ;
 _MATELM1( 20 ,19)  -= _term;
 _MATELM1( 21 ,19)  -= _term;
 _MATELM1( 19 ,19)  += _term;
 /*REACTION*/
  /* ~ mg + PV <-> PV_mg ( p1 * _zdsqvol , p2 * _zdsqvol )*/
 f_flux =  p1 * _zdsqvol * PV * mg ;
 b_flux =  p2 * _zdsqvol * PV_mg ;
 _RHS1( 20) -= (f_flux - b_flux);
 _RHS1( 22) -= (f_flux - b_flux);
 _RHS1( 18) += (f_flux - b_flux);
 
 _term =  p1 * _zdsqvol * mg ;
 _MATELM1( 20 ,20)  += _term;
 _MATELM1( 22 ,20)  += _term;
 _MATELM1( 18 ,20)  -= _term;
 _term =  p1 * _zdsqvol * PV ;
 _MATELM1( 20 ,22)  += _term;
 _MATELM1( 22 ,22)  += _term;
 _MATELM1( 18 ,22)  -= _term;
 _term =  p2 * _zdsqvol ;
 _MATELM1( 20 ,18)  -= _term;
 _MATELM1( 22 ,18)  -= _term;
 _MATELM1( 18 ,18)  += _term;
 /*REACTION*/
  /* ~ ca + CAM0 <-> CAM1C ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 f_flux =  K1Con * _zdsqvol * CAM0 * ca ;
 b_flux =  K1Coff * _zdsqvol * CAM1C ;
 _RHS1( 15) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 14) += (f_flux - b_flux);
 
 _term =  K1Con * _zdsqvol * ca ;
 _MATELM1( 15 ,15)  += _term;
 _MATELM1( 21 ,15)  += _term;
 _MATELM1( 14 ,15)  -= _term;
 _term =  K1Con * _zdsqvol * CAM0 ;
 _MATELM1( 15 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 14 ,21)  -= _term;
 _term =  K1Coff * _zdsqvol ;
 _MATELM1( 15 ,14)  -= _term;
 _MATELM1( 21 ,14)  -= _term;
 _MATELM1( 14 ,14)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C <-> CAM2C ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 f_flux =  K2Con * _zdsqvol * CAM1C * ca ;
 b_flux =  K2Coff * _zdsqvol * CAM2C ;
 _RHS1( 14) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 13) += (f_flux - b_flux);
 
 _term =  K2Con * _zdsqvol * ca ;
 _MATELM1( 14 ,14)  += _term;
 _MATELM1( 21 ,14)  += _term;
 _MATELM1( 13 ,14)  -= _term;
 _term =  K2Con * _zdsqvol * CAM1C ;
 _MATELM1( 14 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 13 ,21)  -= _term;
 _term =  K2Coff * _zdsqvol ;
 _MATELM1( 14 ,13)  -= _term;
 _MATELM1( 21 ,13)  -= _term;
 _MATELM1( 13 ,13)  += _term;
 /*REACTION*/
  /* ~ ca + CAM2C <-> CAM1N2C ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 f_flux =  K1Non * _zdsqvol * CAM2C * ca ;
 b_flux =  K1Noff * _zdsqvol * CAM1N2C ;
 _RHS1( 13) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 12) += (f_flux - b_flux);
 
 _term =  K1Non * _zdsqvol * ca ;
 _MATELM1( 13 ,13)  += _term;
 _MATELM1( 21 ,13)  += _term;
 _MATELM1( 12 ,13)  -= _term;
 _term =  K1Non * _zdsqvol * CAM2C ;
 _MATELM1( 13 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 12 ,21)  -= _term;
 _term =  K1Noff * _zdsqvol ;
 _MATELM1( 13 ,12)  -= _term;
 _MATELM1( 21 ,12)  -= _term;
 _MATELM1( 12 ,12)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1N2C <-> CAM4 ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 f_flux =  K2Non * _zdsqvol * CAM1N2C * ca ;
 b_flux =  K2Noff * _zdsqvol * CAM4 ;
 _RHS1( 12) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  K2Non * _zdsqvol * ca ;
 _MATELM1( 12 ,12)  += _term;
 _MATELM1( 21 ,12)  += _term;
 _MATELM1( 7 ,12)  -= _term;
 _term =  K2Non * _zdsqvol * CAM1N2C ;
 _MATELM1( 12 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 7 ,21)  -= _term;
 _term =  K2Noff * _zdsqvol ;
 _MATELM1( 12 ,7)  -= _term;
 _MATELM1( 21 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ ca + CAM0 <-> CAM1N ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 f_flux =  K1Non * _zdsqvol * CAM0 * ca ;
 b_flux =  K1Noff * _zdsqvol * CAM1N ;
 _RHS1( 15) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 11) += (f_flux - b_flux);
 
 _term =  K1Non * _zdsqvol * ca ;
 _MATELM1( 15 ,15)  += _term;
 _MATELM1( 21 ,15)  += _term;
 _MATELM1( 11 ,15)  -= _term;
 _term =  K1Non * _zdsqvol * CAM0 ;
 _MATELM1( 15 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 11 ,21)  -= _term;
 _term =  K1Noff * _zdsqvol ;
 _MATELM1( 15 ,11)  -= _term;
 _MATELM1( 21 ,11)  -= _term;
 _MATELM1( 11 ,11)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1N <-> CAM2N ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 f_flux =  K2Non * _zdsqvol * CAM1N * ca ;
 b_flux =  K2Noff * _zdsqvol * CAM2N ;
 _RHS1( 11) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 10) += (f_flux - b_flux);
 
 _term =  K2Non * _zdsqvol * ca ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 21 ,11)  += _term;
 _MATELM1( 10 ,11)  -= _term;
 _term =  K2Non * _zdsqvol * CAM1N ;
 _MATELM1( 11 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 10 ,21)  -= _term;
 _term =  K2Noff * _zdsqvol ;
 _MATELM1( 11 ,10)  -= _term;
 _MATELM1( 21 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  /* ~ ca + CAM2N <-> CAM2N1C ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 f_flux =  K1Con * _zdsqvol * CAM2N * ca ;
 b_flux =  K1Coff * _zdsqvol * CAM2N1C ;
 _RHS1( 10) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 9) += (f_flux - b_flux);
 
 _term =  K1Con * _zdsqvol * ca ;
 _MATELM1( 10 ,10)  += _term;
 _MATELM1( 21 ,10)  += _term;
 _MATELM1( 9 ,10)  -= _term;
 _term =  K1Con * _zdsqvol * CAM2N ;
 _MATELM1( 10 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 9 ,21)  -= _term;
 _term =  K1Coff * _zdsqvol ;
 _MATELM1( 10 ,9)  -= _term;
 _MATELM1( 21 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ ca + CAM2N1C <-> CAM4 ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 f_flux =  K2Con * _zdsqvol * CAM2N1C * ca ;
 b_flux =  K2Coff * _zdsqvol * CAM4 ;
 _RHS1( 9) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  K2Con * _zdsqvol * ca ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 21 ,9)  += _term;
 _MATELM1( 7 ,9)  -= _term;
 _term =  K2Con * _zdsqvol * CAM2N1C ;
 _MATELM1( 9 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 7 ,21)  -= _term;
 _term =  K2Coff * _zdsqvol ;
 _MATELM1( 9 ,7)  -= _term;
 _MATELM1( 21 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C <-> CAM1C1N ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 f_flux =  K1Non * _zdsqvol * CAM1C * ca ;
 b_flux =  K1Noff * _zdsqvol * CAM1C1N ;
 _RHS1( 14) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  K1Non * _zdsqvol * ca ;
 _MATELM1( 14 ,14)  += _term;
 _MATELM1( 21 ,14)  += _term;
 _MATELM1( 8 ,14)  -= _term;
 _term =  K1Non * _zdsqvol * CAM1C ;
 _MATELM1( 14 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 8 ,21)  -= _term;
 _term =  K1Noff * _zdsqvol ;
 _MATELM1( 14 ,8)  -= _term;
 _MATELM1( 21 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1N <-> CAM1C1N ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 f_flux =  K1Con * _zdsqvol * CAM1N * ca ;
 b_flux =  K1Coff * _zdsqvol * CAM1C1N ;
 _RHS1( 11) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  K1Con * _zdsqvol * ca ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 21 ,11)  += _term;
 _MATELM1( 8 ,11)  -= _term;
 _term =  K1Con * _zdsqvol * CAM1N ;
 _MATELM1( 11 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 8 ,21)  -= _term;
 _term =  K1Coff * _zdsqvol ;
 _MATELM1( 11 ,8)  -= _term;
 _MATELM1( 21 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C1N <-> CAM1N2C ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 f_flux =  K2Con * _zdsqvol * CAM1C1N * ca ;
 b_flux =  K2Coff * _zdsqvol * CAM1N2C ;
 _RHS1( 8) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 12) += (f_flux - b_flux);
 
 _term =  K2Con * _zdsqvol * ca ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 21 ,8)  += _term;
 _MATELM1( 12 ,8)  -= _term;
 _term =  K2Con * _zdsqvol * CAM1C1N ;
 _MATELM1( 8 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 12 ,21)  -= _term;
 _term =  K2Coff * _zdsqvol ;
 _MATELM1( 8 ,12)  -= _term;
 _MATELM1( 21 ,12)  -= _term;
 _MATELM1( 12 ,12)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C1N <-> CAM2N1C ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 f_flux =  K2Non * _zdsqvol * CAM1C1N * ca ;
 b_flux =  K2Noff * _zdsqvol * CAM2N1C ;
 _RHS1( 8) -= (f_flux - b_flux);
 _RHS1( 21) -= (f_flux - b_flux);
 _RHS1( 9) += (f_flux - b_flux);
 
 _term =  K2Non * _zdsqvol * ca ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 21 ,8)  += _term;
 _MATELM1( 9 ,8)  -= _term;
 _term =  K2Non * _zdsqvol * CAM1C1N ;
 _MATELM1( 8 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 9 ,21)  -= _term;
 _term =  K2Noff * _zdsqvol ;
 _MATELM1( 8 ,9)  -= _term;
 _MATELM1( 21 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  cai = ca ;
   mgi = mg ;
   icazz = nrvci ;
     } return _reset;
 }
 
double ssBuff1 ( _threadargsproto_ ) {
   double _lssBuff1;
 _lssBuff1 = Buffnull1 / ( 1.0 + ( ( rf1 / rf2 ) * cainull ) ) ;
   
return _lssBuff1;
 }
 
static void _hoc_ssBuff1(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssBuff1 ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssBuff1ca ( _threadargsproto_ ) {
   double _lssBuff1ca;
 _lssBuff1ca = Buffnull1 / ( 1.0 + ( rf2 / ( rf1 * cainull ) ) ) ;
   
return _lssBuff1ca;
 }
 
static void _hoc_ssBuff1ca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssBuff1ca ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssBuff2 ( _threadargsproto_ ) {
   double _lssBuff2;
 _lssBuff2 = Buffnull2 / ( 1.0 + ( ( rf3 / rf4 ) * cainull ) ) ;
   
return _lssBuff2;
 }
 
static void _hoc_ssBuff2(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssBuff2 ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssBuff2ca ( _threadargsproto_ ) {
   double _lssBuff2ca;
 _lssBuff2ca = Buffnull2 / ( 1.0 + ( rf4 / ( rf3 * cainull ) ) ) ;
   
return _lssBuff2ca;
 }
 
static void _hoc_ssBuff2ca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssBuff2ca ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssBTC ( _threadargsproto_ ) {
   double _lssBTC;
 _lssBTC = BTCnull / ( 1.0 + ( ( b1 / b2 ) * cainull ) ) ;
   
return _lssBTC;
 }
 
static void _hoc_ssBTC(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssBTC ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssBTCca ( _threadargsproto_ ) {
   double _lssBTCca;
 _lssBTCca = BTCnull / ( 1.0 + ( b2 / ( b1 * cainull ) ) ) ;
   
return _lssBTCca;
 }
 
static void _hoc_ssBTCca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssBTCca ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssDMNPE ( _threadargsproto_ ) {
   double _lssDMNPE;
 _lssDMNPE = DMNPEnull / ( 1.0 + ( ( c1 / c2 ) * cainull ) ) ;
   
return _lssDMNPE;
 }
 
static void _hoc_ssDMNPE(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssDMNPE ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssDMNPEca ( _threadargsproto_ ) {
   double _lssDMNPEca;
 _lssDMNPEca = DMNPEnull / ( 1.0 + ( c2 / ( c1 * cainull ) ) ) ;
   
return _lssDMNPEca;
 }
 
static void _hoc_ssDMNPEca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssDMNPEca ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double kdc ( _threadargsproto_ ) {
   double _lkdc;
 _lkdc = ( cainull * m1 ) / m2 ;
   
return _lkdc;
 }
 
static void _hoc_kdc(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  kdc ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double kdm ( _threadargsproto_ ) {
   double _lkdm;
 _lkdm = ( mginull * p1 ) / p2 ;
   
return _lkdm;
 }
 
static void _hoc_kdm(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  kdm ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
double ssPV ( _threadargsprotocomma_ double _lkdc , double _lkdm ) {
   double _lssPV;
 _lssPV = PVnull / ( 1.0 + kdc ( _threadargs_ ) + kdm ( _threadargs_ ) ) ;
   
return _lssPV;
 }
 
static void _hoc_ssPV(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssPV ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double ssPVca ( _threadargsprotocomma_ double _lkdc , double _lkdm ) {
   double _lssPVca;
 _lssPVca = ( PVnull * kdc ( _threadargs_ ) ) / ( 1.0 + kdc ( _threadargs_ ) + kdm ( _threadargs_ ) ) ;
   
return _lssPVca;
 }
 
static void _hoc_ssPVca(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssPVca ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double ssPVmg ( _threadargsprotocomma_ double _lkdc , double _lkdm ) {
   double _lssPVmg;
 _lssPVmg = ( PVnull * kdm ( _threadargs_ ) ) / ( 1.0 + kdc ( _threadargs_ ) + kdm ( _threadargs_ ) ) ;
   
return _lssPVmg;
 }
 
static void _hoc_ssPVmg(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ssPVmg ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<24;_i++) _p[_dlist1[_i]] = 0.0;}
 /* COMPARTMENT diam * diam * vrat {
   ca mg Buff1 Buff1_ca Buff2 Buff2_ca BTC BTC_ca DMNPE DMNPE_ca PV PV_ca PV_mg }
 */
 /* COMPARTMENT ( 1e10 ) * parea {
   pump pumpca }
 */
 /* ~ ca + pump <-> pumpca ( kpmp1 * parea * ( 1e10 ) , kpmp2 * parea * ( 1e10 ) )*/
 f_flux =  kpmp1 * parea * ( 1e10 ) * pump * ca ;
 b_flux =  kpmp2 * parea * ( 1e10 ) * pumpca ;
 Dpump -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 Dpumpca += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ pumpca <-> pump ( kpmp3 * parea * ( 1e10 ) , 0.0 )*/
 f_flux =  kpmp3 * parea * ( 1e10 ) * pumpca ;
 b_flux =  0.0 * pump ;
 Dpumpca -= (f_flux - b_flux);
 Dpump += (f_flux - b_flux);
 
 /*REACTION*/
   /* pump + pumpca = TotalPump * parea * ( 1e10 ) */
 /*CONSERVATION*/
 ica_pmp = 2.0 * FARADAY * ( f_flux - b_flux ) / parea ;
 /* ~ ca < < ( - ica * PI * diam / ( 2.0 * FARADAY ) )*/
 f_flux = b_flux = 0.;
 Dca += (b_flux =   ( - ica * PI * diam / ( 2.0 * FARADAY ) ) );
 /*FLUX*/
  _zdsq = diam * diam ;
 _zdsqvol = _zdsq * vrat ;
 /* ~ ca + Buff1 <-> Buff1_ca ( rf1 * _zdsqvol , rf2 * _zdsqvol )*/
 f_flux =  rf1 * _zdsqvol * Buff1 * ca ;
 b_flux =  rf2 * _zdsqvol * Buff1_ca ;
 DBuff1 -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DBuff1_ca += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + Buff2 <-> Buff2_ca ( rf3 * _zdsqvol , rf4 * _zdsqvol )*/
 f_flux =  rf3 * _zdsqvol * Buff2 * ca ;
 b_flux =  rf4 * _zdsqvol * Buff2_ca ;
 DBuff2 -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DBuff2_ca += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + BTC <-> BTC_ca ( b1 * _zdsqvol , b2 * _zdsqvol )*/
 f_flux =  b1 * _zdsqvol * BTC * ca ;
 b_flux =  b2 * _zdsqvol * BTC_ca ;
 DBTC -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DBTC_ca += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + DMNPE <-> DMNPE_ca ( c1 * _zdsqvol , c2 * _zdsqvol )*/
 f_flux =  c1 * _zdsqvol * DMNPE * ca ;
 b_flux =  c2 * _zdsqvol * DMNPE_ca ;
 DDMNPE -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DDMNPE_ca += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + PV <-> PV_ca ( m1 * _zdsqvol , m2 * _zdsqvol )*/
 f_flux =  m1 * _zdsqvol * PV * ca ;
 b_flux =  m2 * _zdsqvol * PV_ca ;
 DPV -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DPV_ca += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ mg + PV <-> PV_mg ( p1 * _zdsqvol , p2 * _zdsqvol )*/
 f_flux =  p1 * _zdsqvol * PV * mg ;
 b_flux =  p2 * _zdsqvol * PV_mg ;
 DPV -= (f_flux - b_flux);
 Dmg -= (f_flux - b_flux);
 DPV_mg += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM0 <-> CAM1C ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 f_flux =  K1Con * _zdsqvol * CAM0 * ca ;
 b_flux =  K1Coff * _zdsqvol * CAM1C ;
 DCAM0 -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM1C += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1C <-> CAM2C ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 f_flux =  K2Con * _zdsqvol * CAM1C * ca ;
 b_flux =  K2Coff * _zdsqvol * CAM2C ;
 DCAM1C -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM2C += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM2C <-> CAM1N2C ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 f_flux =  K1Non * _zdsqvol * CAM2C * ca ;
 b_flux =  K1Noff * _zdsqvol * CAM1N2C ;
 DCAM2C -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM1N2C += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1N2C <-> CAM4 ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 f_flux =  K2Non * _zdsqvol * CAM1N2C * ca ;
 b_flux =  K2Noff * _zdsqvol * CAM4 ;
 DCAM1N2C -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM0 <-> CAM1N ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 f_flux =  K1Non * _zdsqvol * CAM0 * ca ;
 b_flux =  K1Noff * _zdsqvol * CAM1N ;
 DCAM0 -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM1N += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1N <-> CAM2N ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 f_flux =  K2Non * _zdsqvol * CAM1N * ca ;
 b_flux =  K2Noff * _zdsqvol * CAM2N ;
 DCAM1N -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM2N += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM2N <-> CAM2N1C ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 f_flux =  K1Con * _zdsqvol * CAM2N * ca ;
 b_flux =  K1Coff * _zdsqvol * CAM2N1C ;
 DCAM2N -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM2N1C += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM2N1C <-> CAM4 ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 f_flux =  K2Con * _zdsqvol * CAM2N1C * ca ;
 b_flux =  K2Coff * _zdsqvol * CAM4 ;
 DCAM2N1C -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1C <-> CAM1C1N ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 f_flux =  K1Non * _zdsqvol * CAM1C * ca ;
 b_flux =  K1Noff * _zdsqvol * CAM1C1N ;
 DCAM1C -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM1C1N += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1N <-> CAM1C1N ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 f_flux =  K1Con * _zdsqvol * CAM1N * ca ;
 b_flux =  K1Coff * _zdsqvol * CAM1C1N ;
 DCAM1N -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM1C1N += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1C1N <-> CAM1N2C ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 f_flux =  K2Con * _zdsqvol * CAM1C1N * ca ;
 b_flux =  K2Coff * _zdsqvol * CAM1N2C ;
 DCAM1C1N -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM1N2C += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ ca + CAM1C1N <-> CAM2N1C ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 f_flux =  K2Non * _zdsqvol * CAM1C1N * ca ;
 b_flux =  K2Noff * _zdsqvol * CAM2N1C ;
 DCAM1C1N -= (f_flux - b_flux);
 Dca -= (f_flux - b_flux);
 DCAM2N1C += (f_flux - b_flux);
 
 /*REACTION*/
  cai = ca ;
 mgi = mg ;
 icazz = nrvci ;
 _p[_dlist1[0]] /= ( ( 1e10 ) * parea);
 _p[_dlist1[1]] /= ( diam * diam * vrat);
 _p[_dlist1[2]] /= ( diam * diam * vrat);
 _p[_dlist1[3]] /= ( diam * diam * vrat);
 _p[_dlist1[4]] /= ( diam * diam * vrat);
 _p[_dlist1[5]] /= ( diam * diam * vrat);
 _p[_dlist1[6]] /= ( diam * diam * vrat);
 _p[_dlist1[16]] /= ( diam * diam * vrat);
 _p[_dlist1[17]] /= ( diam * diam * vrat);
 _p[_dlist1[18]] /= ( diam * diam * vrat);
 _p[_dlist1[19]] /= ( diam * diam * vrat);
 _p[_dlist1[20]] /= ( diam * diam * vrat);
 _p[_dlist1[21]] /= ( diam * diam * vrat);
 _p[_dlist1[22]] /= ( diam * diam * vrat);
 _p[_dlist1[23]] /= ( ( 1e10 ) * parea);
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<24;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
}  
_RHS1(0) *= ( ( 1e10 ) * parea) ;
_MATELM1(0, 0) *= ( ( 1e10 ) * parea); 
_RHS1(1) *= ( diam * diam * vrat) ;
_MATELM1(1, 1) *= ( diam * diam * vrat); 
_RHS1(2) *= ( diam * diam * vrat) ;
_MATELM1(2, 2) *= ( diam * diam * vrat); 
_RHS1(3) *= ( diam * diam * vrat) ;
_MATELM1(3, 3) *= ( diam * diam * vrat); 
_RHS1(4) *= ( diam * diam * vrat) ;
_MATELM1(4, 4) *= ( diam * diam * vrat); 
_RHS1(5) *= ( diam * diam * vrat) ;
_MATELM1(5, 5) *= ( diam * diam * vrat); 
_RHS1(6) *= ( diam * diam * vrat) ;
_MATELM1(6, 6) *= ( diam * diam * vrat); 
_RHS1(16) *= ( diam * diam * vrat) ;
_MATELM1(16, 16) *= ( diam * diam * vrat); 
_RHS1(17) *= ( diam * diam * vrat) ;
_MATELM1(17, 17) *= ( diam * diam * vrat); 
_RHS1(18) *= ( diam * diam * vrat) ;
_MATELM1(18, 18) *= ( diam * diam * vrat); 
_RHS1(19) *= ( diam * diam * vrat) ;
_MATELM1(19, 19) *= ( diam * diam * vrat); 
_RHS1(20) *= ( diam * diam * vrat) ;
_MATELM1(20, 20) *= ( diam * diam * vrat); 
_RHS1(21) *= ( diam * diam * vrat) ;
_MATELM1(21, 21) *= ( diam * diam * vrat); 
_RHS1(22) *= ( diam * diam * vrat) ;
_MATELM1(22, 22) *= ( diam * diam * vrat); 
_RHS1(23) *= ( ( 1e10 ) * parea) ;
_MATELM1(23, 23) *= ( ( 1e10 ) * parea);  }
 /* COMPARTMENT diam * diam * vrat {
 ca mg Buff1 Buff1_ca Buff2 Buff2_ca BTC BTC_ca DMNPE DMNPE_ca PV PV_ca PV_mg }
 */
 /* COMPARTMENT ( 1e10 ) * parea {
 pump pumpca }
 */
 /* ~ ca + pump <-> pumpca ( kpmp1 * parea * ( 1e10 ) , kpmp2 * parea * ( 1e10 ) )*/
 _term =  kpmp1 * parea * ( 1e10 ) * ca ;
 _MATELM1( 23 ,23)  += _term;
 _MATELM1( 21 ,23)  += _term;
 _MATELM1( 0 ,23)  -= _term;
 _term =  kpmp1 * parea * ( 1e10 ) * pump ;
 _MATELM1( 23 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 0 ,21)  -= _term;
 _term =  kpmp2 * parea * ( 1e10 ) ;
 _MATELM1( 23 ,0)  -= _term;
 _MATELM1( 21 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ pumpca <-> pump ( kpmp3 * parea * ( 1e10 ) , 0.0 )*/
 _term =  kpmp3 * parea * ( 1e10 ) ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 23 ,0)  -= _term;
 _term =  0.0 ;
 _MATELM1( 0 ,23)  -= _term;
 _MATELM1( 23 ,23)  += _term;
 /* ~ ca < < ( - ica * PI * diam / ( 2.0 * FARADAY ) )*/
 /*FLUX*/
  _zdsq = diam * diam ;
 _zdsqvol = _zdsq * vrat ;
 /* ~ ca + Buff1 <-> Buff1_ca ( rf1 * _zdsqvol , rf2 * _zdsqvol )*/
 _term =  rf1 * _zdsqvol * ca ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 21 ,6)  += _term;
 _MATELM1( 5 ,6)  -= _term;
 _term =  rf1 * _zdsqvol * Buff1 ;
 _MATELM1( 6 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 5 ,21)  -= _term;
 _term =  rf2 * _zdsqvol ;
 _MATELM1( 6 ,5)  -= _term;
 _MATELM1( 21 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ ca + Buff2 <-> Buff2_ca ( rf3 * _zdsqvol , rf4 * _zdsqvol )*/
 _term =  rf3 * _zdsqvol * ca ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 21 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  rf3 * _zdsqvol * Buff2 ;
 _MATELM1( 4 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 3 ,21)  -= _term;
 _term =  rf4 * _zdsqvol ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 21 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ ca + BTC <-> BTC_ca ( b1 * _zdsqvol , b2 * _zdsqvol )*/
 _term =  b1 * _zdsqvol * ca ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 21 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  b1 * _zdsqvol * BTC ;
 _MATELM1( 2 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 1 ,21)  -= _term;
 _term =  b2 * _zdsqvol ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 21 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ ca + DMNPE <-> DMNPE_ca ( c1 * _zdsqvol , c2 * _zdsqvol )*/
 _term =  c1 * _zdsqvol * ca ;
 _MATELM1( 17 ,17)  += _term;
 _MATELM1( 21 ,17)  += _term;
 _MATELM1( 16 ,17)  -= _term;
 _term =  c1 * _zdsqvol * DMNPE ;
 _MATELM1( 17 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 16 ,21)  -= _term;
 _term =  c2 * _zdsqvol ;
 _MATELM1( 17 ,16)  -= _term;
 _MATELM1( 21 ,16)  -= _term;
 _MATELM1( 16 ,16)  += _term;
 /*REACTION*/
  /* ~ ca + PV <-> PV_ca ( m1 * _zdsqvol , m2 * _zdsqvol )*/
 _term =  m1 * _zdsqvol * ca ;
 _MATELM1( 20 ,20)  += _term;
 _MATELM1( 21 ,20)  += _term;
 _MATELM1( 19 ,20)  -= _term;
 _term =  m1 * _zdsqvol * PV ;
 _MATELM1( 20 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 19 ,21)  -= _term;
 _term =  m2 * _zdsqvol ;
 _MATELM1( 20 ,19)  -= _term;
 _MATELM1( 21 ,19)  -= _term;
 _MATELM1( 19 ,19)  += _term;
 /*REACTION*/
  /* ~ mg + PV <-> PV_mg ( p1 * _zdsqvol , p2 * _zdsqvol )*/
 _term =  p1 * _zdsqvol * mg ;
 _MATELM1( 20 ,20)  += _term;
 _MATELM1( 22 ,20)  += _term;
 _MATELM1( 18 ,20)  -= _term;
 _term =  p1 * _zdsqvol * PV ;
 _MATELM1( 20 ,22)  += _term;
 _MATELM1( 22 ,22)  += _term;
 _MATELM1( 18 ,22)  -= _term;
 _term =  p2 * _zdsqvol ;
 _MATELM1( 20 ,18)  -= _term;
 _MATELM1( 22 ,18)  -= _term;
 _MATELM1( 18 ,18)  += _term;
 /*REACTION*/
  /* ~ ca + CAM0 <-> CAM1C ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 _term =  K1Con * _zdsqvol * ca ;
 _MATELM1( 15 ,15)  += _term;
 _MATELM1( 21 ,15)  += _term;
 _MATELM1( 14 ,15)  -= _term;
 _term =  K1Con * _zdsqvol * CAM0 ;
 _MATELM1( 15 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 14 ,21)  -= _term;
 _term =  K1Coff * _zdsqvol ;
 _MATELM1( 15 ,14)  -= _term;
 _MATELM1( 21 ,14)  -= _term;
 _MATELM1( 14 ,14)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C <-> CAM2C ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 _term =  K2Con * _zdsqvol * ca ;
 _MATELM1( 14 ,14)  += _term;
 _MATELM1( 21 ,14)  += _term;
 _MATELM1( 13 ,14)  -= _term;
 _term =  K2Con * _zdsqvol * CAM1C ;
 _MATELM1( 14 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 13 ,21)  -= _term;
 _term =  K2Coff * _zdsqvol ;
 _MATELM1( 14 ,13)  -= _term;
 _MATELM1( 21 ,13)  -= _term;
 _MATELM1( 13 ,13)  += _term;
 /*REACTION*/
  /* ~ ca + CAM2C <-> CAM1N2C ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 _term =  K1Non * _zdsqvol * ca ;
 _MATELM1( 13 ,13)  += _term;
 _MATELM1( 21 ,13)  += _term;
 _MATELM1( 12 ,13)  -= _term;
 _term =  K1Non * _zdsqvol * CAM2C ;
 _MATELM1( 13 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 12 ,21)  -= _term;
 _term =  K1Noff * _zdsqvol ;
 _MATELM1( 13 ,12)  -= _term;
 _MATELM1( 21 ,12)  -= _term;
 _MATELM1( 12 ,12)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1N2C <-> CAM4 ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 _term =  K2Non * _zdsqvol * ca ;
 _MATELM1( 12 ,12)  += _term;
 _MATELM1( 21 ,12)  += _term;
 _MATELM1( 7 ,12)  -= _term;
 _term =  K2Non * _zdsqvol * CAM1N2C ;
 _MATELM1( 12 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 7 ,21)  -= _term;
 _term =  K2Noff * _zdsqvol ;
 _MATELM1( 12 ,7)  -= _term;
 _MATELM1( 21 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ ca + CAM0 <-> CAM1N ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 _term =  K1Non * _zdsqvol * ca ;
 _MATELM1( 15 ,15)  += _term;
 _MATELM1( 21 ,15)  += _term;
 _MATELM1( 11 ,15)  -= _term;
 _term =  K1Non * _zdsqvol * CAM0 ;
 _MATELM1( 15 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 11 ,21)  -= _term;
 _term =  K1Noff * _zdsqvol ;
 _MATELM1( 15 ,11)  -= _term;
 _MATELM1( 21 ,11)  -= _term;
 _MATELM1( 11 ,11)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1N <-> CAM2N ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 _term =  K2Non * _zdsqvol * ca ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 21 ,11)  += _term;
 _MATELM1( 10 ,11)  -= _term;
 _term =  K2Non * _zdsqvol * CAM1N ;
 _MATELM1( 11 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 10 ,21)  -= _term;
 _term =  K2Noff * _zdsqvol ;
 _MATELM1( 11 ,10)  -= _term;
 _MATELM1( 21 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  /* ~ ca + CAM2N <-> CAM2N1C ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 _term =  K1Con * _zdsqvol * ca ;
 _MATELM1( 10 ,10)  += _term;
 _MATELM1( 21 ,10)  += _term;
 _MATELM1( 9 ,10)  -= _term;
 _term =  K1Con * _zdsqvol * CAM2N ;
 _MATELM1( 10 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 9 ,21)  -= _term;
 _term =  K1Coff * _zdsqvol ;
 _MATELM1( 10 ,9)  -= _term;
 _MATELM1( 21 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ ca + CAM2N1C <-> CAM4 ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 _term =  K2Con * _zdsqvol * ca ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 21 ,9)  += _term;
 _MATELM1( 7 ,9)  -= _term;
 _term =  K2Con * _zdsqvol * CAM2N1C ;
 _MATELM1( 9 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 7 ,21)  -= _term;
 _term =  K2Coff * _zdsqvol ;
 _MATELM1( 9 ,7)  -= _term;
 _MATELM1( 21 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C <-> CAM1C1N ( K1Non * _zdsqvol , K1Noff * _zdsqvol )*/
 _term =  K1Non * _zdsqvol * ca ;
 _MATELM1( 14 ,14)  += _term;
 _MATELM1( 21 ,14)  += _term;
 _MATELM1( 8 ,14)  -= _term;
 _term =  K1Non * _zdsqvol * CAM1C ;
 _MATELM1( 14 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 8 ,21)  -= _term;
 _term =  K1Noff * _zdsqvol ;
 _MATELM1( 14 ,8)  -= _term;
 _MATELM1( 21 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1N <-> CAM1C1N ( K1Con * _zdsqvol , K1Coff * _zdsqvol )*/
 _term =  K1Con * _zdsqvol * ca ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 21 ,11)  += _term;
 _MATELM1( 8 ,11)  -= _term;
 _term =  K1Con * _zdsqvol * CAM1N ;
 _MATELM1( 11 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 8 ,21)  -= _term;
 _term =  K1Coff * _zdsqvol ;
 _MATELM1( 11 ,8)  -= _term;
 _MATELM1( 21 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C1N <-> CAM1N2C ( K2Con * _zdsqvol , K2Coff * _zdsqvol )*/
 _term =  K2Con * _zdsqvol * ca ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 21 ,8)  += _term;
 _MATELM1( 12 ,8)  -= _term;
 _term =  K2Con * _zdsqvol * CAM1C1N ;
 _MATELM1( 8 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 12 ,21)  -= _term;
 _term =  K2Coff * _zdsqvol ;
 _MATELM1( 8 ,12)  -= _term;
 _MATELM1( 21 ,12)  -= _term;
 _MATELM1( 12 ,12)  += _term;
 /*REACTION*/
  /* ~ ca + CAM1C1N <-> CAM2N1C ( K2Non * _zdsqvol , K2Noff * _zdsqvol )*/
 _term =  K2Non * _zdsqvol * ca ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 21 ,8)  += _term;
 _MATELM1( 9 ,8)  -= _term;
 _term =  K2Non * _zdsqvol * CAM1C1N ;
 _MATELM1( 8 ,21)  += _term;
 _MATELM1( 21 ,21)  += _term;
 _MATELM1( 9 ,21)  -= _term;
 _term =  K2Noff * _zdsqvol ;
 _MATELM1( 8 ,9)  -= _term;
 _MATELM1( 21 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  cai = ca ;
 mgi = mg ;
 icazz = nrvci ;
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 24;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cao = _ion_cao;
  cai = _ion_cai;
  ica = _ion_ica;
  cai = _ion_cai;
  nrvci = _ion_nrvci;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  _ion_cai = cai;
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 24; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 static void _ode_synonym(int _cnt, double** _pp, Datum** _ppd) { 
	double* _p; Datum* _ppvar;
 	int _i; 
	for (_i=0; _i < _cnt; ++_i) {_p = _pp[_i]; _ppvar = _ppd[_i];
 _ion_cai =  ca ;
 }}
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 24, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cao = _ion_cao;
  cai = _ion_cai;
  ica = _ion_ica;
  cai = _ion_cai;
  nrvci = _ion_nrvci;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[2]._pval = (double*)ecalloc(3, sizeof(double));
 }
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
   free((void*)(_thread[2]._pval));
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_nrvc_sym, _ppvar, 4, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  BTC_ca = BTC_ca0;
  BTC = BTC0;
  Buff2_ca = Buff2_ca0;
  Buff2 = Buff20;
  Buff1_ca = Buff1_ca0;
  Buff1 = Buff10;
  CAM4 = CAM40;
  CAM1C1N = CAM1C1N0;
  CAM2N1C = CAM2N1C0;
  CAM2N = CAM2N0;
  CAM1N = CAM1N0;
  CAM1N2C = CAM1N2C0;
  CAM2C = CAM2C0;
  CAM1C = CAM1C0;
  CAM0 = CAM00;
  DMNPE_ca = DMNPE_ca0;
  DMNPE = DMNPE0;
  PV_mg = PV_mg0;
  PV_ca = PV_ca0;
  PV = PV0;
  ca = ca0;
  mg = mg0;
  pumpca = pumpca0;
  pump = pump0;
 {
   factors ( _threadargs_ ) ;
   ca = cainull ;
   mg = mginull ;
   Buff1 = ssBuff1 ( _threadargs_ ) ;
   Buff1_ca = ssBuff1ca ( _threadargs_ ) ;
   Buff2 = ssBuff2 ( _threadargs_ ) ;
   Buff2_ca = ssBuff2ca ( _threadargs_ ) ;
   BTC = ssBTC ( _threadargs_ ) ;
   BTC_ca = ssBTCca ( _threadargs_ ) ;
   DMNPE = ssDMNPE ( _threadargs_ ) ;
   DMNPE_ca = ssDMNPEca ( _threadargs_ ) ;
   PV = ssPV ( _threadargscomma_ kdc ( _threadargs_ ) , kdm ( _threadargs_ ) ) ;
   PV_ca = ssPVca ( _threadargscomma_ kdc ( _threadargs_ ) , kdm ( _threadargs_ ) ) ;
   PV_mg = ssPVmg ( _threadargscomma_ kdc ( _threadargs_ ) , kdm ( _threadargs_ ) ) ;
   CAM0 = CAM_start ;
   CAM1C = 0.0 ;
   CAM2C = 0.0 ;
   CAM1N2C = 0.0 ;
   CAM1N = 0.0 ;
   CAM2N = 0.0 ;
   CAM2N1C = 0.0 ;
   CAM1C1N = 0.0 ;
   CAM4 = 0.0 ;
   parea = PI * diam ;
   parea2 = PI * ( diam - 0.2 ) ;
   ica = 0.0 ;
   ica_pmp = 0.0 ;
   pump = TotalPump ;
   pumpca = 0.0 ;
   cai = ca ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  cao = _ion_cao;
  cai = _ion_cai;
  ica = _ion_ica;
  cai = _ion_cai;
  nrvci = _ion_nrvci;
 initmodel(_p, _ppvar, _thread, _nt);
  _ion_cai = cai;
  nrn_wrote_conc(_ca_sym, (&(_ion_cai)) - 1, _style_ca);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  cao = _ion_cao;
  cai = _ion_cai;
  ica = _ion_ica;
  cai = _ion_cai;
  nrvci = _ion_nrvci;
 {  sparse_thread(&_thread[_spth1]._pvoid, 24, _slist1, _dlist1, _p, &t, dt, state, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 24; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } {
   }
  _ion_cai = cai;
}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = pumpca_columnindex;  _dlist1[0] = Dpumpca_columnindex;
 _slist1[1] = BTC_ca_columnindex;  _dlist1[1] = DBTC_ca_columnindex;
 _slist1[2] = BTC_columnindex;  _dlist1[2] = DBTC_columnindex;
 _slist1[3] = Buff2_ca_columnindex;  _dlist1[3] = DBuff2_ca_columnindex;
 _slist1[4] = Buff2_columnindex;  _dlist1[4] = DBuff2_columnindex;
 _slist1[5] = Buff1_ca_columnindex;  _dlist1[5] = DBuff1_ca_columnindex;
 _slist1[6] = Buff1_columnindex;  _dlist1[6] = DBuff1_columnindex;
 _slist1[7] = CAM4_columnindex;  _dlist1[7] = DCAM4_columnindex;
 _slist1[8] = CAM1C1N_columnindex;  _dlist1[8] = DCAM1C1N_columnindex;
 _slist1[9] = CAM2N1C_columnindex;  _dlist1[9] = DCAM2N1C_columnindex;
 _slist1[10] = CAM2N_columnindex;  _dlist1[10] = DCAM2N_columnindex;
 _slist1[11] = CAM1N_columnindex;  _dlist1[11] = DCAM1N_columnindex;
 _slist1[12] = CAM1N2C_columnindex;  _dlist1[12] = DCAM1N2C_columnindex;
 _slist1[13] = CAM2C_columnindex;  _dlist1[13] = DCAM2C_columnindex;
 _slist1[14] = CAM1C_columnindex;  _dlist1[14] = DCAM1C_columnindex;
 _slist1[15] = CAM0_columnindex;  _dlist1[15] = DCAM0_columnindex;
 _slist1[16] = DMNPE_ca_columnindex;  _dlist1[16] = DDMNPE_ca_columnindex;
 _slist1[17] = DMNPE_columnindex;  _dlist1[17] = DDMNPE_columnindex;
 _slist1[18] = PV_mg_columnindex;  _dlist1[18] = DPV_mg_columnindex;
 _slist1[19] = PV_ca_columnindex;  _dlist1[19] = DPV_ca_columnindex;
 _slist1[20] = PV_columnindex;  _dlist1[20] = DPV_columnindex;
 _slist1[21] = ca_columnindex;  _dlist1[21] = Dca_columnindex;
 _slist1[22] = mg_columnindex;  _dlist1[22] = Dmg_columnindex;
 _slist1[23] = pump_columnindex;  _dlist1[23] = Dpump_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/cdp5StCmod.mod";
static const char* nmodl_file_text = 
  ": Calcium ion accumulation with endogenous buffers, DCM and pump\n"
  "\n"
  "COMMENT\n"
  "\n"
  "The basic code of Example 9.8 and Example 9.9 from NEURON book was adapted as:\n"
  "\n"
  "1) Extended using parameters from Schmidt et al. 2003.\n"
  "2) Pump rate was tuned according to data from Maeda et al. 1999\n"
  "3) DCM was introduced and tuned to approximate the effect of radial diffusion\n"
  "\n"
  "Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*\n"
  "\n"
  "*Article available as Open Access\n"
  "\n"
  "PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513\n"
  "\n"
  "Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.\n"
  "Contact: Haroon Anwar (anwar@oist.jp)\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "NEURON {\n"
  "  SUFFIX cdp5StCmod\n"
  "  USEION ca READ cao, cai, ica WRITE cai\n"
  "  USEION nrvc READ nrvci VALENCE 1\n"
  "  RANGE ica_pmp\n"
  "  RANGE Nannuli, Buffnull2, rf3, rf4, vrat\n"
  "    RANGE CAM0, CAM1C, CAM2C, CAM1N2C, CAM1N, CAM2N, CAM2N1C, CAM1C1N, CAM4, icazz\n"
  "  RANGE TotalPump\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mol)   = (1)\n"
  "	(molar) = (1/liter)\n"
  "	(mM)    = (millimolar)\n"
  "	(um)    = (micron)\n"
  "	(mA)    = (milliamp)\n"
  "	FARADAY = (faraday)  (10000 coulomb)\n"
  "	PI      = (pi)       (1)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	Nannuli = 10.9495 (1)\n"
  "	celsius (degC)\n"
  "        \n"
  "	cainull = 45e-6 (mM)\n"
  "        mginull =.59    (mM)\n"
  "\n"
  ":	values for a buffer compensating the diffusion\n"
  "\n"
  "	Buffnull1 = 0	(mM)\n"
  "	rf1 = 0.0134329	(/ms mM)\n"
  "	rf2 = 0.0397469	(/ms)\n"
  "\n"
  "	Buffnull2 = 60.9091	(mM)\n"
  "	rf3 = 0.1435	(/ms mM)\n"
  "	rf4 = 0.0014	(/ms)\n"
  "\n"
  ":	values for benzothiazole coumarin (BTC)\n"
  "	BTCnull = 0	(mM)\n"
  "	b1 = 5.33	(/ms mM)\n"
  "	b2 = 0.08	(/ms)\n"
  "\n"
  ":	values for caged compound DMNPE-4\n"
  "	DMNPEnull = 0	(mM)\n"
  "	c1 = 5.63	(/ms mM)\n"
  "	c2 = 0.107e-3	(/ms)\n"
  "\n"
  ":       values for Calbindin (2 high and 2 low affinity binding sites)\n"
  "\n"
  "        :CBnull=	.16             (mM)\n"
  "        :nf1   =43.5           (/ms mM)\n"
  "        :nf2   =3.58e-2        (/ms)\n"
  "        :ns1   =5.5            (/ms mM)\n"
  "        :ns2   =0.26e-2        (/ms)\n"
  "\n"
  ":       values for Parvalbumin\n"
  "\n"
  "        PVnull  = .08           (mM)\n"
  "        m1    = 1.07e2        (/ms mM)\n"
  "        m2    = 9.5e-4                (/ms)\n"
  "        p1    = 0.8           (/ms mM)\n"
  "        p2    = 2.5e-2                (/ms)\n"
  "\n"
  ":	Calmodulin concentration\n"
  "	CAM_start 	= 0.03		(mM) :Pepke 2010\n"
  "\n"
  ": 	Calmodulin Kinetic parameters. The values are the mean between max and min.\n"
  "	:C-lobe\n"
  "	Kd1C = 		0.00965	(mM)						: Kd - Equilibrium binding of 1st Ca2+ to CaM C-terminus \n"
  "	K1Coff = 	0.04	(/ms)						: From 0C to 1C with X ions on N-lobe\n"
  "	K1Con = 	5.4	(/mM ms)					: From 1C to 0C with X ions on N-lobe\n"
  "	Kd2C = 		0.00105	(mM)						: Kd - Equilibrium binding of 2nd Ca2+ to CaM C-terminus\n"
  "	K2Coff = 	0.00925	(/ms)						: From 1C to 2C with X ions on N-lobe\n"
  "	K2Con = 	15	(/mM ms)					: From 2C to 1C with X ions on N-lobe\n"
  "\n"
  "	:N-lobe\n"
  "	Kd1N = 		0.0275	(uM)						: Kd - Equilibrium binding of 1st Ca2+ to CaM N-terminus \n"
  "	K1Noff = 	2.5	(/ms)						: From 0N to 1N with X ions on C-lobe\n"
  "	K1Non = 	142.5	(/mM ms)					: From 1N to 0N with X ions on C-lobe\n"
  "	Kd2N = 		0.00615	(mM)						: Kd - Equilibrium binding of 2nd Ca2+ to CaM N-terminus\n"
  "	K2Noff = 	0.75	(/ms)						: From 1N to 2N with X ions on C-lobe\n"
  "	K2Non = 	175	(/mM ms)					: From 2N to 1N with X ions on C-lobe        \n"
  "\n"
  "\n"
  "  	kpmp1    = 3e-3       (/mM-ms)\n"
  "  	kpmp2    = 1.75e-5   (/ms)\n"
  "  	kpmp3    = 7.255e-5  (/ms)\n"
  "	TotalPump = 1e-9	(mol/cm2)	\n"
  "	nrvci (nA)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	diam      (um)\n"
  "	ica       (mA/cm2)\n"
  "	ica_pmp   (mA/cm2)\n"
  "	parea     (um)     : pump area per unit length\n"
  "	parea2	  (um)\n"
  "	cai       (mM)\n"
  "	mgi	(mM)\n"
  "	vrat	(1)	\n"
  "        icazz (nA)\n"
  "}\n"
  "\n"
  "CONSTANT { cao = 2	(mM) }\n"
  "\n"
  "STATE {\n"
  "	: ca[0] is equivalent to cai\n"
  "	: ca[] are very small, so specify absolute tolerance\n"
  "	: let it be ~1.5 - 2 orders of magnitude smaller than baseline level\n"
  "\n"
  "	ca		(mM)    <1e-3>\n"
  "	mg		(mM)	<1e-6>\n"
  "	\n"
  "	Buff1		(mM)	\n"
  "	Buff1_ca	(mM)\n"
  "\n"
  "	Buff2		(mM)\n"
  "	Buff2_ca	(mM)\n"
  "\n"
  "	BTC		(mM)\n"
  "	BTC_ca		(mM)\n"
  "\n"
  "	DMNPE		(mM)\n"
  "	DMNPE_ca	(mM)	\n"
  "\n"
  "        :CB		(mM)\n"
  "        :CB_f_ca		(mM)\n"
  "        :CB_ca_s		(mM)\n"
  "        :CB_ca_ca	(mM)\n"
  "\n"
  "        PV		(mM)\n"
  "        PV_ca		(mM)\n"
  "        PV_mg		(mM)\n"
  "\n"
  ":State for the Calmodulin      \n"
  "\n"
  "	CAM0		(mM)\n"
  "\n"
  "	:C-lobe mainly\n"
  "	CAM1C		(mM)\n"
  "	CAM2C		(mM)\n"
  "	CAM1N2C		(mM)\n"
  "	\n"
  "	:N-Lobe Mainly\n"
  "	CAM1N		(mM)\n"
  "	CAM2N		(mM)\n"
  "	CAM2N1C		(mM)\n"
  "\n"
  "	:One ion on C-lobe and one on N-lobe\n"
  "	CAM1C1N		(mM)\n"
  "\n"
  "	:CaM complete\n"
  "	CAM4		(mM)	\n"
  "\n"
  "	\n"
  "	pump		(mol/cm2) <1e-15>\n"
  "	pumpca		(mol/cm2) <1e-15>\n"
  "\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD sparse\n"
  "}\n"
  "\n"
  "LOCAL factors_done\n"
  "\n"
  "INITIAL {\n"
  "		factors()\n"
  "\n"
  "		ca = cainull\n"
  "		mg = mginull\n"
  "		\n"
  "		Buff1 = ssBuff1()\n"
  "		Buff1_ca = ssBuff1ca()\n"
  "\n"
  "		Buff2 = ssBuff2()\n"
  "		Buff2_ca = ssBuff2ca()\n"
  "\n"
  "		BTC = ssBTC()\n"
  "		BTC_ca = ssBTCca()		\n"
  "\n"
  "		DMNPE = ssDMNPE()\n"
  "		DMNPE_ca = ssDMNPEca()\n"
  "\n"
  "		:CB = ssCB( kdf(), kds())   \n"
  "	        :CB_f_ca = ssCBfast( kdf(), kds())\n"
  "       	 	:CB_ca_s = ssCBslow( kdf(), kds())\n"
  "        	:CB_ca_ca = ssCBca( kdf(), kds())\n"
  "\n"
  "        	PV = ssPV( kdc(), kdm())\n"
  "        	PV_ca = ssPVca(kdc(), kdm())\n"
  "        	PV_mg = ssPVmg(kdc(), kdm())\n"
  "\n"
  "	:Calmodulin\n"
  "	CAM0	= CAM_start		\n"
  "	CAM1C	= 0\n"
  "	CAM2C	= 0\n"
  "	CAM1N2C = 0\n"
  "	CAM1N   = 0\n"
  "	CAM2N	= 0\n"
  "	CAM2N1C = 0\n"
  "	CAM1C1N = 0\n"
  "	CAM4	= 0\n"
  "		\n"
  "  	parea = PI*diam\n"
  "	parea2 = PI*(diam-0.2)\n"
  "	ica = 0\n"
  "	ica_pmp = 0\n"
  ":	ica_pmp_last = 0\n"
  "	pump = TotalPump\n"
  "	pumpca = 0\n"
  "	\n"
  "	cai = ca\n"
  "}\n"
  "\n"
  "PROCEDURE factors() {\n"
  "        LOCAL r, dr2\n"
  "        r = 1/2                : starts at edge (half diam)\n"
  "        dr2 = r/(Nannuli-1)/2  : full thickness of outermost annulus,\n"
  "        vrat = PI*(r-dr2/2)*2*dr2  : interior half\n"
  "        r = r - dr2\n"
  "}\n"
  "\n"
  "\n"
  "LOCAL dsq, dsqvol  : can't define local variable in KINETIC block\n"
  "                   :   or use in COMPARTMENT statement\n"
  "\n"
  "KINETIC state {\n"
  "  COMPARTMENT diam*diam*vrat {ca mg Buff1 Buff1_ca Buff2 Buff2_ca BTC BTC_ca DMNPE DMNPE_ca PV PV_ca PV_mg}\n"
  "  COMPARTMENT (1e10)*parea {pump pumpca}\n"
  "\n"
  "\n"
  "	:pump\n"
  "	~ ca + pump <-> pumpca  (kpmp1*parea*(1e10), kpmp2*parea*(1e10))\n"
  "	~ pumpca <-> pump   (kpmp3*parea*(1e10), 0)\n"
  "  	CONSERVE pump + pumpca = TotalPump * parea * (1e10)\n"
  "	\n"
  "	ica_pmp = 2*FARADAY*(f_flux - b_flux)/parea	\n"
  "	: all currents except pump\n"
  "	: ica is Ca efflux\n"
  "	~ ca << (-ica*PI*diam/(2*FARADAY))\n"
  "\n"
  "	:RADIAL DIFFUSION OF ca, mg and mobile buffers\n"
  "\n"
  "	dsq = diam*diam\n"
  "		dsqvol = dsq*vrat\n"
  "		~ ca + Buff1 <-> Buff1_ca (rf1*dsqvol, rf2*dsqvol)\n"
  "		~ ca + Buff2 <-> Buff2_ca (rf3*dsqvol, rf4*dsqvol)\n"
  "		~ ca + BTC <-> BTC_ca (b1*dsqvol, b2*dsqvol)\n"
  "		~ ca + DMNPE <-> DMNPE_ca (c1*dsqvol, c2*dsqvol)\n"
  "		:Calbindin	\n"
  "		:~ ca + CB <-> CB_ca_s (nf1*dsqvol, nf2*dsqvol)\n"
  "	       	:~ ca + CB <-> CB_f_ca (ns1*dsqvol, ns2*dsqvol)\n"
  "        	:~ ca + CB_f_ca <-> CB_ca_ca (nf1*dsqvol, nf2*dsqvol)\n"
  "        	:~ ca + CB_ca_s <-> CB_ca_ca (ns1*dsqvol, ns2*dsqvol)\n"
  "\n"
  "		:Paravalbumin\n"
  "        	~ ca + PV <-> PV_ca (m1*dsqvol, m2*dsqvol)\n"
  "        	~ mg + PV <-> PV_mg (p1*dsqvol, p2*dsqvol)\n"
  "\n"
  "		:Calmodulin\n"
  "		  :C-lobe\n"
  "		~ ca + CAM0 <-> CAM1C (K1Con*dsqvol, K1Coff*dsqvol)\n"
  "		~ ca + CAM1C <-> CAM2C (K2Con*dsqvol, K2Coff*dsqvol)\n"
  "		~ ca + CAM2C <-> CAM1N2C (K1Non*dsqvol, K1Noff*dsqvol)\n"
  "		~ ca + CAM1N2C <-> CAM4 (K2Non*dsqvol, K2Noff*dsqvol) \n"
  "\n"
  "		  :N-lobe\n"
  "		~ ca + CAM0 <-> CAM1N (K1Non*dsqvol, K1Noff*dsqvol)\n"
  "		~ ca + CAM1N <-> CAM2N (K2Non*dsqvol, K2Noff*dsqvol) \n"
  "		~ ca + CAM2N <-> CAM2N1C (K1Con*dsqvol, K1Coff*dsqvol)\n"
  "		~ ca + CAM2N1C <-> CAM4 (K2Con*dsqvol, K2Coff*dsqvol)\n"
  "\n"
  "		  :Mixed C and N lobes\n"
  "		~ ca + CAM1C <-> CAM1C1N (K1Non*dsqvol, K1Noff*dsqvol) \n"
  "		~ ca + CAM1N <-> CAM1C1N (K1Con*dsqvol, K1Coff*dsqvol)\n"
  "		~ ca + CAM1C1N <-> CAM1N2C (K2Con*dsqvol, K2Coff*dsqvol)\n"
  "		~ ca + CAM1C1N <-> CAM2N1C (K2Non*dsqvol, K2Noff*dsqvol) \n"
  "\n"
  "\n"
  "  	cai = ca\n"
  "	mgi = mg\n"
  "        icazz = nrvci\n"
  "}\n"
  "\n"
  "FUNCTION ssBuff1() (mM) {\n"
  "	ssBuff1 = Buffnull1/(1+((rf1/rf2)*cainull))\n"
  "}\n"
  "FUNCTION ssBuff1ca() (mM) {\n"
  "	ssBuff1ca = Buffnull1/(1+(rf2/(rf1*cainull)))\n"
  "}\n"
  "FUNCTION ssBuff2() (mM) {\n"
  "        ssBuff2 = Buffnull2/(1+((rf3/rf4)*cainull))\n"
  "}\n"
  "FUNCTION ssBuff2ca() (mM) {\n"
  "        ssBuff2ca = Buffnull2/(1+(rf4/(rf3*cainull)))\n"
  "}\n"
  "\n"
  "FUNCTION ssBTC() (mM) {\n"
  "	ssBTC = BTCnull/(1+((b1/b2)*cainull))\n"
  "}\n"
  "\n"
  "FUNCTION ssBTCca() (mM) {\n"
  "	ssBTCca = BTCnull/(1+(b2/(b1*cainull)))\n"
  "}\n"
  "\n"
  "FUNCTION ssDMNPE() (mM) {\n"
  "	ssDMNPE = DMNPEnull/(1+((c1/c2)*cainull))\n"
  "}\n"
  "\n"
  "FUNCTION ssDMNPEca() (mM) {\n"
  "	ssDMNPEca = DMNPEnull/(1+(c2/(c1*cainull)))\n"
  "}\n"
  "\n"
  ":FUNCTION ssCB( kdf(), kds()) (mM) {\n"
  ":	ssCB = CBnull/(1+kdf()+kds()+(kdf()*kds()))\n"
  ":}\n"
  ":FUNCTION ssCBfast( kdf(), kds()) (mM) {\n"
  ":	ssCBfast = (CBnull*kds())/(1+kdf()+kds()+(kdf()*kds()))\n"
  ":}\n"
  ":FUNCTION ssCBslow( kdf(), kds()) (mM) {\n"
  ":	ssCBslow = (CBnull*kdf())/(1+kdf()+kds()+(kdf()*kds()))\n"
  ":}\n"
  ":FUNCTION ssCBca(kdf(), kds()) (mM) {\n"
  ":	ssCBca = (CBnull*kdf()*kds())/(1+kdf()+kds()+(kdf()*kds()))\n"
  ":}\n"
  ":FUNCTION kdf() (1) {\n"
  ":	kdf = (cainull*nf1)/nf2\n"
  ":}\n"
  ":FUNCTION kds() (1) {\n"
  ":	kds = (cainull*ns1)/ns2\n"
  ":}\n"
  "FUNCTION kdc() (1) {\n"
  "	kdc = (cainull*m1)/m2\n"
  "}\n"
  "FUNCTION kdm() (1) {\n"
  "	kdm = (mginull*p1)/p2\n"
  "}\n"
  "FUNCTION ssPV( kdc(), kdm()) (mM) {\n"
  "	ssPV = PVnull/(1+kdc()+kdm())\n"
  "}\n"
  "FUNCTION ssPVca( kdc(), kdm()) (mM) {\n"
  "	ssPVca = (PVnull*kdc())/(1+kdc()+kdm())\n"
  "}\n"
  "FUNCTION ssPVmg( kdc(), kdm()) (mM) {\n"
  "	ssPVmg = (PVnull*kdm())/(1+kdc()+kdm())\n"
  "}\n"
  ;
#endif
