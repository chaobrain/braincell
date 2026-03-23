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
 
#define nrn_init _nrn_init__Kca1_1
#define _nrn_initial _nrn_initial__Kca1_1
#define nrn_cur _nrn_cur__Kca1_1
#define _nrn_current _nrn_current__Kca1_1
#define nrn_jacob _nrn_jacob__Kca1_1
#define nrn_state _nrn_state__Kca1_1
#define _net_receive _net_receive__Kca1_1 
#define activation activation__Kca1_1 
#define rates rates__Kca1_1 
 
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
#define gbar _p[0]
#define gbar_columnindex 0
#define ik _p[1]
#define ik_columnindex 1
#define g _p[2]
#define g_columnindex 2
#define C0 _p[3]
#define C0_columnindex 3
#define C1 _p[4]
#define C1_columnindex 4
#define C2 _p[5]
#define C2_columnindex 5
#define C3 _p[6]
#define C3_columnindex 6
#define C4 _p[7]
#define C4_columnindex 7
#define O0 _p[8]
#define O0_columnindex 8
#define O1 _p[9]
#define O1_columnindex 9
#define O2 _p[10]
#define O2_columnindex 10
#define O3 _p[11]
#define O3_columnindex 11
#define O4 _p[12]
#define O4_columnindex 12
#define c01 _p[13]
#define c01_columnindex 13
#define c12 _p[14]
#define c12_columnindex 14
#define c23 _p[15]
#define c23_columnindex 15
#define c34 _p[16]
#define c34_columnindex 16
#define o01 _p[17]
#define o01_columnindex 17
#define o12 _p[18]
#define o12_columnindex 18
#define o23 _p[19]
#define o23_columnindex 19
#define o34 _p[20]
#define o34_columnindex 20
#define f0 _p[21]
#define f0_columnindex 21
#define f1 _p[22]
#define f1_columnindex 22
#define f2 _p[23]
#define f2_columnindex 23
#define f3 _p[24]
#define f3_columnindex 24
#define f4 _p[25]
#define f4_columnindex 25
#define c10 _p[26]
#define c10_columnindex 26
#define c21 _p[27]
#define c21_columnindex 27
#define c32 _p[28]
#define c32_columnindex 28
#define c43 _p[29]
#define c43_columnindex 29
#define o10 _p[30]
#define o10_columnindex 30
#define o21 _p[31]
#define o21_columnindex 31
#define o32 _p[32]
#define o32_columnindex 32
#define o43 _p[33]
#define o43_columnindex 33
#define b0 _p[34]
#define b0_columnindex 34
#define b1 _p[35]
#define b1_columnindex 35
#define b2 _p[36]
#define b2_columnindex 36
#define b3 _p[37]
#define b3_columnindex 37
#define b4 _p[38]
#define b4_columnindex 38
#define cai _p[39]
#define cai_columnindex 39
#define ek _p[40]
#define ek_columnindex 40
#define DC0 _p[41]
#define DC0_columnindex 41
#define DC1 _p[42]
#define DC1_columnindex 42
#define DC2 _p[43]
#define DC2_columnindex 43
#define DC3 _p[44]
#define DC3_columnindex 44
#define DC4 _p[45]
#define DC4_columnindex 45
#define DO0 _p[46]
#define DO0_columnindex 46
#define DO1 _p[47]
#define DO1_columnindex 47
#define DO2 _p[48]
#define DO2_columnindex 48
#define DO3 _p[49]
#define DO3_columnindex 49
#define DO4 _p[50]
#define DO4_columnindex 50
#define v _p[51]
#define v_columnindex 51
#define _g _p[52]
#define _g_columnindex 52
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
#define _ion_cai	*_ppvar[3]._pval
 
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
 static void _hoc_rates(void);
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
 "setdata_Kca1_1", _hoc_setdata,
 "rates_Kca1_1", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define Ko Ko_Kca1_1
 double Ko = 0.0011;
#define Kc Kc_Kca1_1
 double Kc = 0.011;
#define L0 L0_Kca1_1
 double L0 = 1806;
#define Qc Qc_Kca1_1
 double Qc = -0.67;
#define Qo Qo_Kca1_1
 double Qo = 0.73;
#define k1 k1_Kca1_1
 double k1 = 1000;
#define onoffrate onoffrate_Kca1_1
 double onoffrate = 1;
#define pb4 pb4_Kca1_1
 double pb4 = 0.092;
#define pb3 pb3_Kca1_1
 double pb3 = 0.486;
#define pb2 pb2_Kca1_1
 double pb2 = 0.659;
#define pb1 pb1_Kca1_1
 double pb1 = 1.152;
#define pb0 pb0_Kca1_1
 double pb0 = 3.936;
#define pf4 pf4_Kca1_1
 double pf4 = 0.557;
#define pf3 pf3_Kca1_1
 double pf3 = 0.295;
#define pf2 pf2_Kca1_1
 double pf2 = 0.04;
#define pf1 pf1_Kca1_1
 double pf1 = 0.007;
#define pf0 pf0_Kca1_1
 double pf0 = 0.00239;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "k1_Kca1_1", "/mM",
 "onoffrate_Kca1_1", "/ms",
 "Kc_Kca1_1", "mM",
 "Ko_Kca1_1", "mM",
 "pf0_Kca1_1", "/ms",
 "pf1_Kca1_1", "/ms",
 "pf2_Kca1_1", "/ms",
 "pf3_Kca1_1", "/ms",
 "pf4_Kca1_1", "/ms",
 "pb0_Kca1_1", "/ms",
 "pb1_Kca1_1", "/ms",
 "pb2_Kca1_1", "/ms",
 "pb3_Kca1_1", "/ms",
 "pb4_Kca1_1", "/ms",
 "gbar_Kca1_1", "S/cm2",
 "ik_Kca1_1", "milliamp/cm2",
 "g_Kca1_1", "S/cm2",
 0,0
};
 static double C40 = 0;
 static double C30 = 0;
 static double C20 = 0;
 static double C10 = 0;
 static double C00 = 0;
 static double O40 = 0;
 static double O30 = 0;
 static double O20 = 0;
 static double O10 = 0;
 static double O00 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Qo_Kca1_1", &Qo_Kca1_1,
 "Qc_Kca1_1", &Qc_Kca1_1,
 "k1_Kca1_1", &k1_Kca1_1,
 "onoffrate_Kca1_1", &onoffrate_Kca1_1,
 "L0_Kca1_1", &L0_Kca1_1,
 "Kc_Kca1_1", &Kc_Kca1_1,
 "Ko_Kca1_1", &Ko_Kca1_1,
 "pf0_Kca1_1", &pf0_Kca1_1,
 "pf1_Kca1_1", &pf1_Kca1_1,
 "pf2_Kca1_1", &pf2_Kca1_1,
 "pf3_Kca1_1", &pf3_Kca1_1,
 "pf4_Kca1_1", &pf4_Kca1_1,
 "pb0_Kca1_1", &pb0_Kca1_1,
 "pb1_Kca1_1", &pb1_Kca1_1,
 "pb2_Kca1_1", &pb2_Kca1_1,
 "pb3_Kca1_1", &pb3_Kca1_1,
 "pb4_Kca1_1", &pb4_Kca1_1,
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
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Kca1_1",
 "gbar_Kca1_1",
 0,
 "ik_Kca1_1",
 "g_Kca1_1",
 0,
 "C0_Kca1_1",
 "C1_Kca1_1",
 "C2_Kca1_1",
 "C3_Kca1_1",
 "C4_Kca1_1",
 "O0_Kca1_1",
 "O1_Kca1_1",
 "O2_Kca1_1",
 "O3_Kca1_1",
 "O4_Kca1_1",
 0,
 0};
 static Symbol* _k_sym;
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 53, _prop);
 	/*initialize range parameters*/
 	gbar = 0.01;
 	_prop->param = _p;
 	_prop->param_size = 53;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[3]._pval = &prop_ion->param[1]; /* cai */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Kca11_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	ion_reg("ca", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 3);
  _extcall_thread = (Datum*)ecalloc(2, sizeof(Datum));
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 53, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kca1_1 /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Kca11.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.81f0fae775425p+6, 96.4853}; /* 96.4853321233100161 */
 
#define R _nrnunit_R[_nrnunit_use_legacy_]
static double _nrnunit_R[2] = {0x1.0a1013e8990bep+3, 8.3145}; /* 8.3144626181532395 */
 static double q10 = 3;
static int _reset;
static char *modelname = "Large conductance Ca2+ activated K+ channel mslo";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double, double);
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[10], _dlist1[10]; static double *_temp1;
 static int activation();
 
static int activation (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<10;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ v , cai ) ;
   /* ~ C0 <-> C1 ( c01 , c10 )*/
 f_flux =  c01 * C0 ;
 b_flux =  c10 * C1 ;
 _RHS1( 5) -= (f_flux - b_flux);
 _RHS1( 4) += (f_flux - b_flux);
 
 _term =  c01 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 4 ,5)  -= _term;
 _term =  c10 ;
 _MATELM1( 5 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ C1 <-> C2 ( c12 , c21 )*/
 f_flux =  c12 * C1 ;
 b_flux =  c21 * C2 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  c12 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  c21 ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C3 ( c23 , c32 )*/
 f_flux =  c23 * C2 ;
 b_flux =  c32 * C3 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  c23 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  c32 ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C3 <-> C4 ( c34 , c43 )*/
 f_flux =  c34 * C3 ;
 b_flux =  c43 * C4 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  c34 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  c43 ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ O0 <-> O1 ( o01 , o10 )*/
 f_flux =  o01 * O0 ;
 b_flux =  o10 * O1 ;
 _RHS1( 9) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  o01 ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 8 ,9)  -= _term;
 _term =  o10 ;
 _MATELM1( 9 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ O1 <-> O2 ( o12 , o21 )*/
 f_flux =  o12 * O1 ;
 b_flux =  o21 * O2 ;
 _RHS1( 8) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  o12 ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 7 ,8)  -= _term;
 _term =  o21 ;
 _MATELM1( 8 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ O2 <-> O3 ( o23 , o32 )*/
 f_flux =  o23 * O2 ;
 b_flux =  o32 * O3 ;
 _RHS1( 7) -= (f_flux - b_flux);
 _RHS1( 6) += (f_flux - b_flux);
 
 _term =  o23 ;
 _MATELM1( 7 ,7)  += _term;
 _MATELM1( 6 ,7)  -= _term;
 _term =  o32 ;
 _MATELM1( 7 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ O3 <-> O4 ( o34 , o43 )*/
 f_flux =  o34 * O3 ;
 b_flux =  o43 * O4 ;
 _RHS1( 6) -= (f_flux - b_flux);
 
 _term =  o34 ;
 _MATELM1( 6 ,6)  += _term;
 _term =  o43 ;
 _MATELM1( 6 ,0)  -= _term;
 /*REACTION*/
  /* ~ C0 <-> O0 ( f0 , b0 )*/
 f_flux =  f0 * C0 ;
 b_flux =  b0 * O0 ;
 _RHS1( 5) -= (f_flux - b_flux);
 _RHS1( 9) += (f_flux - b_flux);
 
 _term =  f0 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 9 ,5)  -= _term;
 _term =  b0 ;
 _MATELM1( 5 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ C1 <-> O1 ( f1 , b1 )*/
 f_flux =  f1 * C1 ;
 b_flux =  b1 * O1 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  f1 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 8 ,4)  -= _term;
 _term =  b1 ;
 _MATELM1( 4 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ C2 <-> O2 ( f2 , b2 )*/
 f_flux =  f2 * C2 ;
 b_flux =  b2 * O2 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  f2 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 7 ,3)  -= _term;
 _term =  b2 ;
 _MATELM1( 3 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ C3 <-> O3 ( f3 , b3 )*/
 f_flux =  f3 * C3 ;
 b_flux =  b3 * O3 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 6) += (f_flux - b_flux);
 
 _term =  f3 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 6 ,2)  -= _term;
 _term =  b3 ;
 _MATELM1( 2 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ C4 <-> O4 ( f4 , b4 )*/
 f_flux =  f4 * C4 ;
 b_flux =  b4 * O4 ;
 _RHS1( 1) -= (f_flux - b_flux);
 
 _term =  f4 ;
 _MATELM1( 1 ,1)  += _term;
 _term =  b4 ;
 _MATELM1( 1 ,0)  -= _term;
 /*REACTION*/
   /* C0 + C1 + C2 + C3 + C4 + O0 + O1 + O2 + O3 + O4 = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= O4 ;
 _MATELM1(0, 6) = 1;
 _RHS1(0) -= O3 ;
 _MATELM1(0, 7) = 1;
 _RHS1(0) -= O2 ;
 _MATELM1(0, 8) = 1;
 _RHS1(0) -= O1 ;
 _MATELM1(0, 9) = 1;
 _RHS1(0) -= O0 ;
 _MATELM1(0, 1) = 1;
 _RHS1(0) -= C4 ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= C3 ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= C2 ;
 _MATELM1(0, 4) = 1;
 _RHS1(0) -= C1 ;
 _MATELM1(0, 5) = 1;
 _RHS1(0) -= C0 ;
 /*CONSERVATION*/
   } return _reset;
 }
 
static int  rates ( _threadargsprotocomma_ double _lv , double _lca ) {
   double _lqt , _lalpha , _lbeta ;
 _lqt = pow( q10 , ( ( celsius - 23.0 ) / 10.0 ) ) ;
   c01 = 4.0 * _lca * k1 * onoffrate * _lqt ;
   c12 = 3.0 * _lca * k1 * onoffrate * _lqt ;
   c23 = 2.0 * _lca * k1 * onoffrate * _lqt ;
   c34 = 1.0 * _lca * k1 * onoffrate * _lqt ;
   o01 = 4.0 * _lca * k1 * onoffrate * _lqt ;
   o12 = 3.0 * _lca * k1 * onoffrate * _lqt ;
   o23 = 2.0 * _lca * k1 * onoffrate * _lqt ;
   o34 = 1.0 * _lca * k1 * onoffrate * _lqt ;
   c10 = 1.0 * Kc * k1 * onoffrate * _lqt ;
   c21 = 2.0 * Kc * k1 * onoffrate * _lqt ;
   c32 = 3.0 * Kc * k1 * onoffrate * _lqt ;
   c43 = 4.0 * Kc * k1 * onoffrate * _lqt ;
   o10 = 1.0 * Ko * k1 * onoffrate * _lqt ;
   o21 = 2.0 * Ko * k1 * onoffrate * _lqt ;
   o32 = 3.0 * Ko * k1 * onoffrate * _lqt ;
   o43 = 4.0 * Ko * k1 * onoffrate * _lqt ;
   _lalpha = exp ( Qo * FARADAY * _lv / R / ( 273.15 + celsius ) ) ;
   _lbeta = exp ( Qc * FARADAY * _lv / R / ( 273.15 + celsius ) ) ;
   f0 = pf0 * _lalpha * _lqt ;
   f1 = pf1 * _lalpha * _lqt ;
   f2 = pf2 * _lalpha * _lqt ;
   f3 = pf3 * _lalpha * _lqt ;
   f4 = pf4 * _lalpha * _lqt ;
   b0 = pb0 * _lbeta * _lqt ;
   b1 = pb1 * _lbeta * _lqt ;
   b2 = pb2 * _lbeta * _lqt ;
   b3 = pb3 * _lbeta * _lqt ;
   b4 = pb4 * _lbeta * _lqt ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<10;_i++) _p[_dlist1[_i]] = 0.0;}
 rates ( _threadargscomma_ v , cai ) ;
 /* ~ C0 <-> C1 ( c01 , c10 )*/
 f_flux =  c01 * C0 ;
 b_flux =  c10 * C1 ;
 DC0 -= (f_flux - b_flux);
 DC1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C1 <-> C2 ( c12 , c21 )*/
 f_flux =  c12 * C1 ;
 b_flux =  c21 * C2 ;
 DC1 -= (f_flux - b_flux);
 DC2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> C3 ( c23 , c32 )*/
 f_flux =  c23 * C2 ;
 b_flux =  c32 * C3 ;
 DC2 -= (f_flux - b_flux);
 DC3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C3 <-> C4 ( c34 , c43 )*/
 f_flux =  c34 * C3 ;
 b_flux =  c43 * C4 ;
 DC3 -= (f_flux - b_flux);
 DC4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O0 <-> O1 ( o01 , o10 )*/
 f_flux =  o01 * O0 ;
 b_flux =  o10 * O1 ;
 DO0 -= (f_flux - b_flux);
 DO1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O1 <-> O2 ( o12 , o21 )*/
 f_flux =  o12 * O1 ;
 b_flux =  o21 * O2 ;
 DO1 -= (f_flux - b_flux);
 DO2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O2 <-> O3 ( o23 , o32 )*/
 f_flux =  o23 * O2 ;
 b_flux =  o32 * O3 ;
 DO2 -= (f_flux - b_flux);
 DO3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O3 <-> O4 ( o34 , o43 )*/
 f_flux =  o34 * O3 ;
 b_flux =  o43 * O4 ;
 DO3 -= (f_flux - b_flux);
 DO4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C0 <-> O0 ( f0 , b0 )*/
 f_flux =  f0 * C0 ;
 b_flux =  b0 * O0 ;
 DC0 -= (f_flux - b_flux);
 DO0 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C1 <-> O1 ( f1 , b1 )*/
 f_flux =  f1 * C1 ;
 b_flux =  b1 * O1 ;
 DC1 -= (f_flux - b_flux);
 DO1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> O2 ( f2 , b2 )*/
 f_flux =  f2 * C2 ;
 b_flux =  b2 * O2 ;
 DC2 -= (f_flux - b_flux);
 DO2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C3 <-> O3 ( f3 , b3 )*/
 f_flux =  f3 * C3 ;
 b_flux =  b3 * O3 ;
 DC3 -= (f_flux - b_flux);
 DO3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C4 <-> O4 ( f4 , b4 )*/
 f_flux =  f4 * C4 ;
 b_flux =  b4 * O4 ;
 DC4 -= (f_flux - b_flux);
 DO4 += (f_flux - b_flux);
 
 /*REACTION*/
   /* C0 + C1 + C2 + C3 + C4 + O0 + O1 + O2 + O3 + O4 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<10;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ v , cai ) ;
 /* ~ C0 <-> C1 ( c01 , c10 )*/
 _term =  c01 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 4 ,5)  -= _term;
 _term =  c10 ;
 _MATELM1( 5 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ C1 <-> C2 ( c12 , c21 )*/
 _term =  c12 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  c21 ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C3 ( c23 , c32 )*/
 _term =  c23 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  c32 ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C3 <-> C4 ( c34 , c43 )*/
 _term =  c34 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  c43 ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ O0 <-> O1 ( o01 , o10 )*/
 _term =  o01 ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 8 ,9)  -= _term;
 _term =  o10 ;
 _MATELM1( 9 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ O1 <-> O2 ( o12 , o21 )*/
 _term =  o12 ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 7 ,8)  -= _term;
 _term =  o21 ;
 _MATELM1( 8 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ O2 <-> O3 ( o23 , o32 )*/
 _term =  o23 ;
 _MATELM1( 7 ,7)  += _term;
 _MATELM1( 6 ,7)  -= _term;
 _term =  o32 ;
 _MATELM1( 7 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ O3 <-> O4 ( o34 , o43 )*/
 _term =  o34 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 0 ,6)  -= _term;
 _term =  o43 ;
 _MATELM1( 6 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ C0 <-> O0 ( f0 , b0 )*/
 _term =  f0 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 9 ,5)  -= _term;
 _term =  b0 ;
 _MATELM1( 5 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ C1 <-> O1 ( f1 , b1 )*/
 _term =  f1 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 8 ,4)  -= _term;
 _term =  b1 ;
 _MATELM1( 4 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ C2 <-> O2 ( f2 , b2 )*/
 _term =  f2 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 7 ,3)  -= _term;
 _term =  b2 ;
 _MATELM1( 3 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ C3 <-> O3 ( f3 , b3 )*/
 _term =  f3 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 6 ,2)  -= _term;
 _term =  b3 ;
 _MATELM1( 2 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ C4 <-> O4 ( f4 , b4 )*/
 _term =  f4 ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 0 ,1)  -= _term;
 _term =  b4 ;
 _MATELM1( 1 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
   /* C0 + C1 + C2 + C3 + C4 + O0 + O1 + O2 + O3 + O4 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 10;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
  cai = _ion_cai;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 10; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 10, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
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
  ek = _ion_ek;
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  C4 = C40;
  C3 = C30;
  C2 = C20;
  C1 = C10;
  C0 = C00;
  O4 = O40;
  O3 = O30;
  O2 = O20;
  O1 = O10;
  O0 = O00;
 {
    _ss_sparse_thread(&_thread[_spth1]._pvoid, 10, _slist1, _dlist1, _p, &t, dt, activation, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 10; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
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
  ek = _ion_ek;
  cai = _ion_cai;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gbar * ( O0 + O1 + O2 + O3 + O4 ) ;
   ik = g * ( v - ek ) ;
   }
 _current += ik;

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
  ek = _ion_ek;
  cai = _ion_cai;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
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
  ek = _ion_ek;
  cai = _ion_cai;
 {  sparse_thread(&_thread[_spth1]._pvoid, 10, _slist1, _dlist1, _p, &t, dt, activation, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 10; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } }}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = O4_columnindex;  _dlist1[0] = DO4_columnindex;
 _slist1[1] = C4_columnindex;  _dlist1[1] = DC4_columnindex;
 _slist1[2] = C3_columnindex;  _dlist1[2] = DC3_columnindex;
 _slist1[3] = C2_columnindex;  _dlist1[3] = DC2_columnindex;
 _slist1[4] = C1_columnindex;  _dlist1[4] = DC1_columnindex;
 _slist1[5] = C0_columnindex;  _dlist1[5] = DC0_columnindex;
 _slist1[6] = O3_columnindex;  _dlist1[6] = DO3_columnindex;
 _slist1[7] = O2_columnindex;  _dlist1[7] = DO2_columnindex;
 _slist1[8] = O1_columnindex;  _dlist1[8] = DO1_columnindex;
 _slist1[9] = O0_columnindex;  _dlist1[9] = DO0_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Kca11.mod";
static const char* nmodl_file_text = 
  "TITLE Large conductance Ca2+ activated K+ channel mslo\n"
  "\n"
  "COMMENT\n"
  "\n"
  "Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).\n"
  "\n"
  "Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*\n"
  "\n"
  "*Article available as Open Access\n"
  "\n"
  "PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513\n"
  "\n"
  "\n"
  "Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.\n"
  "Contact: Sungho Hong (shhong@oist.jp)\n"
  "\n"
  "Suffix from mslo to Kca1_1\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "  SUFFIX Kca1_1\n"
  "  USEION k READ ek WRITE ik\n"
  "  USEION ca READ cai\n"
  "  RANGE g, gbar, ik\n"
  "\n"
  "}\n"
  "\n"
  "UNITS { \n"
  "    (mV) = (millivolt)\n"
  "    (S) = (siemens)\n"
  "    (molar) = (1/liter)\n"
  "    (mM) = (millimolar)\n"
  "    FARADAY = (faraday) (kilocoulombs)\n"
  "    R = (k-mole) (joule/degC)\n"
  "}\n"
  "\n"
  "CONSTANT {\n"
  "    q10 = 3\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    gbar = 0.01 (S/cm2)\n"
  "    \n"
  "    Qo = 0.73\n"
  "    Qc = -0.67\n"
  "    \n"
  "    k1 = 1.0e3 (/mM)\n"
  "    onoffrate = 1 (/ms)\n"
  "    \n"
  "    L0 = 1806\n"
  "    Kc = 11.0e-3 (mM)\n"
  "    Ko = 1.1e-3 (mM)\n"
  "    \n"
  "    pf0 = 2.39e-3  (/ms)\n"
  "    pf1 = 7.0e-3  (/ms)\n"
  "    pf2 = 40e-3   (/ms)\n"
  "    pf3 = 295e-3  (/ms)\n"
  "    pf4 = 557e-3  (/ms)\n"
  "    \n"
  "    pb0 = 3936e-3 (/ms)\n"
  "    pb1 = 1152e-3 (/ms)\n"
  "    pb2 = 659e-3  (/ms)\n"
  "    pb3 = 486e-3  (/ms)\n"
  "    pb4 = 92e-3  (/ms)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    : rates\n"
  "    c01    (/ms)\n"
  "    c12    (/ms)\n"
  "    c23    (/ms)\n"
  "    c34    (/ms)\n"
  "    o01    (/ms)\n"
  "    o12    (/ms)\n"
  "    o23    (/ms)\n"
  "    o34    (/ms)\n"
  "    f0     (/ms)\n"
  "    f1     (/ms)\n"
  "    f2     (/ms)\n"
  "    f3     (/ms)\n"
  "    f4     (/ms)\n"
  "\n"
  "    c10    (/ms)\n"
  "    c21    (/ms)\n"
  "    c32    (/ms)\n"
  "    c43    (/ms)\n"
  "    o10    (/ms)\n"
  "    o21    (/ms)\n"
  "    o32    (/ms)\n"
  "    o43    (/ms)\n"
  "    b0     (/ms)\n"
  "    b1     (/ms)\n"
  "    b2     (/ms)\n"
  "    b3     (/ms)\n"
  "    b4     (/ms)\n"
  "    \n"
  "    v            (mV)\n"
  "    cai          (mM)\n"
  "    ek           (mV)\n"
  "    ik           (milliamp/cm2)\n"
  "    g            (S/cm2)\n"
  "    celsius      (degC)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    C0 FROM 0 TO 1\n"
  "    C1 FROM 0 TO 1\n"
  "    C2 FROM 0 TO 1\n"
  "    C3 FROM 0 TO 1\n"
  "    C4 FROM 0 TO 1\n"
  "    O0 FROM 0 TO 1\n"
  "    O1 FROM 0 TO 1\n"
  "    O2 FROM 0 TO 1\n"
  "    O3 FROM 0 TO 1\n"
  "    O4 FROM 0 TO 1\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE activation METHOD sparse\n"
  "    g = gbar * (O0 + O1 + O2 + O3 + O4)\n"
  "    ik = g * (v - ek)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  ":    rates(v, cai)\n"
  ":    SOLVE seqinitial\n"
  "    SOLVE activation STEADYSTATE sparse\n"
  "}\n"
  "\n"
  "KINETIC activation {\n"
  "    rates(v, cai)\n"
  "    ~ C0 <-> C1      (c01,c10)\n"
  "    ~ C1 <-> C2      (c12,c21)\n"
  "    ~ C2 <-> C3      (c23,c32)\n"
  "    ~ C3 <-> C4      (c34,c43)\n"
  "    ~ O0 <-> O1      (o01,o10)\n"
  "    ~ O1 <-> O2      (o12,o21)\n"
  "    ~ O2 <-> O3      (o23,o32)\n"
  "    ~ O3 <-> O4      (o34,o43)\n"
  "    ~ C0 <-> O0      (f0 , b0)\n"
  "    ~ C1 <-> O1      (f1 , b1)\n"
  "    ~ C2 <-> O2      (f2 , b2)\n"
  "    ~ C3 <-> O3      (f3 , b3)\n"
  "    ~ C4 <-> O4      (f4 , b4)\n"
  "\n"
  "CONSERVE C0 + C1 + C2 + C3 + C4 + O0 + O1 + O2 + O3 + O4 = 1\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV), ca (mM)) { \n"
  "    LOCAL qt, alpha, beta\n"
  "    \n"
  "    qt = q10^((celsius-23 (degC))/10 (degC))\n"
  "    \n"
  "    c01 = 4*ca*k1*onoffrate*qt\n"
  "    c12 = 3*ca*k1*onoffrate*qt\n"
  "    c23 = 2*ca*k1*onoffrate*qt\n"
  "    c34 = 1*ca*k1*onoffrate*qt\n"
  "    o01 = 4*ca*k1*onoffrate*qt\n"
  "    o12 = 3*ca*k1*onoffrate*qt\n"
  "    o23 = 2*ca*k1*onoffrate*qt\n"
  "    o34 = 1*ca*k1*onoffrate*qt\n"
  "    \n"
  "    c10 = 1*Kc*k1*onoffrate*qt\n"
  "    c21 = 2*Kc*k1*onoffrate*qt\n"
  "    c32 = 3*Kc*k1*onoffrate*qt\n"
  "    c43 = 4*Kc*k1*onoffrate*qt\n"
  "    o10 = 1*Ko*k1*onoffrate*qt\n"
  "    o21 = 2*Ko*k1*onoffrate*qt\n"
  "    o32 = 3*Ko*k1*onoffrate*qt\n"
  "    o43 = 4*Ko*k1*onoffrate*qt\n"
  "    \n"
  "    alpha = exp(Qo*FARADAY*v/R/(273.15 + celsius))\n"
  "    beta  = exp(Qc*FARADAY*v/R/(273.15 + celsius))\n"
  "    \n"
  "    f0  = pf0*alpha*qt\n"
  "    f1  = pf1*alpha*qt\n"
  "    f2  = pf2*alpha*qt\n"
  "    f3  = pf3*alpha*qt\n"
  "    f4  = pf4*alpha*qt\n"
  "    \n"
  "    b0  = pb0*beta*qt\n"
  "    b1  = pb1*beta*qt\n"
  "    b2  = pb2*beta*qt\n"
  "    b3  = pb3*beta*qt\n"
  "    b4  = pb4*beta*qt\n"
  "}\n"
  ;
#endif
