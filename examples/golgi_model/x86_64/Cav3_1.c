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
 
#define nrn_init _nrn_init__Cav3_1
#define _nrn_initial _nrn_initial__Cav3_1
#define nrn_cur _nrn_cur__Cav3_1
#define _nrn_current _nrn_current__Cav3_1
#define nrn_jacob _nrn_jacob__Cav3_1
#define nrn_state _nrn_state__Cav3_1
#define _net_receive _net_receive__Cav3_1 
#define castate castate__Cav3_1 
#define evaluate_fct evaluate_fct__Cav3_1 
 
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
#define pcabar _p[0]
#define pcabar_columnindex 0
#define ica _p[1]
#define ica_columnindex 1
#define g _p[2]
#define g_columnindex 2
#define minf _p[3]
#define minf_columnindex 3
#define taum _p[4]
#define taum_columnindex 4
#define hinf _p[5]
#define hinf_columnindex 5
#define tauh _p[6]
#define tauh_columnindex 6
#define m _p[7]
#define m_columnindex 7
#define h _p[8]
#define h_columnindex 8
#define cai _p[9]
#define cai_columnindex 9
#define cao _p[10]
#define cao_columnindex 10
#define Dm _p[11]
#define Dm_columnindex 11
#define Dh _p[12]
#define Dh_columnindex 12
#define T _p[13]
#define T_columnindex 13
#define E _p[14]
#define E_columnindex 14
#define zeta _p[15]
#define zeta_columnindex 15
#define qt _p[16]
#define qt_columnindex 16
#define v _p[17]
#define v_columnindex 17
#define _g _p[18]
#define _g_columnindex 18
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
 
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
 static void _hoc_evaluate_fct(void);
 static void _hoc_ghk(void);
 static void _hoc_kelvinfkt(void);
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
 "setdata_Cav3_1", _hoc_setdata,
 "evaluate_fct_Cav3_1", _hoc_evaluate_fct,
 "ghk_Cav3_1", _hoc_ghk,
 "kelvinfkt_Cav3_1", _hoc_kelvinfkt,
 0, 0
};
#define ghk ghk_Cav3_1
#define kelvinfkt kelvinfkt_Cav3_1
 extern double ghk( _threadargsprotocomma_ double , double , double , double );
 extern double kelvinfkt( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define A_tau_h A_tau_h_Cav3_1
 double A_tau_h = 1;
#define A_tau_m A_tau_m_Cav3_1
 double A_tau_m = 1;
#define C_tau_h C_tau_h_Cav3_1
 double C_tau_h = 15;
#define C_tau_m C_tau_m_Cav3_1
 double C_tau_m = 1;
#define eca eca_Cav3_1
 double eca = 0;
#define k_tau_h1 k_tau_h1_Cav3_1
 double k_tau_h1 = 7;
#define k_tau_m2 k_tau_m2_Cav3_1
 double k_tau_m2 = -18;
#define k_tau_m1 k_tau_m1_Cav3_1
 double k_tau_m1 = 9;
#define k_h_inf k_h_inf_Cav3_1
 double k_h_inf = 7;
#define k_m_inf k_m_inf_Cav3_1
 double k_m_inf = -5;
#define v0_tau_h1 v0_tau_h1_Cav3_1
 double v0_tau_h1 = -32;
#define v0_tau_m2 v0_tau_m2_Cav3_1
 double v0_tau_m2 = -102;
#define v0_tau_m1 v0_tau_m1_Cav3_1
 double v0_tau_m1 = -40;
#define v0_h_inf v0_h_inf_Cav3_1
 double v0_h_inf = -72;
#define v0_m_inf v0_m_inf_Cav3_1
 double v0_m_inf = -52;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "eca_Cav3_1", "mV",
 "v0_m_inf_Cav3_1", "mV",
 "v0_h_inf_Cav3_1", "mV",
 "k_m_inf_Cav3_1", "mV",
 "k_h_inf_Cav3_1", "mV",
 "v0_tau_m1_Cav3_1", "mV",
 "v0_tau_m2_Cav3_1", "mV",
 "k_tau_m1_Cav3_1", "mV",
 "k_tau_m2_Cav3_1", "mV",
 "v0_tau_h1_Cav3_1", "mV",
 "k_tau_h1_Cav3_1", "mV",
 "pcabar_Cav3_1", "cm/s",
 "ica_Cav3_1", "mA/cm2",
 "g_Cav3_1", "coulombs/cm3",
 "taum_Cav3_1", "ms",
 "tauh_Cav3_1", "ms",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "eca_Cav3_1", &eca_Cav3_1,
 "v0_m_inf_Cav3_1", &v0_m_inf_Cav3_1,
 "v0_h_inf_Cav3_1", &v0_h_inf_Cav3_1,
 "k_m_inf_Cav3_1", &k_m_inf_Cav3_1,
 "k_h_inf_Cav3_1", &k_h_inf_Cav3_1,
 "C_tau_m_Cav3_1", &C_tau_m_Cav3_1,
 "A_tau_m_Cav3_1", &A_tau_m_Cav3_1,
 "v0_tau_m1_Cav3_1", &v0_tau_m1_Cav3_1,
 "v0_tau_m2_Cav3_1", &v0_tau_m2_Cav3_1,
 "k_tau_m1_Cav3_1", &k_tau_m1_Cav3_1,
 "k_tau_m2_Cav3_1", &k_tau_m2_Cav3_1,
 "C_tau_h_Cav3_1", &C_tau_h_Cav3_1,
 "A_tau_h_Cav3_1", &A_tau_h_Cav3_1,
 "v0_tau_h1_Cav3_1", &v0_tau_h1_Cav3_1,
 "k_tau_h1_Cav3_1", &k_tau_h1_Cav3_1,
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
"Cav3_1",
 "pcabar_Cav3_1",
 0,
 "ica_Cav3_1",
 "g_Cav3_1",
 "minf_Cav3_1",
 "taum_Cav3_1",
 "hinf_Cav3_1",
 "tauh_Cav3_1",
 0,
 "m_Cav3_1",
 "h_Cav3_1",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	pcabar = 0.00025;
 	_prop->param = _p;
 	_prop->param_size = 19;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Cav3_1_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", 2.0);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 19, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Cav3_1 /home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Cav3_1.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double F = 9.6485e4;
 static double R = 8.3145;
 static double q10 = 3;
static int _reset;
static char *modelname = "Low threshold calcium current Cerebellum Purkinje Cell Model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int evaluate_fct(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int castate(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   evaluate_fct ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / taum ;
   Dh = ( hinf - h ) / tauh ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 evaluate_fct ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taum )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauh )) ;
  return 0;
}
 /*END CVODE*/
 static int castate (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   evaluate_fct ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taum)))*(- ( ( ( minf ) ) / taum ) / ( ( ( ( - 1.0 ) ) ) / taum ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauh)))*(- ( ( ( hinf ) ) / tauh ) / ( ( ( ( - 1.0 ) ) ) / tauh ) - h) ;
   }
  return 0;
}
 
double ghk ( _threadargsprotocomma_ double _lv , double _lci , double _lco , double _lz ) {
   double _lghk;
 E = ( 1e-3 ) * _lv ;
   zeta = ( _lz * F * E ) / ( R * T ) ;
   if ( fabs ( 1.0 - exp ( - zeta ) ) < 1e-6 ) {
     _lghk = ( 1e-6 ) * ( _lz * F ) * ( _lci - _lco * exp ( - zeta ) ) * ( 1.0 + zeta / 2.0 ) ;
     }
   else {
     _lghk = ( 1e-6 ) * ( _lz * zeta * F ) * ( _lci - _lco * exp ( - zeta ) ) / ( 1.0 - exp ( - zeta ) ) ;
     }
   
return _lghk;
 }
 
static void _hoc_ghk(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ghk ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
static int  evaluate_fct ( _threadargsprotocomma_ double _lv ) {
   minf = 1.0 / ( 1.0 + exp ( ( _lv - v0_m_inf ) / k_m_inf ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( _lv - v0_h_inf ) / k_h_inf ) ) ;
   if ( _lv <= - 90.0 ) {
     taum = 1.0 ;
     }
   else {
     taum = ( C_tau_m + A_tau_m / ( exp ( ( _lv - v0_tau_m1 ) / k_tau_m1 ) + exp ( ( _lv - v0_tau_m2 ) / k_tau_m2 ) ) ) / qt ;
     }
   tauh = ( C_tau_h + A_tau_h / exp ( ( _lv - v0_tau_h1 ) / k_tau_h1 ) ) / qt ;
   g = ghk ( _threadargscomma_ _lv , cai , cao , 2.0 ) ;
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 evaluate_fct ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double kelvinfkt ( _threadargsprotocomma_ double _lt ) {
   double _lkelvinfkt;
 _lkelvinfkt = 273.19 + _lt ;
   
return _lkelvinfkt;
 }
 
static void _hoc_kelvinfkt(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  kelvinfkt ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
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
  cai = _ion_cai;
  cao = _ion_cao;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   T = kelvinfkt ( _threadargscomma_ celsius ) ;
   evaluate_fct ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   qt = pow( q10 , ( ( celsius - 37.0 ) / 10.0 ) ) ;
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
  cai = _ion_cai;
  cao = _ion_cao;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ica = ( 1e3 ) * pcabar * m * m * h * g ;
   }
 _current += ica;

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
  cai = _ion_cai;
  cao = _ion_cao;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
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
  cai = _ion_cai;
  cao = _ion_cao;
 {   castate(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Cav3_1.mod";
static const char* nmodl_file_text = 
  "TITLE Low threshold calcium current Cerebellum Purkinje Cell Model\n"
  "\n"
  "COMMENT\n"
  "\n"
  "Kinetics adapted to fit the Cav3.1 Iftinca et al 2006, Temperature dependence of T-type Calcium channel gating, NEUROSCIENCE\n"
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
  "Suffix from CaT3_1 to CaV3_1\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "        SUFFIX Cav3_1\n"
  "        USEION ca READ cai, cao WRITE ica VALENCE 2\n"
  "        RANGE g, pcabar, minf, taum, hinf, tauh\n"
  "	RANGE ica, m ,h\n"
  "\n"
  "    }\n"
  "\n"
  "UNITS {\n"
  "        (molar) = (1/liter)\n"
  "        (mV) =  (millivolt)\n"
  "        (mA) =  (milliamp)\n"
  "        (mM) =  (millimolar)\n"
  "\n"
  "}\n"
  "\n"
  "CONSTANT {\n"
  "	F = 9.6485e4 (coulombs)\n"
  "	R = 8.3145 (joule/kelvin)\n"
  "	q10 = 3\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "        v               (mV)\n"
  "        celsius (degC)\n"
  "        eca (mV)\n"
  "	pcabar  = 2.5e-4 (cm/s)\n"
  "        cai  (mM)           : adjusted for eca=120 mV\n"
  "	cao  (mM)\n"
  "	\n"
  "	v0_m_inf = -52 (mV)\n"
  "	v0_h_inf = -72 (mV)\n"
  "	k_m_inf = -5 (mV)\n"
  "	k_h_inf = 7  (mV)\n"
  "	\n"
  "	C_tau_m = 1\n"
  "	A_tau_m = 1.0\n"
  "	v0_tau_m1 = -40 (mV)\n"
  "	v0_tau_m2 = -102 (mV)\n"
  "	k_tau_m1 = 9 (mV)\n"
  "	k_tau_m2 = -18 (mV)\n"
  "	\n"
  "	C_tau_h = 15\n"
  "	A_tau_h = 1.0\n"
  "	v0_tau_h1 = -32 (mV)\n"
  "	k_tau_h1 = 7 (mV)\n"
  "	\n"
  "    }\n"
  "    \n"
  "\n"
  "STATE {\n"
  "        m h\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "        ica     (mA/cm2)\n"
  "	g        (coulombs/cm3) \n"
  "        minf\n"
  "        taum   (ms)\n"
  "        hinf\n"
  "        tauh   (ms)\n"
  "	T (kelvin)\n"
  "	E (volt)\n"
  "	zeta\n"
  "	qt\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE castate METHOD cnexp \n"
  "\n"
  "        ica = (1e3) *pcabar*m*m *h * g\n"
  "}\n"
  "\n"
  "DERIVATIVE castate {\n"
  "        evaluate_fct(v)\n"
  "\n"
  "        m' = (minf - m) / taum\n"
  "        h' = (hinf - h) / tauh\n"
  "}\n"
  "\n"
  "FUNCTION ghk( v (mV), ci (mM), co (mM), z )  (coulombs/cm3) {\n"
  "    E = (1e-3) * v\n"
  "      zeta = (z*F*E)/(R*T)\n"
  "\n"
  "\n"
  "    if ( fabs(1-exp(-zeta)) < 1e-6 ) {\n"
  "        ghk = (1e-6) * (z*F) * (ci - co*exp(-zeta)) * (1 + zeta/2)\n"
  "    } else {\n"
  "        ghk = (1e-6) * (z*zeta*F) * (ci - co*exp(-zeta)) / (1-exp(-zeta))\n"
  "    }\n"
  "}\n"
  "\n"
  "\n"
  "UNITSOFF\n"
  "INITIAL {\n"
  "	\n"
  "	T = kelvinfkt (celsius)\n"
  "\n"
  "        evaluate_fct(v)\n"
  "        m = minf\n"
  "        h = hinf\n"
  "	qt = q10^((celsius-37 (degC))/10 (degC))\n"
  "}\n"
  "\n"
  "PROCEDURE evaluate_fct(v(mV)) { \n"
  "\n"
  "        minf = 1.0 / ( 1 + exp((v  - v0_m_inf)/k_m_inf) )\n"
  "        hinf = 1.0 / ( 1 + exp((v - v0_h_inf)/k_h_inf) )\n"
  "        if (v<=-90) {\n"
  "	taum = 1\n"
  "	} else {\n"
  "	taum = ( C_tau_m + A_tau_m / (exp((v - v0_tau_m1)/ k_tau_m1) + exp((v - v0_tau_m2)/k_tau_m2))) / qt\n"
  "	}\n"
  "	tauh = ( C_tau_h + A_tau_h / exp((v - v0_tau_h1)/k_tau_h1) ) / qt\n"
  "	g = ghk(v, cai, cao, 2)\n"
  "}\n"
  "\n"
  "FUNCTION kelvinfkt( t (degC) )  (kelvin) {\n"
  "    kelvinfkt = 273.19 + t\n"
  "}\n"
  "\n"
  "UNITSON\n"
  ;
#endif
