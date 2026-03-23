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
 
#define nrn_init _nrn_init__GRC_KM
#define _nrn_initial _nrn_initial__GRC_KM
#define nrn_cur _nrn_cur__GRC_KM
#define _nrn_current _nrn_current__GRC_KM
#define nrn_jacob _nrn_jacob__GRC_KM
#define nrn_state _nrn_state__GRC_KM
#define _net_receive _net_receive__GRC_KM 
#define _f_rate _f_rate__GRC_KM 
#define rate rate__GRC_KM 
#define states states__GRC_KM 
 
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
#define Aalpha_n _p[0]
#define Aalpha_n_columnindex 0
#define Kalpha_n _p[1]
#define Kalpha_n_columnindex 1
#define V0alpha_n _p[2]
#define V0alpha_n_columnindex 2
#define Abeta_n _p[3]
#define Abeta_n_columnindex 3
#define Kbeta_n _p[4]
#define Kbeta_n_columnindex 4
#define V0beta_n _p[5]
#define V0beta_n_columnindex 5
#define V0_ninf _p[6]
#define V0_ninf_columnindex 6
#define B_ninf _p[7]
#define B_ninf_columnindex 7
#define gkbar _p[8]
#define gkbar_columnindex 8
#define ik _p[9]
#define ik_columnindex 9
#define n_inf _p[10]
#define n_inf_columnindex 10
#define tau_n _p[11]
#define tau_n_columnindex 11
#define g _p[12]
#define g_columnindex 12
#define alpha_n _p[13]
#define alpha_n_columnindex 13
#define beta_n _p[14]
#define beta_n_columnindex 14
#define n _p[15]
#define n_columnindex 15
#define ek _p[16]
#define ek_columnindex 16
#define Dn _p[17]
#define Dn_columnindex 17
#define v _p[18]
#define v_columnindex 18
#define _g _p[19]
#define _g_columnindex 19
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
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
 static void _hoc_alp_n(void);
 static void _hoc_bet_n(void);
 static void _hoc_rate(void);
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
 "setdata_GRC_KM", _hoc_setdata,
 "alp_n_GRC_KM", _hoc_alp_n,
 "bet_n_GRC_KM", _hoc_bet_n,
 "rate_GRC_KM", _hoc_rate,
 0, 0
};
#define alp_n alp_n_GRC_KM
#define bet_n bet_n_GRC_KM
 extern double alp_n( _threadargsprotocomma_ double );
 extern double bet_n( _threadargsprotocomma_ double );
 
static void _check_rate(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_rate(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_GRC_KM
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_GRC_KM", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Aalpha_n_GRC_KM", "/ms",
 "Kalpha_n_GRC_KM", "mV",
 "V0alpha_n_GRC_KM", "mV",
 "Abeta_n_GRC_KM", "/ms",
 "Kbeta_n_GRC_KM", "mV",
 "V0beta_n_GRC_KM", "mV",
 "V0_ninf_GRC_KM", "mV",
 "B_ninf_GRC_KM", "mV",
 "gkbar_GRC_KM", "mho/cm2",
 "ik_GRC_KM", "mA/cm2",
 "tau_n_GRC_KM", "ms",
 "g_GRC_KM", "mho/cm2",
 "alpha_n_GRC_KM", "/ms",
 "beta_n_GRC_KM", "/ms",
 0,0
};
 static double delta_t = 0.01;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_GRC_KM", &usetable_GRC_KM,
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
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GRC_KM",
 "Aalpha_n_GRC_KM",
 "Kalpha_n_GRC_KM",
 "V0alpha_n_GRC_KM",
 "Abeta_n_GRC_KM",
 "Kbeta_n_GRC_KM",
 "V0beta_n_GRC_KM",
 "V0_ninf_GRC_KM",
 "B_ninf_GRC_KM",
 "gkbar_GRC_KM",
 0,
 "ik_GRC_KM",
 "n_inf_GRC_KM",
 "tau_n_GRC_KM",
 "g_GRC_KM",
 "alpha_n_GRC_KM",
 "beta_n_GRC_KM",
 0,
 "n_GRC_KM",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 20, _prop);
 	/*initialize range parameters*/
 	Aalpha_n = 0.0033;
 	Kalpha_n = 40;
 	V0alpha_n = -30;
 	Abeta_n = 0.0033;
 	Kbeta_n = -20;
 	V0beta_n = -30;
 	V0_ninf = -35;
 	B_ninf = 6;
 	gkbar = 0.00025;
 	_prop->param = _p;
 	_prop->param_size = 20;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
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

 void _GRC_KM_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 5);
  _extcall_thread = (Datum*)ecalloc(4, sizeof(Datum));
  _thread_mem_init(_extcall_thread);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 20, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GRC_KM /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/GRC_KM.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_n_inf;
 static double *_t_tau_n;
static int _reset;
static char *modelname = "Cerebellum Granule Cell Model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rate(_threadargsprotocomma_ double);
static int rate(_threadargsprotocomma_ double);
 
#define _deriv1_advance _thread[0]._i
#define _dith1 1
#define _recurse _thread[2]._i
#define _newtonspace1 _thread[3]._pvoid
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_rate(_threadargsprotocomma_ double _lv);
 static int _slist2[1];
  static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rate ( _threadargscomma_ v ) ;
   Dn = ( n_inf - n ) / tau_n ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rate ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_n )) ;
  return 0;
}
 /*END CVODE*/
 
static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0; int error = 0;
 { double* _savstate1 = _thread[_dith1]._pval;
 double* _dlist2 = _thread[_dith1]._pval + 1;
 int _counte = -1;
 if (!_recurse) {
 _recurse = 1;
 {int _id; for(_id=0; _id < 1; _id++) { _savstate1[_id] = _p[_slist1[_id]];}}
 error = nrn_newton_thread(_newtonspace1, 1,_slist2, _p, states, _dlist2, _ppvar, _thread, _nt);
 _recurse = 0; if(error) {abort_run(error);}}
 {
   rate ( _threadargscomma_ v ) ;
   Dn = ( n_inf - n ) / tau_n ;
   {int _id; for(_id=0; _id < 1; _id++) {
if (_deriv1_advance) {
 _dlist2[++_counte] = _p[_dlist1[_id]] - (_p[_slist1[_id]] - _savstate1[_id])/dt;
 }else{
_dlist2[++_counte] = _p[_slist1[_id]] - _savstate1[_id];}}}
 } }
 return _reset;}
 
double alp_n ( _threadargsprotocomma_ double _lv ) {
   double _lalp_n;
 double _lQ10 ;
 _lQ10 = pow( 3.0 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   _lalp_n = _lQ10 * Aalpha_n * exp ( ( _lv - V0alpha_n ) / Kalpha_n ) ;
   
return _lalp_n;
 }
 
static void _hoc_alp_n(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alp_n ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double bet_n ( _threadargsprotocomma_ double _lv ) {
   double _lbet_n;
 double _lQ10 ;
 _lQ10 = pow( 3.0 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   _lbet_n = _lQ10 * Abeta_n * exp ( ( _lv - V0beta_n ) / Kbeta_n ) ;
   
return _lbet_n;
 }
 
static void _hoc_bet_n(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  bet_n ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_rate, _tmin_rate;
  static void _check_rate(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_Aalpha_n;
  static double _sav_Kalpha_n;
  static double _sav_V0alpha_n;
  static double _sav_Abeta_n;
  static double _sav_Kbeta_n;
  static double _sav_V0beta_n;
  static double _sav_V0_ninf;
  static double _sav_B_ninf;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_Aalpha_n != Aalpha_n) { _maktable = 1;}
  if (_sav_Kalpha_n != Kalpha_n) { _maktable = 1;}
  if (_sav_V0alpha_n != V0alpha_n) { _maktable = 1;}
  if (_sav_Abeta_n != Abeta_n) { _maktable = 1;}
  if (_sav_Kbeta_n != Kbeta_n) { _maktable = 1;}
  if (_sav_V0beta_n != V0beta_n) { _maktable = 1;}
  if (_sav_V0_ninf != V0_ninf) { _maktable = 1;}
  if (_sav_B_ninf != B_ninf) { _maktable = 1;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rate =  - 100.0 ;
   _tmax =  30.0 ;
   _dx = (_tmax - _tmin_rate)/13000.; _mfac_rate = 1./_dx;
   for (_i=0, _x=_tmin_rate; _i < 13001; _x += _dx, _i++) {
    _f_rate(_p, _ppvar, _thread, _nt, _x);
    _t_n_inf[_i] = n_inf;
    _t_tau_n[_i] = tau_n;
   }
   _sav_Aalpha_n = Aalpha_n;
   _sav_Kalpha_n = Kalpha_n;
   _sav_V0alpha_n = V0alpha_n;
   _sav_Abeta_n = Abeta_n;
   _sav_Kbeta_n = Kbeta_n;
   _sav_V0beta_n = V0beta_n;
   _sav_V0_ninf = V0_ninf;
   _sav_B_ninf = B_ninf;
   _sav_celsius = celsius;
  }
 }

 static int rate(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_rate(_p, _ppvar, _thread, _nt);
#endif
 _n_rate(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_rate(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rate(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_rate * (_lv - _tmin_rate);
 if (isnan(_xi)) {
  n_inf = _xi;
  tau_n = _xi;
  return;
 }
 if (_xi <= 0.) {
 n_inf = _t_n_inf[0];
 tau_n = _t_tau_n[0];
 return; }
 if (_xi >= 13000.) {
 n_inf = _t_n_inf[13000];
 tau_n = _t_tau_n[13000];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 n_inf = _t_n_inf[_i] + _theta*(_t_n_inf[_i+1] - _t_n_inf[_i]);
 tau_n = _t_tau_n[_i] + _theta*(_t_tau_n[_i+1] - _t_tau_n[_i]);
 }

 
static int  _f_rate ( _threadargsprotocomma_ double _lv ) {
   double _la_n , _lb_n ;
 _la_n = alp_n ( _threadargscomma_ _lv ) ;
   _lb_n = bet_n ( _threadargscomma_ _lv ) ;
   tau_n = 1.0 / ( _la_n + _lb_n ) ;
   n_inf = 1.0 / ( 1.0 + exp ( - ( _lv - V0_ninf ) / B_ninf ) ) ;
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_rate(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
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
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
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
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
   _thread[_dith1]._pval = (double*)ecalloc(2, sizeof(double));
   _newtonspace1 = nrn_cons_newtonspace(1);
 }
 
static void _thread_cleanup(Datum* _thread) {
   free((void*)(_thread[_dith1]._pval));
   nrn_destroy_newtonspace(_newtonspace1);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  n = n0;
 {
   rate ( _threadargscomma_ v ) ;
   n = n_inf ;
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

#if 0
 _check_rate(_p, _ppvar, _thread, _nt);
#endif
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
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gkbar * n ;
   ik = g * ( v - ek ) ;
   alpha_n = alp_n ( _threadargscomma_ v ) ;
   beta_n = bet_n ( _threadargscomma_ v ) ;
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
 {  _deriv1_advance = 1;
 derivimplicit_thread(1, _slist1, _dlist1, _p, states, _ppvar, _thread, _nt);
_deriv1_advance = 0;
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 1; ++_i) {
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
 _slist1[0] = n_columnindex;  _dlist1[0] = Dn_columnindex;
 _slist2[0] = n_columnindex;
   _t_n_inf = makevector(13001*sizeof(double));
   _t_tau_n = makevector(13001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/GRC_KM.mod";
static const char* nmodl_file_text = 
  "TITLE Cerebellum Granule Cell Model\n"
  "\n"
  "COMMENT\n"
  "        KM channel\n"
  "   \n"
  "	Author: A. Fontana\n"
  "	CoAuthor: T.Nieus Last revised: 20.11.99\n"
  "	\n"
  "ENDCOMMENT\n"
  " \n"
  "NEURON { \n"
  "	SUFFIX GRC_KM \n"
  "	USEION k READ ek WRITE ik \n"
  "	RANGE gkbar, ik, g, alpha_n, beta_n \n"
  "	RANGE Aalpha_n, Kalpha_n, V0alpha_n\n"
  "	RANGE Abeta_n, Kbeta_n, V0beta_n\n"
  "	RANGE V0_ninf, B_ninf\n"
  "	RANGE n_inf, tau_n \n"
  "} \n"
  " \n"
  "UNITS { \n"
  "	(mA) = (milliamp) \n"
  "	(mV) = (millivolt) \n"
  "} \n"
  " \n"
  "PARAMETER { \n"
  "	Aalpha_n = 0.0033 (/ms)\n"
  "	Kalpha_n = 40 (mV)\n"
  "\n"
  "	V0alpha_n = -30 (mV)\n"
  "	Abeta_n = 0.0033 (/ms)\n"
  "	Kbeta_n = -20 (mV)\n"
  "\n"
  "	V0beta_n = -30 (mV)\n"
  "	V0_ninf = -35 (mV)	:-30\n"
  "	B_ninf = 6 (mV)		:6:4 rimesso a 6 dopo calibrazione febbraio 2003	\n"
  "	v (mV) \n"
  "	gkbar= 0.00025 (mho/cm2) :0.0001\n"
  "	ek = -84.69 (mV) \n"
  "	celsius = 30 (degC) \n"
  "} \n"
  "\n"
  "STATE { \n"
  "	n \n"
  "} \n"
  "\n"
  "ASSIGNED { \n"
  "	ik (mA/cm2) \n"
  "	n_inf \n"
  "	tau_n (ms) \n"
  "	g (mho/cm2) \n"
  "	alpha_n (/ms) \n"
  "	beta_n (/ms) \n"
  "} \n"
  " \n"
  "INITIAL { \n"
  "	rate(v) \n"
  "	n = n_inf \n"
  "} \n"
  " \n"
  "BREAKPOINT { \n"
  "	SOLVE states METHOD derivimplicit \n"
  "	g = gkbar*n \n"
  "	ik = g*(v - ek) \n"
  "	alpha_n = alp_n(v) \n"
  "	beta_n = bet_n(v) \n"
  "} \n"
  " \n"
  "DERIVATIVE states { \n"
  "	rate(v) \n"
  "	n' =(n_inf - n)/tau_n \n"
  "} \n"
  " \n"
  "FUNCTION alp_n(v(mV))(/ms) { LOCAL Q10\n"
  "	Q10 = 3^((celsius-22(degC))/10(degC)) \n"
  "	alp_n = Q10*Aalpha_n*exp((v-V0alpha_n)/Kalpha_n) \n"
  "} \n"
  " \n"
  "FUNCTION bet_n(v(mV))(/ms) { LOCAL Q10\n"
  "	Q10 = 3^((celsius-22(degC))/10(degC)) \n"
  "	bet_n = Q10*Abeta_n*exp((v-V0beta_n)/Kbeta_n) \n"
  "} \n"
  " \n"
  "PROCEDURE rate(v (mV)) {LOCAL a_n, b_n \n"
  "	TABLE n_inf, tau_n \n"
  "	DEPEND Aalpha_n, Kalpha_n, V0alpha_n, \n"
  "	       Abeta_n, Kbeta_n, V0beta_n, V0_ninf, B_ninf, celsius FROM -100 TO 30 WITH 13000 \n"
  "	a_n = alp_n(v)  \n"
  "	b_n = bet_n(v) \n"
  "	tau_n = 1/(a_n + b_n) \n"
  ":	n_inf = a_n/(a_n + b_n) \n"
  "	n_inf = 1/(1+exp(-(v-V0_ninf)/B_ninf))\n"
  "} \n"
  ;
#endif
