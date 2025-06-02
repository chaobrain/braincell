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
 
#define nrn_init _nrn_init__Kca3_1
#define _nrn_initial _nrn_initial__Kca3_1
#define nrn_cur _nrn_cur__Kca3_1
#define _nrn_current _nrn_current__Kca3_1
#define nrn_jacob _nrn_jacob__Kca3_1
#define nrn_state _nrn_state__Kca3_1
#define _net_receive _net_receive__Kca3_1 
#define _f_concdep _f_concdep__Kca3_1 
#define _f_vdep _f_vdep__Kca3_1 
#define concdep concdep__Kca3_1 
#define rate rate__Kca3_1 
#define state state__Kca3_1 
#define vdep vdep__Kca3_1 
 
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
#define gkbar _p[0]
#define gkbar_columnindex 0
#define Ybeta _p[1]
#define Ybeta_columnindex 1
#define ik _p[2]
#define ik_columnindex 2
#define Yalpha _p[3]
#define Yalpha_columnindex 3
#define Yvdep _p[4]
#define Yvdep_columnindex 4
#define Yconcdep _p[5]
#define Yconcdep_columnindex 5
#define tauY _p[6]
#define tauY_columnindex 6
#define Y_inf _p[7]
#define Y_inf_columnindex 7
#define Y _p[8]
#define Y_columnindex 8
#define cai _p[9]
#define cai_columnindex 9
#define DY _p[10]
#define DY_columnindex 10
#define ek _p[11]
#define ek_columnindex 11
#define qt _p[12]
#define qt_columnindex 12
#define v _p[13]
#define v_columnindex 13
#define _g _p[14]
#define _g_columnindex 14
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
 static void _hoc_concdep(void);
 static void _hoc_rate(void);
 static void _hoc_vdep(void);
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
 "setdata_Kca3_1", _hoc_setdata,
 "concdep_Kca3_1", _hoc_concdep,
 "rate_Kca3_1", _hoc_rate,
 "vdep_Kca3_1", _hoc_vdep,
 0, 0
};
 
static void _check_vdep(double*, Datum*, Datum*, NrnThread*); 
static void _check_concdep(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_vdep(_p, _ppvar, _thread, _nt);
   _check_concdep(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_Kca3_1
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gkbar_Kca3_1", 0, 1e+09,
 "usetable_Kca3_1", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gkbar_Kca3_1", "mho/cm2",
 "Ybeta_Kca3_1", "/ms",
 "ik_Kca3_1", "mA/cm2",
 "Yalpha_Kca3_1", "/ms",
 "Yconcdep_Kca3_1", "/ms",
 "tauY_Kca3_1", "ms",
 0,0
};
 static double Y0 = 0;
 static double delta_t = 1;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_Kca3_1", &usetable_Kca3_1,
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
"Kca3_1",
 "gkbar_Kca3_1",
 "Ybeta_Kca3_1",
 0,
 "ik_Kca3_1",
 "Yalpha_Kca3_1",
 "Yvdep_Kca3_1",
 "Yconcdep_Kca3_1",
 "tauY_Kca3_1",
 "Y_inf_Kca3_1",
 0,
 "Y_Kca3_1",
 0,
 0};
 static Symbol* _k_sym;
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 15, _prop);
 	/*initialize range parameters*/
 	gkbar = 0.12;
 	Ybeta = 0.05;
 	_prop->param = _p;
 	_prop->param_size = 15;
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
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Kca31_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	ion_reg("ca", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 15, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kca3_1 /home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca31.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double q10 = 3;
 static double *_t_Yvdep;
 static double *_t_Yconcdep;
static int _reset;
static char *modelname = "Calcium dependent potassium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_concdep(_threadargsprotocomma_ double);
static int _f_vdep(_threadargsprotocomma_ double);
static int concdep(_threadargsprotocomma_ double);
static int rate(_threadargsprotocomma_ double, double);
static int vdep(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_concdep(_threadargsprotocomma_ double _lv);
 static void _n_vdep(_threadargsprotocomma_ double _lv);
 static int _slist1[1], _dlist1[1];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rate ( _threadargscomma_ v , cai ) ;
   DY = Yalpha * ( 1.0 - Y ) - Ybeta * Y ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rate ( _threadargscomma_ v , cai ) ;
 DY = DY  / (1. - dt*( ( Yalpha )*( ( ( - 1.0 ) ) ) - ( Ybeta )*( 1.0 ) )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rate ( _threadargscomma_ v , cai ) ;
    Y = Y + (1. - exp(dt*(( Yalpha )*( ( ( - 1.0 ) ) ) - ( Ybeta )*( 1.0 ))))*(- ( ( Yalpha )*( ( 1.0 ) ) ) / ( ( Yalpha )*( ( ( - 1.0 ) ) ) - ( Ybeta )*( 1.0 ) ) - Y) ;
   }
  return 0;
}
 
static int  rate ( _threadargsprotocomma_ double _lv , double _lcai ) {
   vdep ( _threadargscomma_ _lv ) ;
   concdep ( _threadargscomma_ _lcai ) ;
   Yalpha = Yvdep * Yconcdep ;
   tauY = 1.0 / ( Yalpha + Ybeta ) ;
   Y_inf = Yalpha / ( Yalpha + Ybeta ) / qt ;
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 static double _mfac_vdep, _tmin_vdep;
  static void _check_vdep(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_vdep =  - 100.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_vdep)/100.; _mfac_vdep = 1./_dx;
   for (_i=0, _x=_tmin_vdep; _i < 101; _x += _dx, _i++) {
    _f_vdep(_p, _ppvar, _thread, _nt, _x);
    _t_Yvdep[_i] = Yvdep;
   }
  }
 }

 static int vdep(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_vdep(_p, _ppvar, _thread, _nt);
#endif
 _n_vdep(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_vdep(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_vdep(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_vdep * (_lv - _tmin_vdep);
 if (isnan(_xi)) {
  Yvdep = _xi;
  return;
 }
 if (_xi <= 0.) {
 Yvdep = _t_Yvdep[0];
 return; }
 if (_xi >= 100.) {
 Yvdep = _t_Yvdep[100];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 Yvdep = _t_Yvdep[_i] + _theta*(_t_Yvdep[_i+1] - _t_Yvdep[_i]);
 }

 
static int  _f_vdep ( _threadargsprotocomma_ double _lv ) {
   Yvdep = exp ( ( _lv * 1.0 + 70.0 ) / 27.0 ) ;
    return 0; }
 
static void _hoc_vdep(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_vdep(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 vdep ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_concdep, _tmin_concdep;
  static void _check_concdep(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_concdep =  0.0 ;
   _tmax =  0.01 ;
   _dx = (_tmax - _tmin_concdep)/1000.; _mfac_concdep = 1./_dx;
   for (_i=0, _x=_tmin_concdep; _i < 1001; _x += _dx, _i++) {
    _f_concdep(_p, _ppvar, _thread, _nt, _x);
    _t_Yconcdep[_i] = Yconcdep;
   }
  }
 }

 static int concdep(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lcai) { 
#if 0
_check_concdep(_p, _ppvar, _thread, _nt);
#endif
 _n_concdep(_p, _ppvar, _thread, _nt, _lcai);
 return 0;
 }

 static void _n_concdep(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lcai){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_concdep(_p, _ppvar, _thread, _nt, _lcai); return; 
}
 _xi = _mfac_concdep * (_lcai - _tmin_concdep);
 if (isnan(_xi)) {
  Yconcdep = _xi;
  return;
 }
 if (_xi <= 0.) {
 Yconcdep = _t_Yconcdep[0];
 return; }
 if (_xi >= 1000.) {
 Yconcdep = _t_Yconcdep[1000];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 Yconcdep = _t_Yconcdep[_i] + _theta*(_t_Yconcdep[_i+1] - _t_Yconcdep[_i]);
 }

 
static int  _f_concdep ( _threadargsprotocomma_ double _lcai ) {
   if ( _lcai < 0.01 ) {
     Yconcdep = 500.0 * ( 0.015 - _lcai * 1.0 ) / ( exp ( ( 0.015 - _lcai * 1.0 ) / 0.0013 ) - 1.0 ) ;
     }
   else {
     Yconcdep = 500.0 * 0.005 / ( exp ( 0.005 / 0.0013 ) - 1.0 ) ;
     }
    return 0; }
 
static void _hoc_concdep(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_concdep(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 concdep ( _p, _ppvar, _thread, _nt, *getarg(1) );
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
  cai = _ion_cai;
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
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  Y = Y0;
 {
   rate ( _threadargscomma_ v , cai ) ;
   Y = Yalpha / ( Yalpha + Ybeta ) ;
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

#if 0
 _check_vdep(_p, _ppvar, _thread, _nt);
 _check_concdep(_p, _ppvar, _thread, _nt);
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
  cai = _ion_cai;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ik = gkbar * Y * ( v - ek ) ;
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
 {   state(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = Y_columnindex;  _dlist1[0] = DY_columnindex;
   _t_Yvdep = makevector(101*sizeof(double));
   _t_Yconcdep = makevector(1001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca31.mod";
static const char* nmodl_file_text = 
  "TITLE Calcium dependent potassium channel\n"
  ": Implemented in Rubin and Cleland (2006) J Neurophysiology\n"
  ": Parameters from Bhalla and Bower (1993) J Neurophysiology\n"
  ": Adapted from /usr/local/neuron/demo/release/nachan.mod - squid\n"
  ":   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]\n"
  "\n"
  ":Suffix from Kca3 to Kca3_1\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "	SUFFIX Kca3_1\n"
  "	USEION k READ ek WRITE ik\n"
  "	USEION ca READ cai\n"
  "	RANGE gkbar, ik, Yconcdep, Yvdep\n"
  "	RANGE Yalpha, Ybeta, tauY, Y_inf\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)\n"
  "}\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "CONSTANT {\n"
  "	q10 = 3\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v (mV)\n"
  "	dt (ms)\n"
  "	gkbar= 0.120 (mho/cm2) <0,1e9>\n"
  "	Ybeta = 0.05 (/ms)\n"
  "	cai (mM) := 1e-5 (mM)\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "	Y\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ik (mA/cm2)\n"
  "	Yalpha   (/ms)\n"
  "	Yvdep    \n"
  "	Yconcdep (/ms)\n"
  "	tauY (ms)\n"
  "	Y_inf\n"
  "	ek (mV)\n"
  "\n"
  "	qt\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rate(v,cai)\n"
  "	Y = Yalpha/(Yalpha + Ybeta)\n"
  "	qt = q10^((celsius-37 (degC))/10 (degC))\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD cnexp\n"
  "	ik = gkbar*Y*(v - ek)\n"
  "}\n"
  "\n"
  "DERIVATIVE state {\n"
  "	rate(v,cai)\n"
  "	Y' = Yalpha*(1-Y) - Ybeta*Y\n"
  "}\n"
  "\n"
  "PROCEDURE rate(v(mV),cai(mM)) {\n"
  "	vdep(v)\n"
  "	concdep(cai)\n"
  "	Yalpha = Yvdep*Yconcdep\n"
  "	tauY = 1/(Yalpha + Ybeta)\n"
  "	Y_inf = Yalpha/(Yalpha + Ybeta) /qt\n"
  "}\n"
  "\n"
  "PROCEDURE vdep(v(mV)) {\n"
  "	TABLE Yvdep FROM -100 TO 100 WITH 100\n"
  "	Yvdep = exp((v*1(/mV)+70)/27)\n"
  "}\n"
  "\n"
  "PROCEDURE concdep(cai(mM)) {\n"
  "	TABLE Yconcdep FROM 0 TO 0.01 WITH 1000\n"
  "	if (cai < 0.01) {\n"
  "		Yconcdep = 500(/ms)*( 0.015-cai*1(/mM) )/( exp((0.015-cai*1(/mM))/0.0013) -1 )\n"
  "	} else {\n"
  "		Yconcdep = 500(/ms)*0.005/( exp(0.005/0.0013) -1 )\n"
  "	}\n"
  "}\n"
  ;
#endif
