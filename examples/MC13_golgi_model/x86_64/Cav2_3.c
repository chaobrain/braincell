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
 
#define nrn_init _nrn_init__Cav2_3
#define _nrn_initial _nrn_initial__Cav2_3
#define nrn_cur _nrn_cur__Cav2_3
#define _nrn_current _nrn_current__Cav2_3
#define nrn_jacob _nrn_jacob__Cav2_3
#define nrn_state _nrn_state__Cav2_3
#define _net_receive _net_receive__Cav2_3 
#define _f_mhn _f_mhn__Cav2_3 
#define mhn mhn__Cav2_3 
#define states states__Cav2_3 
 
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
#define gcabar _p[0]
#define gcabar_columnindex 0
#define ica _p[1]
#define ica_columnindex 1
#define inf (_p + 2)
#define inf_columnindex 2
#define tau (_p + 4)
#define tau_columnindex 4
#define g _p[6]
#define g_columnindex 6
#define gmax _p[7]
#define gmax_columnindex 7
#define m _p[8]
#define m_columnindex 8
#define h _p[9]
#define h_columnindex 9
#define eca _p[10]
#define eca_columnindex 10
#define Dm _p[11]
#define Dm_columnindex 11
#define Dh _p[12]
#define Dh_columnindex 12
#define v _p[13]
#define v_columnindex 13
#define _g _p[14]
#define _g_columnindex 14
#define _ion_eca	*_ppvar[0]._pval
#define _ion_ica	*_ppvar[1]._pval
#define _ion_dicadv	*_ppvar[2]._pval
 
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
 static void _hoc_mhn(void);
 static void _hoc_vartau(void);
 static void _hoc_varss(void);
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
 "setdata_Cav2_3", _hoc_setdata,
 "mhn_Cav2_3", _hoc_mhn,
 "vartau_Cav2_3", _hoc_vartau,
 "varss_Cav2_3", _hoc_varss,
 0, 0
};
#define vartau vartau_Cav2_3
#define varss varss_Cav2_3
 extern double vartau( _threadargsprotocomma_ double , double );
 extern double varss( _threadargsprotocomma_ double , double );
 
static void _check_mhn(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_mhn(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_Cav2_3
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_Cav2_3", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gcabar_Cav2_3", "mho/cm2",
 "ica_Cav2_3", "mA/cm2",
 "tau_Cav2_3", "ms",
 "g_Cav2_3", "mho/cm2",
 "gmax_Cav2_3", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_Cav2_3", &usetable_Cav2_3,
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
"Cav2_3",
 "gcabar_Cav2_3",
 0,
 "ica_Cav2_3",
 "inf_Cav2_3[2]",
 "tau_Cav2_3[2]",
 "g_Cav2_3",
 "gmax_Cav2_3",
 0,
 "m_Cav2_3",
 "h_Cav2_3",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 15, _prop);
 	/*initialize range parameters*/
 	gcabar = 0;
 	_prop->param = _p;
 	_prop->param_size = 15;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* eca */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
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

 void _Cav2_3_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
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
  hoc_register_prop_size(_mechtype, 15, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Cav2_3 /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Cav2_3.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_inf[2];
 static double *_t_tau[2];
static int _reset;
static char *modelname = "Ca R-type channel with medium threshold for activation";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_mhn(_threadargsprotocomma_ double);
static int mhn(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_mhn(_threadargsprotocomma_ double _lv);
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   mhn ( _threadargscomma_ v ) ;
   Dm = ( inf [ 0 ] - m ) / tau [ 0 ] ;
   Dh = ( inf [ 1 ] - h ) / tau [ 1 ] ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 mhn ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau[0] )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau[1] )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   mhn ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau[0])))*(- ( ( ( inf[0] ) ) / tau[0] ) / ( ( ( ( - 1.0 ) ) ) / tau[0] ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau[1])))*(- ( ( ( inf[1] ) ) / tau[1] ) / ( ( ( ( - 1.0 ) ) ) / tau[1] ) - h) ;
   }
  return 0;
}
 
double varss ( _threadargsprotocomma_ double _lv , double _li ) {
   double _lvarss;
 if ( _li  == 0.0 ) {
     _lvarss = 1.0 / ( 1.0 + exp ( ( _lv + 48.5 ) / ( - 3.0 ) ) ) ;
     }
   else if ( _li  == 1.0 ) {
     _lvarss = 1.0 / ( 1.0 + exp ( ( _lv + 53.0 ) / ( 1.0 ) ) ) ;
     }
   
return _lvarss;
 }
 
static void _hoc_varss(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  varss ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double vartau ( _threadargsprotocomma_ double _lv , double _li ) {
   double _lvartau;
 if ( _li  == 0.0 ) {
     _lvartau = 50.0 ;
     }
   else if ( _li  == 1.0 ) {
     _lvartau = 5.0 ;
     }
   
return _lvartau;
 }
 
static void _hoc_vartau(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  vartau ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 static double _mfac_mhn, _tmin_mhn;
  static void _check_mhn(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_mhn =  - 100.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_mhn)/200.; _mfac_mhn = 1./_dx;
   for (_i=0, _x=_tmin_mhn; _i < 201; _x += _dx, _i++) {
    _f_mhn(_p, _ppvar, _thread, _nt, _x);
    for (_j = 0; _j < 2; _j++) { _t_inf[_j][_i] = inf[_j];
}    for (_j = 0; _j < 2; _j++) { _t_tau[_j][_i] = tau[_j];
}   }
   _sav_celsius = celsius;
  }
 }

 static int mhn(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_mhn(_p, _ppvar, _thread, _nt);
#endif
 _n_mhn(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_mhn(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_mhn(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_mhn * (_lv - _tmin_mhn);
 if (isnan(_xi)) {
  for (_j = 0; _j < 2; _j++) { inf[_j] = _xi;
}  for (_j = 0; _j < 2; _j++) { tau[_j] = _xi;
}  return;
 }
 if (_xi <= 0.) {
 for (_j = 0; _j < 2; _j++) { inf[_j] = _t_inf[_j][0];
} for (_j = 0; _j < 2; _j++) { tau[_j] = _t_tau[_j][0];
} return; }
 if (_xi >= 200.) {
 for (_j = 0; _j < 2; _j++) { inf[_j] = _t_inf[_j][200];
} for (_j = 0; _j < 2; _j++) { tau[_j] = _t_tau[_j][200];
} return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 for (_j = 0; _j < 2; _j++) {double *_t = _t_inf[_j]; inf[_j] = _t[_i] + _theta*(_t[_i+1] - _t[_i]);}
 for (_j = 0; _j < 2; _j++) {double *_t = _t_tau[_j]; tau[_j] = _t[_i] + _theta*(_t[_i+1] - _t[_i]);}
 }

 
static int  _f_mhn ( _threadargsprotocomma_ double _lv ) {
   double _la , _lb ;
 {int  _li ;for ( _li = 0 ; _li <= 1 ; _li ++ ) {
     tau [ _li ] = vartau ( _threadargscomma_ _lv , ((double) _li ) ) ;
     inf [ _li ] = varss ( _threadargscomma_ _lv , ((double) _li ) ) ;
     } }
    return 0; }
 
static void _hoc_mhn(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_mhn(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 mhn ( _p, _ppvar, _thread, _nt, *getarg(1) );
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
  eca = _ion_eca;
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
  eca = _ion_eca;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   mhn ( _threadargscomma_ v ) ;
   m = inf [ 0 ] ;
   h = inf [ 1 ] ;
   g = gcabar * m * m * m * h ;
   ica = g * ( v - eca ) ;
   gmax = g ;
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
 _check_mhn(_p, _ppvar, _thread, _nt);
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
  eca = _ion_eca;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gcabar * m * m * m * h ;
   ica = g * ( v - eca ) ;
   if ( g > gmax ) {
     gmax = g ;
     }
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
  eca = _ion_eca;
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
  eca = _ion_eca;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
  for (_i=0; _i < 2; _i++) {  _t_inf[_i] = makevector(201*sizeof(double)); }
  for (_i=0; _i < 2; _i++) {  _t_tau[_i] = makevector(201*sizeof(double)); }
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Cav2_3.mod";
static const char* nmodl_file_text = 
  "TITLE Ca R-type channel with medium threshold for activation\n"
  ": used in distal dendritic regions, together with calH.mod, to help\n"
  ": the generation of Ca++ spikes in these regions\n"
  ": uses channel conductance (not permeability)\n"
  ": written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu\n"
  ": From car to Cav2_3\n"
  "\n"
  "NEURON {\n"
  "	  SUFFIX Cav2_3\n"
  "	  THREADSAFE\n"
  "	  USEION ca READ eca WRITE ica\n"
  "	  RANGE gcabar, m, h, g, gmax, ica\n"
  "	  RANGE inf, tau\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	  (mA) = (milliamp)\n"
  "	  (mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {              : parameters that can be entered when function is called in cell-setup\n"
  "    v             (mV)\n"
  "    celsius = 34	(degC)\n"
  "    gcabar = 0    (mho/cm2) : initialized conductance\n"
  "	  eca = 140     (mV)      : Ca++ reversal potential\n"
  "}  \n"
  "\n"
  "STATE {	m h }            : unknown activation and inactivation parameters to be solved in the DEs  \n"
  "\n"
  "ASSIGNED {               : parameters needed to solve DE\n"
  "	  ica    (mA/cm2)\n"
  "    inf[2]\n"
  "	  tau[2] (ms)\n"
  "    g      (mho/cm2)\n"
  "    gmax   (mho/cm2)\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	  SOLVE states METHOD cnexp\n"
  "    g = gcabar*m*m*m*h\n"
  "	  ica = g*(v - eca)\n"
  "    if (g > gmax) {\n"
  "        gmax = g\n"
  "    }\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    mhn(v)\n"
  "    m = inf[0]\n"
  "    h = inf[1]\n"
  "    g = gcabar*m*m*m*h\n"
  "    ica = g*(v - eca) : initial Ca++ current value\n"
  "    gmax = g\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "	  mhn(v)\n"
  "	  m' =  (inf[0] - m)/tau[0]\n"
  "	  h' =  (inf[1] - h)/tau[1]\n"
  "}	\n"
  "\n"
  "FUNCTION varss(v (mV), i) {\n"
  "	  if (i==0) {\n"
  "	      varss = 1 / (1 + exp((v+48.5(mV))/(-3(mV)))) : Ca activation\n"
  "	  }\n"
  "	  else if (i==1) {\n"
  "        varss = 1/ (1 + exp((v+53(mV))/(1(mV))))    : Ca inactivation\n"
  "	  }\n"
  "}\n"
  "\n"
  "FUNCTION vartau(v (mV), i) (ms) {\n"
  "	  if (i==0) {\n"
  "        vartau = 50  : activation variable time constant\n"
  "    }\n"
  "	  else if (i==1) {\n"
  "        vartau = 5   : inactivation variable time constant\n"
  "    }\n"
  "	  \n"
  "}	\n"
  "\n"
  "PROCEDURE mhn(v (mV)) {LOCAL a, b :rest = -70\n"
  "    TABLE inf, tau DEPEND celsius FROM -100 TO 100 WITH 200\n"
  "  	FROM i=0 TO 1 {\n"
  "	      tau[i] = vartau(v,i)\n"
  "		    inf[i] = varss(v,i)\n"
  "	  }\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
