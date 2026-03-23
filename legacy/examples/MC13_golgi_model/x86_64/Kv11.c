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
 
#define nrn_init _nrn_init__Kv1_1
#define _nrn_initial _nrn_initial__Kv1_1
#define nrn_cur _nrn_cur__Kv1_1
#define _nrn_current _nrn_current__Kv1_1
#define nrn_jacob _nrn_jacob__Kv1_1
#define nrn_state _nrn_state__Kv1_1
#define _net_receive _net_receive__Kv1_1 
#define rates rates__Kv1_1 
#define states states__Kv1_1 
 
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
#define gateCurrent _p[0]
#define gateCurrent_columnindex 0
#define gbar _p[1]
#define gbar_columnindex 1
#define gunit _p[2]
#define gunit_columnindex 2
#define ik _p[3]
#define ik_columnindex 3
#define i _p[4]
#define i_columnindex 4
#define igate _p[5]
#define igate_columnindex 5
#define g _p[6]
#define g_columnindex 6
#define nc _p[7]
#define nc_columnindex 7
#define ninf _p[8]
#define ninf_columnindex 8
#define taun _p[9]
#define taun_columnindex 9
#define n _p[10]
#define n_columnindex 10
#define ek _p[11]
#define ek_columnindex 11
#define alphan _p[12]
#define alphan_columnindex 12
#define betan _p[13]
#define betan_columnindex 13
#define qt _p[14]
#define qt_columnindex 14
#define Dn _p[15]
#define Dn_columnindex 15
#define v _p[16]
#define v_columnindex 16
#define _g _p[17]
#define _g_columnindex 17
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
 static void _hoc_alphanfkt(void);
 static void _hoc_betanfkt(void);
 static void _hoc_ngateFlip(void);
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
 "setdata_Kv1_1", _hoc_setdata,
 "alphanfkt_Kv1_1", _hoc_alphanfkt,
 "betanfkt_Kv1_1", _hoc_betanfkt,
 "ngateFlip_Kv1_1", _hoc_ngateFlip,
 "rates_Kv1_1", _hoc_rates,
 0, 0
};
#define alphanfkt alphanfkt_Kv1_1
#define betanfkt betanfkt_Kv1_1
#define ngateFlip ngateFlip_Kv1_1
 extern double alphanfkt( _threadargsprotocomma_ double );
 extern double betanfkt( _threadargsprotocomma_ double );
 extern double ngateFlip( _threadargsproto_ );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gbar_Kv1_1", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gateCurrent_Kv1_1", "1",
 "gbar_Kv1_1", "S/cm2",
 "gunit_Kv1_1", "pS",
 "ik_Kv1_1", "mA/cm2",
 "i_Kv1_1", "mA/cm2",
 "igate_Kv1_1", "mA/cm2",
 "g_Kv1_1", "S/cm2",
 "nc_Kv1_1", "1/cm2",
 "ninf_Kv1_1", "1",
 "taun_Kv1_1", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
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
"Kv1_1",
 "gateCurrent_Kv1_1",
 "gbar_Kv1_1",
 "gunit_Kv1_1",
 0,
 "ik_Kv1_1",
 "i_Kv1_1",
 "igate_Kv1_1",
 "g_Kv1_1",
 "nc_Kv1_1",
 "ninf_Kv1_1",
 "taun_Kv1_1",
 0,
 "n_Kv1_1",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	gateCurrent = 0;
 	gbar = 0.004;
 	gunit = 16;
 	_prop->param = _p;
 	_prop->param_size = 18;
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
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Kv11_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 18, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kv1_1 /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Kv11.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double e0 = 1.60217646e-19;
 static double q10 = 2.7;
 static double ca = 0.12889;
 static double cva = 45;
 static double cka = -33.90877;
 static double cb = 0.12889;
 static double cvb = 45;
 static double ckb = 12.42101;
 static double zn = 2.7978;
static int _reset;
static char *modelname = "Voltage-gated low threshold potassium current from Kv1 subunits";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / taun ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taun )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taun)))*(- ( ( ( ninf ) ) / taun ) / ( ( ( ( - 1.0 ) ) ) / taun ) - n) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   alphan = alphanfkt ( _threadargscomma_ _lv ) ;
   betan = betanfkt ( _threadargscomma_ _lv ) ;
   ninf = alphan / ( alphan + betan ) ;
   taun = 1.0 / ( qt * ( alphan + betan ) ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double alphanfkt ( _threadargsprotocomma_ double _lv ) {
   double _lalphanfkt;
 _lalphanfkt = ca * exp ( - ( _lv + cva ) / cka ) ;
   
return _lalphanfkt;
 }
 
static void _hoc_alphanfkt(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphanfkt ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betanfkt ( _threadargsprotocomma_ double _lv ) {
   double _lbetanfkt;
 _lbetanfkt = cb * exp ( - ( _lv + cvb ) / ckb ) ;
   
return _lbetanfkt;
 }
 
static void _hoc_betanfkt(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betanfkt ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double ngateFlip ( _threadargsproto_ ) {
   double _lngateFlip;
 _lngateFlip = ( ninf - n ) / taun ;
   
return _lngateFlip;
 }
 
static void _hoc_ngateFlip(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ngateFlip ( _p, _ppvar, _thread, _nt );
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
   nc = ( 1e12 ) * gbar / gunit ;
   qt = pow( q10 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
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
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gbar * pow( n , 4.0 ) ;
   ik = g * ( v - ek ) ;
   igate = nc * ( 1e6 ) * e0 * 4.0 * zn * ngateFlip ( _threadargs_ ) ;
   if ( gateCurrent  != 0.0 ) {
     i = igate ;
     }
   }
 _current += ik;
 _current += i;

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
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = n_columnindex;  _dlist1[0] = Dn_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Kv11.mod";
static const char* nmodl_file_text = 
  "TITLE Voltage-gated low threshold potassium current from Kv1 subunits\n"
  "\n"
  "COMMENT\n"
  "\n"
  "NEURON implementation of a potassium channel from Kv1.1 subunits\n"
  "Kinetical scheme: Hodgkin-Huxley m^4, no inactivation\n"
  "\n"
  "Experimental data taken from:\n"
  "Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998\n"
  "Vhalf = -28.8 +- 2.3 mV; k = 8.1+- 0.9 mV\n"
  "\n"
  "The voltage dependency of the rate constants was approximated by:\n"
  "\n"
  "alpha = ca * exp(-(v+cva)/cka)\n"
  "beta = cb * exp(-(v+cvb)/ckb)\n"
  "\n"
  "Parameters ca, cva, cka, cb, cvb, ckb\n"
  "were determined from least square-fits to experimental data of G/Gmax(v) and tau(v).\n"
  "Values are defined in the CONSTANT block.\n"
  "Model includes calculation of Kv gating current\n"
  "\n"
  "Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976\n"
  "\n"
  "Laboratory for Neuronal Circuit Dynamics\n"
  "RIKEN Brain Science Institute, Wako City, Japan\n"
  "http://www.neurodynamics.brain.riken.jp\n"
  "\n"
  "Date of Implementation: April 2007\n"
  "Contact: akemann@brain.riken.jp\n"
  "\n"
  "Suffix from Kv1 to Kv1_1\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "	SUFFIX Kv1_1\n"
  "	USEION k READ ek WRITE ik\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	RANGE g, gbar, ik, i , igate, nc\n"
  "	RANGE ninf, taun\n"
  "	RANGE gateCurrent, gunit\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "	(nA) = (nanoamp)\n"
  "	(pA) = (picoamp)\n"
  "	(S)  = (siemens)\n"
  "	(nS) = (nanosiemens)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)		\n"
  "}\n"
  "\n"
  "CONSTANT {\n"
  "	e0 = 1.60217646e-19 (coulombs)\n"
  "	q10 = 2.7\n"
  "\n"
  "	ca = 0.12889 (1/ms)\n"
  "	cva = 45 (mV)\n"
  "	cka = -33.90877 (mV)\n"
  "\n"
  "	cb = 0.12889 (1/ms)\n"
  "      cvb = 45 (mV)\n"
  "	ckb = 12.42101 (mV) \n"
  "\n"
  "	zn = 2.7978 (1)		: valence of n-gate         \n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	gateCurrent = 0 (1)	: gating currents ON = 1 OFF = 0\n"
  "	\n"
  "	gbar = 0.004 (S/cm2)   <0,1e9>\n"
  "	gunit = 16 (pS)		: unitary conductance \n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  " 	celsius (degC)\n"
  "	v (mV)	\n"
  "\n"
  "	ik (mA/cm2)\n"
  "	i (mA/cm2)\n"
  "	igate (mA/cm2) \n"
  "	ek (mV)\n"
  "	g  (S/cm2)\n"
  "	nc (1/cm2)			: membrane density of channel\n"
  "	\n"
  "	ninf (1)\n"
  "	taun (ms)\n"
  "	alphan (1/ms)\n"
  "	betan (1/ms)\n"
  "	qt (1)\n"
  "}\n"
  "\n"
  "STATE { n }\n"
  "\n"
  "INITIAL {\n"
  "	nc = (1e12) * gbar / gunit\n"
  "	qt = q10^((celsius-22 (degC))/10 (degC))\n"
  "	rates(v)\n"
  "	n = ninf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "    g = gbar * n^4 \n"
  "	ik = g * (v - ek)\n"
  "	igate = nc * (1e6) * e0 * 4 * zn * ngateFlip()\n"
  "\n"
  "	if (gateCurrent != 0) { \n"
  "		i = igate\n"
  "	}\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "	rates(v)\n"
  "	n' = (ninf-n)/taun \n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) {\n"
  "	alphan = alphanfkt(v)\n"
  "	betan = betanfkt(v)\n"
  "	ninf = alphan/(alphan+betan) \n"
  "	taun = 1/(qt*(alphan + betan))       \n"
  "}\n"
  "\n"
  "FUNCTION alphanfkt(v (mV)) (1/ms) {\n"
  "	alphanfkt = ca * exp(-(v+cva)/cka) \n"
  "}\n"
  "\n"
  "FUNCTION betanfkt(v (mV)) (1/ms) {\n"
  "	betanfkt = cb * exp(-(v+cvb)/ckb)\n"
  "}\n"
  "\n"
  "FUNCTION ngateFlip() (1/ms) {\n"
  "	ngateFlip = (ninf-n)/taun \n"
  "}\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
