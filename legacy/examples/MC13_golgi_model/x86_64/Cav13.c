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
 
#define nrn_init _nrn_init__Cav13
#define _nrn_initial _nrn_initial__Cav13
#define nrn_cur _nrn_cur__Cav13
#define _nrn_current _nrn_current__Cav13
#define nrn_jacob _nrn_jacob__Cav13
#define nrn_state _nrn_state__Cav13
#define _net_receive _net_receive__Cav13 
#define rates rates__Cav13 
#define state state__Cav13 
 
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
#define g _p[1]
#define g_columnindex 1
#define m _p[2]
#define m_columnindex 2
#define h _p[3]
#define h_columnindex 3
#define h2 _p[4]
#define h2_columnindex 4
#define ilca _p[5]
#define ilca_columnindex 5
#define ica _p[6]
#define ica_columnindex 6
#define eca _p[7]
#define eca_columnindex 7
#define cai _p[8]
#define cai_columnindex 8
#define mInf _p[9]
#define mInf_columnindex 9
#define hInf _p[10]
#define hInf_columnindex 10
#define h2Inf _p[11]
#define h2Inf_columnindex 11
#define mTau _p[12]
#define mTau_columnindex 12
#define Dm _p[13]
#define Dm_columnindex 13
#define Dh _p[14]
#define Dh_columnindex 14
#define Dh2 _p[15]
#define Dh2_columnindex 15
#define v _p[16]
#define v_columnindex 16
#define _g _p[17]
#define _g_columnindex 17
#define _ion_cai	*_ppvar[0]._pval
#define _ion_eca	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
#define _ion_ilca	*_ppvar[4]._pval
#define _ion_dilcadv	*_ppvar[5]._pval
#define diam	*_ppvar[6]._pval
 
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
 "setdata_Cav13", _hoc_setdata,
 "rates_Cav13", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define VDI VDI_Cav13
 double VDI = 1;
#define hTau hTau_Cav13
 double hTau = 44.3;
#define h2Tau h2Tau_Cav13
 double h2Tau = 0.5;
#define kf kf_Cav13
 double kf = 0.0005;
#define vshift vshift_Cav13
 double vshift = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "hTau_Cav13", "ms",
 "h2Tau_Cav13", "ms",
 "vshift_Cav13", "mV",
 "kf_Cav13", "mM",
 "gbar_Cav13", "S/cm2",
 "g_Cav13", "S/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h20 = 0;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "hTau_Cav13", &hTau_Cav13,
 "h2Tau_Cav13", &h2Tau_Cav13,
 "vshift_Cav13", &vshift_Cav13,
 "kf_Cav13", &kf_Cav13,
 "VDI_Cav13", &VDI_Cav13,
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
 
#define _cvode_ieq _ppvar[7]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Cav13",
 "gbar_Cav13",
 0,
 "g_Cav13",
 0,
 "m_Cav13",
 "h_Cav13",
 "h2_Cav13",
 0,
 0};
 static Symbol* _morphology_sym;
 static Symbol* _ca_sym;
 static Symbol* _lca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	gbar = 0;
 	_prop->param = _p;
 	_prop->param_size = 18;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 8, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_morphology_sym);
 	_ppvar[6]._pval = &prop_ion->param[0]; /* diam */
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 1);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[0]; /* eca */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 prop_ion = need_memb(_lca_sym);
 	_ppvar[4]._pval = &prop_ion->param[3]; /* ilca */
 	_ppvar[5]._pval = &prop_ion->param[4]; /* _ion_dilcadv */
 
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

 void _Cav13_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("lca", 0.0);
 	_morphology_sym = hoc_lookup("morphology");
 	_ca_sym = hoc_lookup("ca_ion");
 	_lca_sym = hoc_lookup("lca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 18, 8);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "lca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "lca_ion");
  hoc_register_dparam_semantics(_mechtype, 7, "cvodeieq");
  hoc_register_dparam_semantics(_mechtype, 6, "diam");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Cav13 /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Cav13.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsproto_);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int state(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   Dm = ( mInf - m ) / mTau ;
   Dh = ( hInf - h ) / hTau ;
   Dh2 = ( h2Inf - h2 ) / h2Tau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mTau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / hTau )) ;
 Dh2 = Dh2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / h2Tau )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mTau)))*(- ( ( ( mInf ) ) / mTau ) / ( ( ( ( - 1.0 ) ) ) / mTau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / hTau)))*(- ( ( ( hInf ) ) / hTau ) / ( ( ( ( - 1.0 ) ) ) / hTau ) - h) ;
    h2 = h2 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / h2Tau)))*(- ( ( ( h2Inf ) ) / h2Tau ) / ( ( ( ( - 1.0 ) ) ) / h2Tau ) - h2) ;
   }
  return 0;
}
 
static int  rates ( _threadargsproto_ ) {
   double _lmA , _lmB ;
 _lmA = ( 39800.0 * ( v + 67.24 ) ) / ( exp ( ( v + 67.24 ) / 15.005 ) - 1.0 ) ;
   _lmB = 3500.0 * exp ( v / 31.4 ) ;
   mTau = ( 1.0 / ( _lmA + _lmB ) ) ;
   mInf = 1.0 / ( ( exp ( ( v - ( - 40.0 ) ) / ( - 5.0 ) ) ) + 1.0 ) ;
   hInf = VDI / ( ( exp ( ( v - ( - 37.0 ) ) / ( 5.0 ) ) ) + 1.0 ) + ( 1.0 - VDI ) ;
   h2Inf = kf / ( kf + cai ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
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
  eca = _ion_eca;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
   }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
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
  eca = _ion_eca;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
   nrn_update_ion_pointer(_lca_sym, _ppvar, 4, 3);
   nrn_update_ion_pointer(_lca_sym, _ppvar, 5, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h2 = h20;
  h = h0;
  m = m0;
 {
   rates ( _threadargs_ ) ;
   m = mInf ;
   h = hInf ;
   h2 = h2Inf ;
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
  eca = _ion_eca;
 initmodel(_p, _ppvar, _thread, _nt);
  }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   rates ( _threadargs_ ) ;
   g = gbar * m * h * h2 ;
   ica = ( g ) * ( v - eca ) ;
   ilca = ica ;
   }
 _current += ica;
 _current += ilca;

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
  eca = _ion_eca;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dilca;
 double _dica;
  _dica = ica;
  _dilca = ilca;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
  _ion_dilcadv += (_dilca - ilca)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
  _ion_ilca += ilca ;
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
  eca = _ion_eca;
 {   state(_p, _ppvar, _thread, _nt);
  }  }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
 _slist1[2] = h2_columnindex;  _dlist1[2] = Dh2_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/Cav13.mod";
static const char* nmodl_file_text = 
  ": model from Evans et al 2013, transferred from GENESIS to NEURON by Beining et al (2016), \"A novel comprehensive and consistent electrophysiologcal model of dentate granule cells\"\n"
  ": also added Calcium dependent inactivation\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX Cav13\n"
  "	USEION ca READ cai, eca WRITE ica   :,cai,cao...., cai, cao\n"
  "	USEION lca WRITE ilca VALENCE 0\n"
  "	RANGE gbar, g\n"
  "	GLOBAL kf, h2Tau, VDI\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "	(S) = (siemens)\n"
  "	(um) = (micrometer)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ilca		(mA/cm2) : instantaneous calcium current of l-type calcium channel\n"
  "	v			(mV)\n"
  "	ica		(mA/cm2)\n"
  "	g		(S/cm2)\n"
  "	eca 		(mV)\n"
  "	diam		(um)\n"
  "	cai 		(mM)\n"
  "	mInf  (1)\n"
  "	hInf  (1)\n"
  "	h2Inf (1)\n"
  "	mTau (ms)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	hTau 	= 44.3 (ms)\n"
  "	h2Tau = 0.5 (ms)\n"
  "	gbar = 0	(S/cm2)\n"
  "		vshift = 0 		(mV)\n"
  "		\n"
  "		:parameters for calcium-dep inactivation (CDI) \n"
  "			:f= (0.001/(0.001+[Ca]))Poirazi CA1  2003\n"
  "			:f= (0.0005/(0.0005+[Ca])) Rhodes and Llinas 2001 Cort Pyr\n"
  "	kf		=			0.0005 (mM)  : factor in inactivation, the higher the less sensitive. others uses 0.0002.. standen and stanfield use 0.001mM in original paper	\n"
  "	VDI = 1\n"
  "}\n"
  "\n"
  "STATE {m h h2}  :a b  :cai (mM) cao (mM)\n"
  "\n"
  "INITIAL {\n"
  "	rates()\n"
  "	m = mInf\n"
  "	h = hInf\n"
  "	h2 = h2Inf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	rates()\n"
  "	SOLVE state METHOD cnexp\n"
  "	g = gbar*m*h*h2 : h2 calcium dependent inactivation is taken from santhakumar 05.. tjos assumes instantaneous calcium inactivation\n"
  "	ica = (g)*(v - eca) : \n"
  "	ilca = ica\n"
  "	\n"
  "}\n"
  "\n"
  "DERIVATIVE state {	: exact when v held constant integrates over dt step\n"
  "	m' = (mInf-m) / mTau\n"
  "	h' = (hInf-h) / hTau\n"
  "	h2' = (h2Inf-h2)/h2Tau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(){\n"
  "		LOCAL mA,mB\n"
  "		mA = (39800*( v + 67.24))/( exp ( (v + 67.24)/15.005) - 1.0)\n"
  "		mB = 3500* exp(v/31.4) \n"
  "		mTau = (1/(mA + mB))\n"
  "		\n"
  "		mInf = 1.0/((exp ( (v - (-40.0))/(-5))) + 1.0)\n"
  "\n"
  "		hInf = VDI/( (exp ( (v - (-37))/(5))) + 1.0) + (1-VDI)\n"
  "		:h2 = caIn(cai)\n"
  "		h2Inf = kf/(kf+cai)\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
