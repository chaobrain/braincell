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
 
#define nrn_init _nrn_init__Kv3_4
#define _nrn_initial _nrn_initial__Kv3_4
#define nrn_cur _nrn_cur__Kv3_4
#define _nrn_current _nrn_current__Kv3_4
#define nrn_jacob _nrn_jacob__Kv3_4
#define nrn_state _nrn_state__Kv3_4
#define _net_receive _net_receive__Kv3_4 
#define rates rates__Kv3_4 
#define states states__Kv3_4 
 
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
#define ik _p[1]
#define ik_columnindex 1
#define minf _p[2]
#define minf_columnindex 2
#define mtau _p[3]
#define mtau_columnindex 3
#define hinf _p[4]
#define hinf_columnindex 4
#define htau _p[5]
#define htau_columnindex 5
#define m _p[6]
#define m_columnindex 6
#define h _p[7]
#define h_columnindex 7
#define ek _p[8]
#define ek_columnindex 8
#define qt _p[9]
#define qt_columnindex 9
#define Dm _p[10]
#define Dm_columnindex 10
#define Dh _p[11]
#define Dh_columnindex 11
#define v _p[12]
#define v_columnindex 12
#define _g _p[13]
#define _g_columnindex 13
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
 static void _hoc_htau_func(void);
 static void _hoc_mtau_func(void);
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
 "setdata_Kv3_4", _hoc_setdata,
 "htau_func_Kv3_4", _hoc_htau_func,
 "mtau_func_Kv3_4", _hoc_mtau_func,
 "rates_Kv3_4", _hoc_rates,
 0, 0
};
#define htau_func htau_func_Kv3_4
#define mtau_func mtau_func_Kv3_4
 extern double htau_func( _threadargsprotocomma_ double );
 extern double mtau_func( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define hik hik_Kv3_4
 double hik = 11.2;
#define hivh hivh_Kv3_4
 double hivh = -5.802;
#define hiA hiA_Kv3_4
 double hiA = 0.69;
#define hiy0 hiy0_Kv3_4
 double hiy0 = 0.31;
#define mtk2 mtk2_Kv3_4
 double mtk2 = -23.1;
#define mtvh2 mtvh2_Kv3_4
 double mtvh2 = -56;
#define mtk1 mtk1_Kv3_4
 double mtk1 = 12.9;
#define mtvh1 mtvh1_Kv3_4
 double mtvh1 = 100.7;
#define mty0 mty0_Kv3_4
 double mty0 = 0.00012851;
#define mik mik_Kv3_4
 double mik = 15.4;
#define mivh mivh_Kv3_4
 double mivh = -24;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "mivh_Kv3_4", "mV",
 "mik_Kv3_4", "1",
 "mtvh1_Kv3_4", "mV",
 "mtk1_Kv3_4", "1",
 "mtvh2_Kv3_4", "mV",
 "mtk2_Kv3_4", "1",
 "hivh_Kv3_4", "mV",
 "hik_Kv3_4", "1",
 "gkbar_Kv3_4", "mho/cm2",
 "ik_Kv3_4", "mA/cm2",
 "mtau_Kv3_4", "ms",
 "htau_Kv3_4", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "mivh_Kv3_4", &mivh_Kv3_4,
 "mik_Kv3_4", &mik_Kv3_4,
 "mty0_Kv3_4", &mty0_Kv3_4,
 "mtvh1_Kv3_4", &mtvh1_Kv3_4,
 "mtk1_Kv3_4", &mtk1_Kv3_4,
 "mtvh2_Kv3_4", &mtvh2_Kv3_4,
 "mtk2_Kv3_4", &mtk2_Kv3_4,
 "hiy0_Kv3_4", &hiy0_Kv3_4,
 "hiA_Kv3_4", &hiA_Kv3_4,
 "hivh_Kv3_4", &hivh_Kv3_4,
 "hik_Kv3_4", &hik_Kv3_4,
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
"Kv3_4",
 "gkbar_Kv3_4",
 0,
 "ik_Kv3_4",
 "minf_Kv3_4",
 "mtau_Kv3_4",
 "hinf_Kv3_4",
 "htau_Kv3_4",
 0,
 "m_Kv3_4",
 "h_Kv3_4",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	gkbar = 0.004;
 	_prop->param = _p;
 	_prop->param_size = 14;
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

 void _Kv34_reg() {
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
  hoc_register_prop_size(_mechtype, 14, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kv3_4 /home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kv34.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double q10 = 3;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lVm ) {
   double _lv ;
 _lv = _lVm + 11.0 ;
   minf = 1.0 / ( 1.0 + exp ( - ( _lv - mivh ) / mik ) ) ;
   mtau = ( 1000.0 ) * mtau_func ( _threadargscomma_ _lv ) / qt ;
   hinf = hiy0 + hiA / ( 1.0 + exp ( ( _lv - hivh ) / hik ) ) ;
   htau = 1000.0 * htau_func ( _threadargscomma_ _lv ) / qt ;
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
 
double mtau_func ( _threadargsprotocomma_ double _lv ) {
   double _lmtau_func;
 if ( _lv < - 35.0 ) {
     _lmtau_func = ( 3.4225e-5 + .00498 * exp ( - _lv / - 28.29 ) ) * 3.0 ;
     }
   else {
     _lmtau_func = ( mty0 + 1.0 / ( exp ( ( _lv + mtvh1 ) / mtk1 ) + exp ( ( _lv + mtvh2 ) / mtk2 ) ) ) ;
     }
   
return _lmtau_func;
 }
 
static void _hoc_mtau_func(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  mtau_func ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double htau_func ( _threadargsprotocomma_ double _lVm ) {
   double _lhtau_func;
 if ( _lVm > 0.0 ) {
     _lhtau_func = .0012 + .0023 * exp ( - .141 * _lVm ) ;
     }
   else {
     _lhtau_func = 1.2202e-05 + .012 * exp ( - pow( ( ( _lVm - ( - 56.3 ) ) / 49.6 ) , 2.0 ) ) ;
     }
   
return _lhtau_func;
 }
 
static void _hoc_htau_func(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  htau_func ( _p, _ppvar, _thread, _nt, *getarg(1) );
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
  ek = _ion_ek;
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
  h = h0;
  m = m0;
 {
   rates ( _threadargscomma_ v ) ;
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
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ik = gkbar * pow( m , 3.0 ) * h * ( v - ek ) ;
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
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kv34.mod";
static const char* nmodl_file_text = 
  ": HH TEA-sensitive Purkinje potassium current\n"
  ": Created 8/5/02 - nwg\n"
  "\n"
  ": Suffix from kpkj to Kv3_4\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX Kv3_4\n"
  "	USEION k READ ek WRITE ik\n"
  "	RANGE gkbar, ik\n"
  "	RANGE minf, hinf, mtau, htau\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "}\n"
  "\n"
  "CONSTANT {\n"
  "	q10 = 3\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v (mV)\n"
  "\n"
  "	gkbar = .004	(mho/cm2)\n"
  "\n"
  "	mivh = -24	(mV)\n"
  "	mik = 15.4	(1)\n"
  "	mty0 = .00012851 	\n"
  "	mtvh1 = 100.7	(mV)\n"
  "	mtk1 = 12.9	(1)\n"
  "	mtvh2 = -56.0	(mV)\n"
  "	mtk2 = -23.1	(1)\n"
  "	\n"
  "	hiy0 = .31	\n"
  "	hiA = .69\n"
  "	hivh = -5.802	(mV)\n"
  "	hik = 11.2	(1)\n"
  "\n"
  "	ek\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ik		(mA/cm2)\n"
  "	minf\n"
  "	mtau		(ms)\n"
  "	hinf\n"
  "	htau		(ms)\n"
  "        qt\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m\n"
  "	h\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	m = minf\n"
  "	h = hinf\n"
  "\n"
  "	qt = q10^((celsius-37 (degC))/10 (degC))\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	ik = gkbar * m^3 * h * (v - ek)\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "	rates(v)\n"
  "	m' = (minf - m) / mtau\n"
  "	h' = (hinf - h) / htau\n"
  "}\n"
  "\n"
  "PROCEDURE rates( Vm (mV)) {\n"
  "	LOCAL v\n"
  "	v = Vm + 11	: Account for Junction Potential\n"
  "	minf = 1/(1+exp(-(v-mivh)/mik)) \n"
  "	mtau = (1000) * mtau_func(v) /qt\n"
  "	hinf = hiy0 + hiA/(1+exp((v-hivh)/hik))\n"
  "	htau = 1000 * htau_func(v) / qt\n"
  "}\n"
  "\n"
  "FUNCTION mtau_func (v (mV)) (ms) {\n"
  "	if (v < -35) {\n"
  "		mtau_func = (3.4225e-5+.00498*exp(-v/-28.29))*3\n"
  "	} else {\n"
  "		mtau_func = (mty0 + 1/(exp((v+mtvh1)/mtk1)+exp((v+mtvh2)/mtk2)))\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION htau_func(Vm (mV)) (ms) {\n"
  "	if ( Vm > 0) {\n"
  "		htau_func = .0012+.0023*exp(-.141*Vm)\n"
  "	} else {\n"
  "		htau_func = 1.2202e-05 + .012 * exp(-((Vm-(-56.3))/49.6)^2)\n"
  "	}\n"
  "}\n"
  "	\n"
  ;
#endif
