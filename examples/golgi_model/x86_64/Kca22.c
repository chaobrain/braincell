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
 
#define nrn_init _nrn_init__Kca2_2
#define _nrn_initial _nrn_initial__Kca2_2
#define nrn_cur _nrn_cur__Kca2_2
#define _nrn_current _nrn_current__Kca2_2
#define nrn_jacob _nrn_jacob__Kca2_2
#define nrn_state _nrn_state__Kca2_2
#define _net_receive _net_receive__Kca2_2 
#define kin kin__Kca2_2 
#define rates rates__Kca2_2 
#define rate rate__Kca2_2 
 
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
#define g _p[1]
#define g_columnindex 1
#define ik _p[2]
#define ik_columnindex 2
#define tcorr _p[3]
#define tcorr_columnindex 3
#define c1 _p[4]
#define c1_columnindex 4
#define c2 _p[5]
#define c2_columnindex 5
#define c3 _p[6]
#define c3_columnindex 6
#define c4 _p[7]
#define c4_columnindex 7
#define o1 _p[8]
#define o1_columnindex 8
#define o2 _p[9]
#define o2_columnindex 9
#define cai _p[10]
#define cai_columnindex 10
#define ek _p[11]
#define ek_columnindex 11
#define invc1_t _p[12]
#define invc1_t_columnindex 12
#define invc2_t _p[13]
#define invc2_t_columnindex 13
#define invc3_t _p[14]
#define invc3_t_columnindex 14
#define invo1_t _p[15]
#define invo1_t_columnindex 15
#define invo2_t _p[16]
#define invo2_t_columnindex 16
#define diro1_t _p[17]
#define diro1_t_columnindex 17
#define diro2_t _p[18]
#define diro2_t_columnindex 18
#define dirc2_t _p[19]
#define dirc2_t_columnindex 19
#define dirc3_t _p[20]
#define dirc3_t_columnindex 20
#define dirc4_t _p[21]
#define dirc4_t_columnindex 21
#define dirc2_t_ca _p[22]
#define dirc2_t_ca_columnindex 22
#define dirc3_t_ca _p[23]
#define dirc3_t_ca_columnindex 23
#define dirc4_t_ca _p[24]
#define dirc4_t_ca_columnindex 24
#define Dc1 _p[25]
#define Dc1_columnindex 25
#define Dc2 _p[26]
#define Dc2_columnindex 26
#define Dc3 _p[27]
#define Dc3_columnindex 27
#define Dc4 _p[28]
#define Dc4_columnindex 28
#define Do1 _p[29]
#define Do1_columnindex 29
#define Do2 _p[30]
#define Do2_columnindex 30
#define v _p[31]
#define v_columnindex 31
#define _g _p[32]
#define _g_columnindex 32
#define _ion_cai	*_ppvar[0]._pval
#define _ion_ek	*_ppvar[1]._pval
#define _ion_ik	*_ppvar[2]._pval
#define _ion_dikdv	*_ppvar[3]._pval
 
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
 static void _hoc_rate(void);
 static void _hoc_temper(void);
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
 "setdata_Kca2_2", _hoc_setdata,
 "rates_Kca2_2", _hoc_rates,
 "rate_Kca2_2", _hoc_rate,
 "temper_Kca2_2", _hoc_temper,
 0, 0
};
#define temper temper_Kca2_2
 extern double temper( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
#define Q10 Q10_Kca2_2
 double Q10 = 3;
#define dirc4 dirc4_Kca2_2
 double dirc4 = 80;
#define dirc3 dirc3_Kca2_2
 double dirc3 = 160;
#define dirc2 dirc2_Kca2_2
 double dirc2 = 200;
#define diro2 diro2_Kca2_2
 double diro2 = 1.2;
#define diro1 diro1_Kca2_2
 double diro1 = 0.16;
#define diff diff_Kca2_2
 double diff = 3;
#define invo2 invo2_Kca2_2
 double invo2 = 0.1;
#define invo1 invo1_Kca2_2
 double invo1 = 1;
#define invc3 invc3_Kca2_2
 double invc3 = 0.2;
#define invc2 invc2_Kca2_2
 double invc2 = 0.08;
#define invc1 invc1_Kca2_2
 double invc1 = 0.08;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Q10_Kca2_2", "1",
 "diff_Kca2_2", "1",
 "invc1_Kca2_2", "/ms",
 "invc2_Kca2_2", "/ms",
 "invc3_Kca2_2", "/ms",
 "invo1_Kca2_2", "/ms",
 "invo2_Kca2_2", "/ms",
 "diro1_Kca2_2", "/ms",
 "diro2_Kca2_2", "/ms",
 "dirc2_Kca2_2", "/ms-mM",
 "dirc3_Kca2_2", "/ms-mM",
 "dirc4_Kca2_2", "/ms-mM",
 "gkbar_Kca2_2", "mho/cm2",
 "g_Kca2_2", "mho/cm2",
 "ik_Kca2_2", "mA/cm2",
 "tcorr_Kca2_2", "1",
 0,0
};
 static double c40 = 0;
 static double c30 = 0;
 static double c20 = 0;
 static double c10 = 0;
 static double delta_t = 0.01;
 static double o20 = 0;
 static double o10 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Q10_Kca2_2", &Q10_Kca2_2,
 "diff_Kca2_2", &diff_Kca2_2,
 "invc1_Kca2_2", &invc1_Kca2_2,
 "invc2_Kca2_2", &invc2_Kca2_2,
 "invc3_Kca2_2", &invc3_Kca2_2,
 "invo1_Kca2_2", &invo1_Kca2_2,
 "invo2_Kca2_2", &invo2_Kca2_2,
 "diro1_Kca2_2", &diro1_Kca2_2,
 "diro2_Kca2_2", &diro2_Kca2_2,
 "dirc2_Kca2_2", &dirc2_Kca2_2,
 "dirc3_Kca2_2", &dirc3_Kca2_2,
 "dirc4_Kca2_2", &dirc4_Kca2_2,
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
"Kca2_2",
 "gkbar_Kca2_2",
 0,
 "g_Kca2_2",
 "ik_Kca2_2",
 "tcorr_Kca2_2",
 0,
 "c1_Kca2_2",
 "c2_Kca2_2",
 "c3_Kca2_2",
 "c4_Kca2_2",
 "o1_Kca2_2",
 "o2_Kca2_2",
 0,
 0};
 static Symbol* _ca_sym;
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 33, _prop);
 	/*initialize range parameters*/
 	gkbar = 0.038;
 	_prop->param = _p;
 	_prop->param_size = 33;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[1]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
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

 void _Kca22_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	ion_reg("k", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_k_sym = hoc_lookup("k_ion");
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
  hoc_register_prop_size(_mechtype, 33, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kca2_2 /home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca22.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "SK2 multi-state model Cerebellum Golgi Cell Model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
static int rate(_threadargsprotocomma_ double);
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[6], _dlist1[6]; static double *_temp1;
 static int kin();
 
static int kin (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<6;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ cai / diff ) ;
   /* ~ c1 <-> c2 ( dirc2_t_ca , invc1_t )*/
 f_flux =  dirc2_t_ca * c1 ;
 b_flux =  invc1_t * c2 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  dirc2_t_ca ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  invc1_t ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ c2 <-> c3 ( dirc3_t_ca , invc2_t )*/
 f_flux =  dirc3_t_ca * c2 ;
 b_flux =  invc2_t * c3 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  dirc3_t_ca ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  invc2_t ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ c3 <-> c4 ( dirc4_t_ca , invc3_t )*/
 f_flux =  dirc4_t_ca * c3 ;
 b_flux =  invc3_t * c4 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  dirc4_t_ca ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  invc3_t ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ c3 <-> o1 ( diro1_t , invo1_t )*/
 f_flux =  diro1_t * c3 ;
 b_flux =  invo1_t * o1 ;
 _RHS1( 2) -= (f_flux - b_flux);
 
 _term =  diro1_t ;
 _MATELM1( 2 ,2)  += _term;
 _term =  invo1_t ;
 _MATELM1( 2 ,0)  -= _term;
 /*REACTION*/
  /* ~ c4 <-> o2 ( diro2_t , invo2_t )*/
 f_flux =  diro2_t * c4 ;
 b_flux =  invo2_t * o2 ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 5) += (f_flux - b_flux);
 
 _term =  diro2_t ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 5 ,1)  -= _term;
 _term =  invo2_t ;
 _MATELM1( 1 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
   /* c1 + c2 + c3 + c4 + o2 + o1 = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= o1 ;
 _MATELM1(0, 5) = 1;
 _RHS1(0) -= o2 ;
 _MATELM1(0, 1) = 1;
 _RHS1(0) -= c4 ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= c3 ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= c2 ;
 _MATELM1(0, 4) = 1;
 _RHS1(0) -= c1 ;
 /*CONSERVATION*/
   } return _reset;
 }
 
double temper ( _threadargsprotocomma_ double _lQ10 , double _lcelsius ) {
   double _ltemper;
 _ltemper = pow( _lQ10 , ( ( _lcelsius - 23.0 ) / 10.0 ) ) ;
   
return _ltemper;
 }
 
static void _hoc_temper(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  temper ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int  rates ( _threadargsprotocomma_ double _lcai ) {
   dirc2_t_ca = dirc2_t * _lcai ;
   dirc3_t_ca = dirc3_t * _lcai ;
   dirc4_t_ca = dirc4_t * _lcai ;
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
 
static int  rate ( _threadargsprotocomma_ double _lcelsius ) {
   tcorr = temper ( _threadargscomma_ Q10 , _lcelsius ) ;
   invc1_t = invc1 * tcorr ;
   invc2_t = invc2 * tcorr ;
   invc3_t = invc3 * tcorr ;
   invo1_t = invo1 * tcorr ;
   invo2_t = invo2 * tcorr ;
   diro1_t = diro1 * tcorr ;
   diro2_t = diro2 * tcorr ;
   dirc2_t = dirc2 * tcorr ;
   dirc3_t = dirc3 * tcorr ;
   dirc4_t = dirc4 * tcorr ;
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<6;_i++) _p[_dlist1[_i]] = 0.0;}
 rates ( _threadargscomma_ cai / diff ) ;
 /* ~ c1 <-> c2 ( dirc2_t_ca , invc1_t )*/
 f_flux =  dirc2_t_ca * c1 ;
 b_flux =  invc1_t * c2 ;
 Dc1 -= (f_flux - b_flux);
 Dc2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ c2 <-> c3 ( dirc3_t_ca , invc2_t )*/
 f_flux =  dirc3_t_ca * c2 ;
 b_flux =  invc2_t * c3 ;
 Dc2 -= (f_flux - b_flux);
 Dc3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ c3 <-> c4 ( dirc4_t_ca , invc3_t )*/
 f_flux =  dirc4_t_ca * c3 ;
 b_flux =  invc3_t * c4 ;
 Dc3 -= (f_flux - b_flux);
 Dc4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ c3 <-> o1 ( diro1_t , invo1_t )*/
 f_flux =  diro1_t * c3 ;
 b_flux =  invo1_t * o1 ;
 Dc3 -= (f_flux - b_flux);
 Do1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ c4 <-> o2 ( diro2_t , invo2_t )*/
 f_flux =  diro2_t * c4 ;
 b_flux =  invo2_t * o2 ;
 Dc4 -= (f_flux - b_flux);
 Do2 += (f_flux - b_flux);
 
 /*REACTION*/
   /* c1 + c2 + c3 + c4 + o2 + o1 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<6;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ cai / diff ) ;
 /* ~ c1 <-> c2 ( dirc2_t_ca , invc1_t )*/
 _term =  dirc2_t_ca ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  invc1_t ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ c2 <-> c3 ( dirc3_t_ca , invc2_t )*/
 _term =  dirc3_t_ca ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  invc2_t ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ c3 <-> c4 ( dirc4_t_ca , invc3_t )*/
 _term =  dirc4_t_ca ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  invc3_t ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ c3 <-> o1 ( diro1_t , invo1_t )*/
 _term =  diro1_t ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 0 ,2)  -= _term;
 _term =  invo1_t ;
 _MATELM1( 2 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ c4 <-> o2 ( diro2_t , invo2_t )*/
 _term =  diro2_t ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 5 ,1)  -= _term;
 _term =  invo2_t ;
 _MATELM1( 1 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
   /* c1 + c2 + c3 + c4 + o2 + o1 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 6;}
 
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
  ek = _ion_ek;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 6; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 6, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
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
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  c4 = c40;
  c3 = c30;
  c2 = c20;
  c1 = c10;
  o2 = o20;
  o1 = o10;
 {
   rate ( _threadargscomma_ celsius ) ;
    _ss_sparse_thread(&_thread[_spth1]._pvoid, 6, _slist1, _dlist1, _p, &t, dt, kin, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 6; ++_i) {
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
  cai = _ion_cai;
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gkbar * ( o1 + o2 ) ;
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
  cai = _ion_cai;
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
  cai = _ion_cai;
  ek = _ion_ek;
 {  sparse_thread(&_thread[_spth1]._pvoid, 6, _slist1, _dlist1, _p, &t, dt, kin, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 6; ++_i) {
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
 _slist1[0] = o1_columnindex;  _dlist1[0] = Do1_columnindex;
 _slist1[1] = c4_columnindex;  _dlist1[1] = Dc4_columnindex;
 _slist1[2] = c3_columnindex;  _dlist1[2] = Dc3_columnindex;
 _slist1[3] = c2_columnindex;  _dlist1[3] = Dc2_columnindex;
 _slist1[4] = c1_columnindex;  _dlist1[4] = Dc1_columnindex;
 _slist1[5] = o2_columnindex;  _dlist1[5] = Do2_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/golgi_model/golgi_NEURON/mod_gol/Kca22.mod";
static const char* nmodl_file_text = 
  "TITLE SK2 multi-state model Cerebellum Golgi Cell Model\n"
  "\n"
  "COMMENT\n"
  "\n"
  "Author:Sergio Solinas, Lia Forti, Egidio DAngelo\n"
  "Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998\n"
  "Last revised: May 2007\n"
  "\n"
  "Published in:\n"
  "             Sergio M. Solinas, Lia Forti, Elisabetta Cesana, \n"
  "             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)\n"
  "             Computational reconstruction of pacemaking and intrinsic \n"
  "             electroresponsiveness in cerebellar golgi cells\n"
  "             Frontiers in Cellular Neuroscience 2:2\n"
  "\n"
  "Suffix from SK2 to Kca2_2\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON{\n"
  "	SUFFIX Kca2_2\n"
  "	USEION ca READ cai\n"
  "	USEION k READ ek WRITE ik \n"
  "	RANGE gkbar, g, ik, tcorr\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	celsius  (degC)\n"
  "	cai (mM)\n"
  "	gkbar = 0.038 (mho/cm2)\n"
  "	Q10 = 3 (1)\n"
  "	diff = 3 (1) : diffusion factor\n"
  "\n"
  ": rates ca-indipendent\n"
  "	invc1 = 80e-3  ( /ms)\n"
  "	invc2 = 80e-3  ( /ms)\n"
  "	invc3 = 200e-3 ( /ms)\n"
  "\n"
  "	invo1 = 1      ( /ms)\n"
  "	invo2 = 100e-3 ( /ms)\n"
  "	diro1 = 160e-3 ( /ms)\n"
  "	diro2 = 1.2    ( /ms)\n"
  "\n"
  ": rates ca-dipendent\n"
  "	dirc2 = 200 ( /ms-mM )\n"
  "	dirc3 = 160 ( /ms-mM )\n"
  "	dirc4 = 80  ( /ms-mM )\n"
  "\n"
  "}\n"
  "\n"
  "ASSIGNED{ \n"
  "	v	(mV) \n"
  "	ek	(mV) \n"
  "	g	(mho/cm2) \n"
  "	ik	(mA/cm2) \n"
  "	invc1_t  ( /ms)\n"
  "	invc2_t  ( /ms)\n"
  "	invc3_t  ( /ms)\n"
  "	invo1_t  ( /ms)\n"
  "	invo2_t  ( /ms)\n"
  "	diro1_t  ( /ms)\n"
  "	diro2_t  ( /ms)\n"
  "	dirc2_t  ( /ms-mM)\n"
  "	dirc3_t  ( /ms-mM)\n"
  "	dirc4_t  ( /ms-mM)\n"
  "	tcorr	 (1)\n"
  "\n"
  "	dirc2_t_ca  ( /ms)\n"
  "	dirc3_t_ca  ( /ms)\n"
  "	dirc4_t_ca  ( /ms)\n"
  "} \n"
  "\n"
  "STATE {\n"
  "	c1\n"
  "	c2\n"
  "	c3\n"
  "	c4\n"
  "	o1\n"
  "	o2\n"
  "}\n"
  "\n"
  "BREAKPOINT{ \n"
  "	SOLVE kin METHOD sparse \n"
  "	g = gkbar*(o1+o2)	:(mho/cm2)\n"
  "	ik = g*(v-ek)		:(mA/cm2)\n"
  "} \n"
  "\n"
  "INITIAL{\n"
  "	rate(celsius)\n"
  "	SOLVE kin STEADYSTATE sparse\n"
  "} \n"
  "\n"
  "KINETIC kin{ \n"
  "	rates(cai/diff) \n"
  "	~c1<->c2 (dirc2_t_ca, invc1_t) \n"
  "	~c2<->c3 (dirc3_t_ca, invc2_t) \n"
  "	~c3<->c4 (dirc4_t_ca, invc3_t) \n"
  "	~c3<->o1 (diro1_t, invo1_t) \n"
  "	~c4<->o2 (diro2_t, invo2_t) \n"
  "	CONSERVE c1+c2+c3+c4+o2+o1=1 \n"
  "} \n"
  "\n"
  "FUNCTION temper (Q10, celsius (degC)) {\n"
  "	temper = Q10^((celsius -23(degC)) / 10(degC)) \n"
  "}\n"
  "\n"
  "PROCEDURE rates(cai(mM)){\n"
  "	dirc2_t_ca = dirc2_t*cai\n"
  "	dirc3_t_ca = dirc3_t*cai\n"
  "	dirc4_t_ca = dirc4_t*cai \n"
  "} \n"
  "\n"
  "PROCEDURE rate (celsius(degC)) {\n"
  "	tcorr = temper (Q10,celsius)\n"
  "	invc1_t = invc1*tcorr  \n"
  "	invc2_t = invc2*tcorr\n"
  "	invc3_t = invc3*tcorr \n"
  "	invo1_t = invo1*tcorr \n"
  "	invo2_t = invo2*tcorr \n"
  "	diro1_t = diro1*tcorr \n"
  "	diro2_t = diro2*tcorr \n"
  "	dirc2_t = dirc2*tcorr\n"
  "	dirc3_t = dirc3*tcorr\n"
  "	dirc4_t = dirc4*tcorr\n"
  "}\n"
  ;
#endif
