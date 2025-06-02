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
 
#define nrn_init _nrn_init__Golgi_MF_syn
#define _nrn_initial _nrn_initial__Golgi_MF_syn
#define nrn_cur _nrn_cur__Golgi_MF_syn
#define _nrn_current _nrn_current__Golgi_MF_syn
#define nrn_jacob _nrn_jacob__Golgi_MF_syn
#define nrn_state _nrn_state__Golgi_MF_syn
#define _net_receive _net_receive__Golgi_MF_syn 
#define kstates kstates__Golgi_MF_syn 
 
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
#define syntype _p[0]
#define syntype_columnindex 0
#define gmax_factor _p[1]
#define gmax_factor_columnindex 1
#define Q10_diff _p[2]
#define Q10_diff_columnindex 2
#define Q10_channel _p[3]
#define Q10_channel_columnindex 3
#define gmax _p[4]
#define gmax_columnindex 4
#define U _p[5]
#define U_columnindex 5
#define tau_rec _p[6]
#define tau_rec_columnindex 6
#define tau_facil _p[7]
#define tau_facil_columnindex 7
#define M _p[8]
#define M_columnindex 8
#define Rd _p[9]
#define Rd_columnindex 9
#define Diff _p[10]
#define Diff_columnindex 10
#define Cdur _p[11]
#define Cdur_columnindex 11
#define r1FIX _p[12]
#define r1FIX_columnindex 12
#define r2 _p[13]
#define r2_columnindex 13
#define r3 _p[14]
#define r3_columnindex 14
#define r4 _p[15]
#define r4_columnindex 15
#define r5 _p[16]
#define r5_columnindex 16
#define r6FIX _p[17]
#define r6FIX_columnindex 17
#define Erev _p[18]
#define Erev_columnindex 18
#define kB _p[19]
#define kB_columnindex 19
#define tau_1 _p[20]
#define tau_1_columnindex 20
#define Tmax _p[21]
#define Tmax_columnindex 21
#define diffuse _p[22]
#define diffuse_columnindex 22
#define lamd _p[23]
#define lamd_columnindex 23
#define nd _p[24]
#define nd_columnindex 24
#define i _p[25]
#define i_columnindex 25
#define ic _p[26]
#define ic_columnindex 26
#define g _p[27]
#define g_columnindex 27
#define r1 _p[28]
#define r1_columnindex 28
#define r6 _p[29]
#define r6_columnindex 29
#define T _p[30]
#define T_columnindex 30
#define Trelease _p[31]
#define Trelease_columnindex 31
#define tspike (_p + 32)
#define tspike_columnindex 32
#define PRE (_p + 132)
#define PRE_columnindex 132
#define yuno _p[232]
#define yuno_columnindex 232
#define C _p[233]
#define C_columnindex 233
#define O _p[234]
#define O_columnindex 234
#define D _p[235]
#define D_columnindex 235
#define x _p[236]
#define x_columnindex 236
#define tsyn _p[237]
#define tsyn_columnindex 237
#define Mres _p[238]
#define Mres_columnindex 238
#define numpulses _p[239]
#define numpulses_columnindex 239
#define tzero _p[240]
#define tzero_columnindex 240
#define gbar_Q10 _p[241]
#define gbar_Q10_columnindex 241
#define Q10 _p[242]
#define Q10_columnindex 242
#define DC _p[243]
#define DC_columnindex 243
#define DO _p[244]
#define DO_columnindex 244
#define DD _p[245]
#define DD_columnindex 245
#define v _p[246]
#define v_columnindex 246
#define _g _p[247]
#define _g_columnindex 247
#define _tsav _p[248]
#define _tsav_columnindex 248
#define _nd_area  *_ppvar[0]._pval
 
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
 static double _hoc_diffusione(void*);
 static double _hoc_imax(void*);
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

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "diffusione", _hoc_diffusione,
 "imax", _hoc_imax,
 0, 0
};
#define diffusione diffusione_Golgi_MF_syn
#define imax imax_Golgi_MF_syn
 extern double diffusione( _threadargsproto_ );
 extern double imax( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
#define u0 u0_Golgi_MF_syn
 double u0 = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "U", 0, 1,
 "tau_facil", 0, 1e+09,
 "tau_rec", 1e-09, 1e+09,
 "tau_1", 1e-09, 1e+09,
 "u0_Golgi_MF_syn", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "u0_Golgi_MF_syn", "1",
 "gmax", "pS",
 "U", "1",
 "tau_rec", "ms",
 "tau_facil", "ms",
 "Rd", "um",
 "Diff", "um2/ms",
 "Cdur", "ms",
 "r1FIX", "/ms/mM",
 "r2", "/ms",
 "r3", "/ms",
 "r4", "/ms",
 "r5", "/ms",
 "r6FIX", "/ms/mM",
 "Erev", "mV",
 "kB", "mM",
 "tau_1", "ms",
 "Tmax", "mM",
 "lamd", "nm",
 "i", "nA",
 "ic", "nA",
 "g", "pS",
 "r1", "/ms",
 "r6", "/ms",
 "T", "mM",
 "Trelease", "mM",
 "tspike", "ms",
 0,0
};
 static double C0 = 0;
 static double D0 = 0;
 static double O0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "u0_Golgi_MF_syn", &u0_Golgi_MF_syn,
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
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Golgi_MF_syn",
 "syntype",
 "gmax_factor",
 "Q10_diff",
 "Q10_channel",
 "gmax",
 "U",
 "tau_rec",
 "tau_facil",
 "M",
 "Rd",
 "Diff",
 "Cdur",
 "r1FIX",
 "r2",
 "r3",
 "r4",
 "r5",
 "r6FIX",
 "Erev",
 "kB",
 "tau_1",
 "Tmax",
 "diffuse",
 "lamd",
 "nd",
 0,
 "i",
 "ic",
 "g",
 "r1",
 "r6",
 "T",
 "Trelease",
 "tspike[100]",
 "PRE[100]",
 "yuno",
 0,
 "C",
 "O",
 "D",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 249, _prop);
 	/*initialize range parameters*/
 	syntype = 0;
 	gmax_factor = 1;
 	Q10_diff = 1.5;
 	Q10_channel = 2.4;
 	gmax = 2000;
 	U = 0.2;
 	tau_rec = 8;
 	tau_facil = 5;
 	M = 21.515;
 	Rd = 1.03;
 	Diff = 0.223;
 	Cdur = 0.3;
 	r1FIX = 5.4;
 	r2 = 0.82;
 	r3 = 0;
 	r4 = 0;
 	r5 = 0.013;
 	r6FIX = 1.12;
 	Erev = 0;
 	kB = 0.44;
 	tau_1 = 1;
 	Tmax = 1;
 	diffuse = 1;
 	lamd = 20;
 	nd = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 249;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _GOLGI_Ampa_mossy_det_vi_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 3,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
  _extcall_thread = (Datum*)ecalloc(2, sizeof(Datum));
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 249, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 8;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Golgi_MF_syn /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/GOLGI_Ampa_mossy_det_vi.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define PI _nrnunit_PI[_nrnunit_use_legacy_]
static double _nrnunit_PI[2] = {0x1.921fb54442d18p+1, 3.14159}; /* 3.14159265358979312 */
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3]; static double *_temp1;
 static int kstates();
 
double imax ( _threadargsprotocomma_ double _la , double _lb ) {
   double _limax;
 if ( _la > _lb ) {
     _limax = _la ;
     }
   else {
     _limax = _lb ;
     }
   
return _limax;
 }
 
static double _hoc_imax(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  imax ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
double diffusione ( _threadargsproto_ ) {
   double _ldiffusione;
 double _lDifWave , _li , _lcntc , _lfi , _laaa ;
 _lDifWave = 0.0 ;
   _lcntc = imax ( _threadargscomma_ numpulses - 100.0 , 0.0 ) ;
   {int  _li ;for ( _li = ((int) _lcntc ) ; _li <= ((int) numpulses ) ; _li ++ ) {
     _lfi = fmod ( ((double) _li ) , 100.0 ) ;
     tzero = tspike [ ((int) _lfi ) ] ;
     if ( t > tzero ) {
       _laaa = ( - Rd * Rd / ( 4.0 * Diff * ( t - tzero ) ) ) ;
       if ( fabs ( _laaa ) < 699.0 ) {
         _lDifWave = _lDifWave + PRE [ ((int) _lfi ) ] * Mres * exp ( _laaa ) / ( ( 4.0 * PI * Diff * ( 1e-3 ) * lamd ) * ( t - tzero ) ) ;
         }
       else {
         if ( _laaa > 0.0 ) {
           _lDifWave = _lDifWave + PRE [ ((int) _lfi ) ] * Mres * exp ( 699.0 ) / ( ( 4.0 * PI * Diff * ( 1e-3 ) * lamd ) * ( t - tzero ) ) ;
           }
         else {
           _lDifWave = _lDifWave + PRE [ ((int) _lfi ) ] * Mres * exp ( - 699.0 ) / ( ( 4.0 * PI * Diff * ( 1e-3 ) * lamd ) * ( t - tzero ) ) ;
           }
         }
       }
     } }
   _ldiffusione = _lDifWave ;
   
return _ldiffusione;
 }
 
static double _hoc_diffusione(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  diffusione ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int kstates (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<3;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 r1 = r1FIX * pow( Trelease , 2.0 ) / pow( ( Trelease + kB ) , 2.0 ) ;
   r6 = r6FIX * pow( Trelease , 2.0 ) / pow( ( Trelease + kB ) , 2.0 ) ;
   /* ~ C <-> O ( r1 * Q10 , r2 * Q10 )*/
 f_flux =  r1 * Q10 * C ;
 b_flux =  r2 * Q10 * O ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  r1 * Q10 ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 2 ,1)  -= _term;
 _term =  r2 * Q10 ;
 _MATELM1( 1 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ D <-> C ( r5 * Q10 , r6 * Q10 )*/
 f_flux =  r5 * Q10 * D ;
 b_flux =  r6 * Q10 * C ;
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  r5 * Q10 ;
 _MATELM1( 1 ,0)  -= _term;
 _term =  r6 * Q10 ;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
   /* C + O + D = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= D ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= O ;
 _MATELM1(0, 1) = 1;
 _RHS1(0) -= C ;
 /*CONSERVATION*/
   } return _reset;
 }
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   double _lfi ;
 if ( _lflag  == 0.0 ) {
     _args[2] = _args[2] + 1.0 ;
     if (  ! _args[1] ) {
       _args[3] = t ;
       _args[1] = 1.0 ;
       _args[5] = _args[5] * exp ( - ( t - _args[7] ) / ( tau_rec ) ) ;
       _args[5] = _args[5] + ( _args[4] * ( exp ( - ( t - _args[7] ) / tau_1 ) - exp ( - ( t - _args[7] ) / ( tau_rec ) ) ) / ( ( tau_1 / ( tau_rec ) ) - 1.0 ) ) ;
       _args[4] = _args[4] * exp ( - ( t - _args[7] ) / tau_1 ) ;
       x = 1.0 - _args[4] - _args[5] ;
       if ( tau_facil > 0.0 ) {
         _args[6] = _args[6] * exp ( - ( t - _args[7] ) / tau_facil ) ;
         _args[6] = _args[6] + U * ( 1.0 - _args[6] ) ;
         }
       else {
         _args[6] = U ;
         }
       _args[4] = _args[4] + x * _args[6] ;
       T = Tmax * _args[4] ;
       yuno = _args[4] ;
       _lfi = fmod ( numpulses , 100.0 ) ;
       PRE [ ((int) _lfi ) ] = _args[4] ;
       tspike [ ((int) _lfi ) ] = t ;
       numpulses = numpulses + 1.0 ;
       _args[7] = t ;
       }
     net_send ( _tqitem, _args, _pnt, t +  Cdur , _args[2] ) ;
     }
   if ( _lflag  == _args[2] ) {
     _args[3] = t ;
     T = 0.0 ;
     _args[1] = 0.0 ;
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 _args[4] = 0.0 ;
   _args[5] = 0.0 ;
   _args[6] = u0 ;
   _args[7] = t ;
   _args[2] = 1.0 ;
   }
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<3;_i++) _p[_dlist1[_i]] = 0.0;}
 r1 = r1FIX * pow( Trelease , 2.0 ) / pow( ( Trelease + kB ) , 2.0 ) ;
 r6 = r6FIX * pow( Trelease , 2.0 ) / pow( ( Trelease + kB ) , 2.0 ) ;
 /* ~ C <-> O ( r1 * Q10 , r2 * Q10 )*/
 f_flux =  r1 * Q10 * C ;
 b_flux =  r2 * Q10 * O ;
 DC -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ D <-> C ( r5 * Q10 , r6 * Q10 )*/
 f_flux =  r5 * Q10 * D ;
 b_flux =  r6 * Q10 * C ;
 DD -= (f_flux - b_flux);
 DC += (f_flux - b_flux);
 
 /*REACTION*/
   /* C + O + D = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<3;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 r1 = r1FIX * pow( Trelease , 2.0 ) / pow( ( Trelease + kB ) , 2.0 ) ;
 r6 = r6FIX * pow( Trelease , 2.0 ) / pow( ( Trelease + kB ) , 2.0 ) ;
 /* ~ C <-> O ( r1 * Q10 , r2 * Q10 )*/
 _term =  r1 * Q10 ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 2 ,1)  -= _term;
 _term =  r2 * Q10 ;
 _MATELM1( 1 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ D <-> C ( r5 * Q10 , r6 * Q10 )*/
 _term =  r5 * Q10 ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 1 ,0)  -= _term;
 _term =  r6 * Q10 ;
 _MATELM1( 0 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
   /* C + O + D = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
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
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 3, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
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
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  C = C0;
  D = D0;
  O = O0;
 {
   C = 1.0 ;
   O = 0.0 ;
   D = 0.0 ;
   T = 0.0 ;
   yuno = 0.0 ;
   numpulses = 0.0 ;
   Trelease = 0.0 ;
   tspike [ 0 ] = 1e12 ;
   gbar_Q10 = pow( Q10_diff , ( ( celsius - 30.0 ) / 10.0 ) ) ;
   Q10 = pow( Q10_channel , ( ( celsius - 30.0 ) / 10.0 ) ) ;
   Mres = 1e3 * ( 1e3 * 1e15 / 6.022e23 * M ) ;
   numpulses = 0.0 ;
   {int  _li ;for ( _li = 1 ; _li <= 100 ; _li ++ ) {
     PRE [ _li - 1 ] = 0.0 ;
     tspike [ _li - 1 ] = 0.0 ;
     } }
   tspike [ 0 ] = 1e12 ;
   if ( tau_1 >= tau_rec ) {
     printf ( "Warning: tau_1 (%g) should never be higher neither equal to tau_rec (%g)!\n" , tau_1 , tau_rec ) ;
     tau_rec = tau_1 + 1e-5 ;
     }
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
 _tsav = -1e20;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   if ( diffuse  && ( t > tspike [ 0 ] ) ) {
     Trelease = T + diffusione ( _threadargs_ ) ;
     }
   else {
     Trelease = T ;
     }
   g = gmax * gbar_Q10 * O ;
   i = ( 1e-6 ) * g * ( v - Erev ) * gmax_factor ;
   ic = i ;
   }
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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
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
 {  sparse_thread(&_thread[_spth1]._pvoid, 3, _slist1, _dlist1, _p, &t, dt, kstates, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 3; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 }}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = D_columnindex;  _dlist1[0] = DD_columnindex;
 _slist1[1] = C_columnindex;  _dlist1[1] = DC_columnindex;
 _slist1[2] = O_columnindex;  _dlist1[2] = DO_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/GOLGI_Ampa_mossy_det_vi.mod";
static const char* nmodl_file_text = 
  "TITLE \n"
  "\n"
  "COMMENT\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS Golgi_MF_syn\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	RANGE Q10_diff,Q10_channel\n"
  "	RANGE R, g, ic,yuno\n"
  "	RANGE Cdur, Erev \n"
  "	RANGE r1FIX, r2, r3,r4,r5,gmax,r1,r6,r6FIX,kB\n"
  "	RANGE tau_1, tau_rec, tau_facil, U	 \n"
  "	RANGE PRE,T,Tmax\n"
  "		\n"
  "	RANGE diffuse,Trelease,lamd\n"
  "	RANGE M,Diff,Rd,O\n"
  "	\n"
  "	RANGE tspike\n"
  "	RANGE nd, syntype, gmax_factor\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)	\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "	(pS) = (picosiemens)\n"
  "	(nS) = (nanosiemens)\n"
  "	(um) = (micrometer)\n"
  "	PI	= (pi)		(1)\n"
  "    }\n"
  "    \n"
  "    PARAMETER {\n"
  "	syntype\n"
  "	gmax_factor = 1\n"
  "	Q10_diff	= 1.5\n"
  "	Q10_channel	= 2.4\n"
  "	: Parametri Postsinaptici\n"
  "	gmax		= 2000   (pS)	\n"
  "\n"
  "	U 			= 0.2 (1) 	< 0, 1 >\n"
  "	tau_rec 	= 8 (ms) 	< 1e-9, 1e9 > 	 \n"
  "	tau_facil 	= 5 (ms) 	< 0, 1e9 > 	\n"
  "\n"
  "	M			= 21.515	: numero di (kilo) molecole in una vescicola		\n"
  "	Rd			= 1.03 (um)\n"
  "	Diff		= 0.223 (um2/ms)\n"
  "		 \n"
  "	Cdur		= 0.3	(ms)		 \n"
  "	r1FIX		= 5.4		(/ms/mM) 	 				\n"
  "	r2			= 0.82	(/ms)			 \n"
  "	r3			= 0		(/ms)		 \n"
  "	r4			= 0		(/ms)		 \n"
  "	r5			= 0.013	(/ms)			 \n"
  "	r6FIX		= 1.12	(/ms/mM)		 \n"
  "	Erev		= 0	(mV)\n"
  "	kB			= 0.44	(mM)\n"
  "\n"
  "	: Parametri Presinaptici\n"
  "	tau_1 		= 1 (ms) 	< 1e-9, 1e9 >\n"
  "	 \n"
  "\n"
  "	\n"
  "	u0 			= 0 (1) 	< 0, 1 >	: se u0=0 al primo colpo y=U\n"
  "	Tmax		= 1  (mM)	\n"
  "	\n"
  "	: Diffusion			\n"
  "	diffuse		= 1\n"
  "	\n"
  "	lamd		= 20 (nm)\n"
  "	nd			= 1\n"
  "	\n"
  "	celsius (degC)\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)		: postsynaptic voltage\n"
  "	i 		(nA)		: current = g*(v - Erev)\n"
  "	ic 		(nA)		: current = g*(v - Erev)\n"
  "	g 		(pS)		: conductance\n"
  "	r1		(/ms)\n"
  "	r6		(/ms)\n"
  "	T		(mM)\n"
  "\n"
  "	Trelease	(mM)\n"
  "	tspike[100]	(ms)\n"
  "	x \n"
  "	tsyn		(ms)\n"
  "	PRE[100]\n"
  "	\n"
  "	Mres		(mM)	\n"
  "	numpulses\n"
  "	tzero\n"
  "	gbar_Q10 (mho/cm2)\n"
  "	Q10 (1)\n"
  "	yuno\n"
  "}\n"
  "\n"
  "STATE {	\n"
  "	C\n"
  "	O\n"
  "	D\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	C=1\n"
  "	O=0\n"
  "	D=0\n"
  "	T=0 (mM)\n"
  "	yuno = 0\n"
  "	numpulses=0\n"
  "	Trelease=0 (mM)\n"
  "	tspike[0]=1e12	(ms)\n"
  "	 \n"
  "	gbar_Q10 = Q10_diff^((celsius-30)/10)\n"
  "	Q10 = Q10_channel^((celsius-30)/10)\n"
  "\n"
  "	: fattore di conversione che comprende molecole -> mM\n"
  "	: n molecole/(Na*V) -> M/(6.022e23*1dm^3)\n"
  "\n"
  "	Mres = 1e3 * ( 1e3 * 1e15 / 6.022e23 * M ) : (M) to (mM) so 1e3, 1um^3=1dm^3*1e-15 so 1e15	\n"
  "	numpulses=0\n"
  "\n"
  "	FROM i=1 TO 100 { PRE[i-1]=0 tspike[i-1]=0 }  \n"
  "	tspike[0]=1e12	(ms)\n"
  "	if(tau_1>=tau_rec){ \n"
  "		printf(\"Warning: tau_1 (%g) should never be higher neither equal to tau_rec (%g)!\\n\",tau_1,tau_rec)\n"
  "		tau_rec=tau_1+1e-5\n"
  "	    } \n"
  "	}\n"
  "		\n"
  "	FUNCTION imax(a,b) {\n"
  "	    if (a>b) { imax=a }\n"
  "	    else { imax=b }\n"
  "	}\n"
  "	\n"
  "FUNCTION diffusione(){	 \n"
  "	LOCAL DifWave,i,cntc,fi,aaa\n"
  "	DifWave=0\n"
  "	cntc=imax(numpulses-100,0)\n"
  "	FROM i=cntc  TO numpulses{\n"
  "	    :printf (\"%g %g  \",numpulses,fmod(numpulses,10))\n"
  "	    fi=fmod(i,100)\n"
  "	    :printf (\"%g %g %g __ \",i,numpulses,fi)\n"
  "	    tzero=tspike[fi]\n"
  "	    if(t>tzero){\n"
  "		aaa = (-Rd*Rd/(4*Diff*(t-tzero)))\n"
  "		if(fabs(aaa)<699){\n"
  "		    DifWave=DifWave+PRE[fi]*Mres*exp(aaa)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1\n"
  "		}else{\n"
  "		    if(aaa>0){\n"
  "			DifWave=DifWave+PRE[fi]*Mres*exp(699)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1\n"
  "		    }else{\n"
  "			DifWave=DifWave+PRE[fi]*Mres*exp(-699)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1\n"
  "		    }\n"
  "		}\n"
  "	    }\n"
  "	}	\n"
  "	:printf(\"\\n\\n\")\n"
  "	diffusione=DifWave\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "\n"
  "	if ( diffuse && (t>tspike[0]) ) { Trelease= T + diffusione() } else { Trelease=T }\n"
  "\n"
  "	SOLVE kstates METHOD sparse\n"
  "	g = gmax * gbar_Q10 * O\n"
  "	i = (1e-6) * g * (v - Erev) * gmax_factor\n"
  "	ic = i\n"
  "}\n"
  "\n"
  "KINETIC kstates {\n"
  "	r1 = r1FIX * Trelease^2 / (Trelease + kB)^2\n"
  "	r6 = r6FIX * Trelease^2 / (Trelease + kB)^2\n"
  "	~ C  <-> O	(r1*Q10,r2*Q10)\n"
  "	:~ O  <-> D	(r3*Q10,r4*Q10)\n"
  "	~ D  <-> C	(r5*Q10,r6*Q10)\n"
  "	CONSERVE C+O+D = 1\n"
  "}\n"
  "\n"
  "\n"
  "NET_RECEIVE(weight, on, nspike, tzero (ms),y, z, u, tsyn (ms)) {LOCAL fi\n"
  "\n"
  ": *********** ATTENZIONE! ***********\n"
  ":\n"
  ": Qualora si vogliano utilizzare impulsi di glutammato saturanti e'\n"
  ": necessario che il pulse sia piu' corto dell'intera simulazione\n"
  ": altrimenti la variabile on non torna al suo valore di default.\n"
  "\n"
  "INITIAL {\n"
  "	y = 0\n"
  "	z = 0\n"
  "	u = u0\n"
  "	tsyn = t\n"
  "	nspike = 1\n"
  "}\n"
  "   if (flag == 0) { \n"
  "		: Qui faccio rientrare la modulazione presinaptica\n"
  "		nspike = nspike + 1\n"
  "		if (!on) {\n"
  "			tzero = t\n"
  "			on = 1				\n"
  "			z = z*exp( - (t - tsyn) / (tau_rec) )	 \n"
  "			z = z + ( y*(exp(-(t - tsyn)/tau_1) - exp(-(t - tsyn)/(tau_rec)))/((tau_1/(tau_rec))-1) )  \n"
  "			y = y*exp(-(t - tsyn)/tau_1)			\n"
  "			x = 1-y-z\n"
  "				\n"
  "			if (tau_facil > 0) { \n"
  "				u = u*exp(-(t - tsyn)/tau_facil)\n"
  "				u = u + U * ( 1 - u )							\n"
  "			} else { u = U }\n"
  "			\n"
  "			y = y + x * u\n"
  "			\n"
  "			T=Tmax*y\n"
  "			yuno = y\n"
  "			fi=fmod(numpulses,100)\n"
  "			PRE[fi]=y	: PRE[numpulses]=y\n"
  "			\n"
  "			:PRE=1	: Istruzione non necessaria ma se ommesso allora le copie dell'oggetto successive alla prima non funzionano!\n"
  "			:}\n"
  "			: all'inizio numpulses=0 !			\n"
  "			\n"
  "			tspike[fi] = t\n"
  "			numpulses=numpulses+1\n"
  "			tsyn = t\n"
  "			\n"
  "		}\n"
  "		net_send(Cdur, nspike)	 \n"
  "    }\n"
  "    if (flag == nspike) { \n"
  "		tzero = t\n"
  "		T = 0\n"
  "		on = 0\n"
  "    }\n"
  "}\n"
  "\n"
  ;
#endif
