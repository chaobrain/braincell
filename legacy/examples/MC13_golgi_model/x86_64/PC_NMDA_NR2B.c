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
 
#define nrn_init _nrn_init__PC_NMDA_NR2B
#define _nrn_initial _nrn_initial__PC_NMDA_NR2B
#define nrn_cur _nrn_cur__PC_NMDA_NR2B
#define _nrn_current _nrn_current__PC_NMDA_NR2B
#define nrn_jacob _nrn_jacob__PC_NMDA_NR2B
#define nrn_state _nrn_state__PC_NMDA_NR2B
#define _net_receive _net_receive__PC_NMDA_NR2B 
#define _f_rates _f_rates__PC_NMDA_NR2B 
#define kstates kstates__PC_NMDA_NR2B 
#define rates rates__PC_NMDA_NR2B 
 
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
#define gmax _p[1]
#define gmax_columnindex 1
#define Q10_diff _p[2]
#define Q10_diff_columnindex 2
#define Q10_channel _p[3]
#define Q10_channel_columnindex 3
#define U _p[4]
#define U_columnindex 4
#define tau_rec _p[5]
#define tau_rec_columnindex 5
#define tau_facil _p[6]
#define tau_facil_columnindex 6
#define M _p[7]
#define M_columnindex 7
#define Rd _p[8]
#define Rd_columnindex 8
#define Diff _p[9]
#define Diff_columnindex 9
#define tau_1 _p[10]
#define tau_1_columnindex 10
#define u0 _p[11]
#define u0_columnindex 11
#define Tmax _p[12]
#define Tmax_columnindex 12
#define Cdur _p[13]
#define Cdur_columnindex 13
#define Erev _p[14]
#define Erev_columnindex 14
#define v0_block _p[15]
#define v0_block_columnindex 15
#define k_block _p[16]
#define k_block_columnindex 16
#define nd _p[17]
#define nd_columnindex 17
#define diffuse _p[18]
#define diffuse_columnindex 18
#define lamd _p[19]
#define lamd_columnindex 19
#define i _p[20]
#define i_columnindex 20
#define ic _p[21]
#define ic_columnindex 21
#define ica _p[22]
#define ica_columnindex 22
#define g _p[23]
#define g_columnindex 23
#define rb1 _p[24]
#define rb1_columnindex 24
#define rb2 _p[25]
#define rb2_columnindex 25
#define T _p[26]
#define T_columnindex 26
#define Trelease _p[27]
#define Trelease_columnindex 27
#define PRE (_p + 28)
#define PRE_columnindex 28
#define MgBlock _p[128]
#define MgBlock_columnindex 128
#define C0 _p[129]
#define C0_columnindex 129
#define C1 _p[130]
#define C1_columnindex 130
#define C2 _p[131]
#define C2_columnindex 131
#define C3 _p[132]
#define C3_columnindex 132
#define C4 _p[133]
#define C4_columnindex 133
#define D1 _p[134]
#define D1_columnindex 134
#define D2 _p[135]
#define D2_columnindex 135
#define O _p[136]
#define O_columnindex 136
#define eca _p[137]
#define eca_columnindex 137
#define x _p[138]
#define x_columnindex 138
#define tspike (_p + 139)
#define tspike_columnindex 139
#define Mres _p[239]
#define Mres_columnindex 239
#define numpulses _p[240]
#define numpulses_columnindex 240
#define tzero _p[241]
#define tzero_columnindex 241
#define gbar_Q10 _p[242]
#define gbar_Q10_columnindex 242
#define Q10 _p[243]
#define Q10_columnindex 243
#define DC0 _p[244]
#define DC0_columnindex 244
#define DC1 _p[245]
#define DC1_columnindex 245
#define DC2 _p[246]
#define DC2_columnindex 246
#define DC3 _p[247]
#define DC3_columnindex 247
#define DC4 _p[248]
#define DC4_columnindex 248
#define DD1 _p[249]
#define DD1_columnindex 249
#define DD2 _p[250]
#define DD2_columnindex 250
#define DO _p[251]
#define DO_columnindex 251
#define v _p[252]
#define v_columnindex 252
#define _g _p[253]
#define _g_columnindex 253
#define _tsav _p[254]
#define _tsav_columnindex 254
#define _nd_area  *_ppvar[0]._pval
#define _ion_eca	*_ppvar[2]._pval
#define _ion_ica	*_ppvar[3]._pval
#define _ion_dicadv	*_ppvar[4]._pval
 
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
 static double _hoc_rates(void*);
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
 "rates", _hoc_rates,
 0, 0
};
#define diffusione diffusione_PC_NMDA_NR2B
#define imax imax_PC_NMDA_NR2B
 extern double diffusione( _threadargsproto_ );
 extern double imax( _threadargsprotocomma_ double , double );
 
static void _check_rates(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_rates(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define C4_O_on C4_O_on_PC_NMDA_NR2B
 double C4_O_on = 8.553;
#define C3_O_on C3_O_on_PC_NMDA_NR2B
 double C3_O_on = 0.145;
#define C4_C2_off C4_C2_off_PC_NMDA_NR2B
 double C4_C2_off = 0.694;
#define C2_C4_on C2_C4_on_PC_NMDA_NR2B
 double C2_C4_on = 0.145;
#define C3_C2_off C3_C2_off_PC_NMDA_NR2B
 double C3_C2_off = 0.528;
#define C2_C3_on C2_C3_on_PC_NMDA_NR2B
 double C2_C3_on = 8.553;
#define C2_D2_on C2_D2_on_PC_NMDA_NR2B
 double C2_D2_on = 0.338;
#define C2_D1_on C2_D1_on_PC_NMDA_NR2B
 double C2_D1_on = 1.659;
#define C2_C1_off C2_C1_off_PC_NMDA_NR2B
 double C2_C1_off = 0.23;
#define C1_C2_on C1_C2_on_PC_NMDA_NR2B
 double C1_C2_on = 4.53;
#define C1_C0_off C1_C0_off_PC_NMDA_NR2B
 double C1_C0_off = 0.115;
#define C0_C1_on C0_C1_on_PC_NMDA_NR2B
 double C0_C1_on = 9.06;
#define D2_C2_off D2_C2_off_PC_NMDA_NR2B
 double D2_C2_off = 0.00274;
#define D1_C2_off D1_C2_off_PC_NMDA_NR2B
 double D1_C2_off = 0.245;
#define O_C4_off O_C4_off_PC_NMDA_NR2B
 double O_C4_off = 0.528;
#define O_C3_off O_C3_off_PC_NMDA_NR2B
 double O_C3_off = 0.694;
#define kB kB_PC_NMDA_NR2B
 double kB = 0.44;
#define usetable usetable_PC_NMDA_NR2B
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "U", 0, 1,
 "tau_facil", 0, 1e+09,
 "tau_rec", 1e-09, 1e+09,
 "tau_1", 1e-09, 1e+09,
 "usetable_PC_NMDA_NR2B", 0, 1,
 "u0", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "C0_C1_on_PC_NMDA_NR2B", "/mM",
 "C1_C0_off_PC_NMDA_NR2B", "/ms",
 "C1_C2_on_PC_NMDA_NR2B", "/mM",
 "C2_C1_off_PC_NMDA_NR2B", "/ms",
 "C2_D1_on_PC_NMDA_NR2B", "/ms",
 "D1_C2_off_PC_NMDA_NR2B", "/ms",
 "C2_D2_on_PC_NMDA_NR2B", "/ms",
 "D2_C2_off_PC_NMDA_NR2B", "/ms",
 "C2_C3_on_PC_NMDA_NR2B", "/ms",
 "C3_C2_off_PC_NMDA_NR2B", "/ms",
 "C2_C4_on_PC_NMDA_NR2B", "/ms",
 "C4_C2_off_PC_NMDA_NR2B", "/ms",
 "C3_O_on_PC_NMDA_NR2B", "/ms",
 "O_C3_off_PC_NMDA_NR2B", "/ms",
 "C4_O_on_PC_NMDA_NR2B", "/ms",
 "O_C4_off_PC_NMDA_NR2B", "/ms",
 "kB_PC_NMDA_NR2B", "mM",
 "gmax", "pS",
 "U", "1",
 "tau_rec", "ms",
 "tau_facil", "ms",
 "Rd", "um",
 "Diff", "um2/ms",
 "tau_1", "ms",
 "u0", "1",
 "Tmax", "mM",
 "Cdur", "ms",
 "Erev", "mV",
 "v0_block", "mV",
 "k_block", "mV",
 "lamd", "nm",
 "i", "nA",
 "ic", "nA",
 "ica", "nA",
 "g", "pS",
 "rb1", "/ms",
 "rb2", "/ms",
 "T", "mM",
 "Trelease", "mM",
 0,0
};
 static double C40 = 0;
 static double C30 = 0;
 static double C20 = 0;
 static double C10 = 0;
 static double C00 = 0;
 static double D20 = 0;
 static double D10 = 0;
 static double O0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "C0_C1_on_PC_NMDA_NR2B", &C0_C1_on_PC_NMDA_NR2B,
 "C1_C0_off_PC_NMDA_NR2B", &C1_C0_off_PC_NMDA_NR2B,
 "C1_C2_on_PC_NMDA_NR2B", &C1_C2_on_PC_NMDA_NR2B,
 "C2_C1_off_PC_NMDA_NR2B", &C2_C1_off_PC_NMDA_NR2B,
 "C2_D1_on_PC_NMDA_NR2B", &C2_D1_on_PC_NMDA_NR2B,
 "D1_C2_off_PC_NMDA_NR2B", &D1_C2_off_PC_NMDA_NR2B,
 "C2_D2_on_PC_NMDA_NR2B", &C2_D2_on_PC_NMDA_NR2B,
 "D2_C2_off_PC_NMDA_NR2B", &D2_C2_off_PC_NMDA_NR2B,
 "C2_C3_on_PC_NMDA_NR2B", &C2_C3_on_PC_NMDA_NR2B,
 "C3_C2_off_PC_NMDA_NR2B", &C3_C2_off_PC_NMDA_NR2B,
 "C2_C4_on_PC_NMDA_NR2B", &C2_C4_on_PC_NMDA_NR2B,
 "C4_C2_off_PC_NMDA_NR2B", &C4_C2_off_PC_NMDA_NR2B,
 "C3_O_on_PC_NMDA_NR2B", &C3_O_on_PC_NMDA_NR2B,
 "O_C3_off_PC_NMDA_NR2B", &O_C3_off_PC_NMDA_NR2B,
 "C4_O_on_PC_NMDA_NR2B", &C4_O_on_PC_NMDA_NR2B,
 "O_C4_off_PC_NMDA_NR2B", &O_C4_off_PC_NMDA_NR2B,
 "kB_PC_NMDA_NR2B", &kB_PC_NMDA_NR2B,
 "usetable_PC_NMDA_NR2B", &usetable_PC_NMDA_NR2B,
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
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"PC_NMDA_NR2B",
 "syntype",
 "gmax",
 "Q10_diff",
 "Q10_channel",
 "U",
 "tau_rec",
 "tau_facil",
 "M",
 "Rd",
 "Diff",
 "tau_1",
 "u0",
 "Tmax",
 "Cdur",
 "Erev",
 "v0_block",
 "k_block",
 "nd",
 "diffuse",
 "lamd",
 0,
 "i",
 "ic",
 "ica",
 "g",
 "rb1",
 "rb2",
 "T",
 "Trelease",
 "PRE[100]",
 "MgBlock",
 0,
 "C0",
 "C1",
 "C2",
 "C3",
 "C4",
 "D1",
 "D2",
 "O",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 255, _prop);
 	/*initialize range parameters*/
 	syntype = 0;
 	gmax = 5000;
 	Q10_diff = 1.4;
 	Q10_channel = 2.2;
 	U = 0.2;
 	tau_rec = 8;
 	tau_facil = 5;
 	M = 21.515;
 	Rd = 1.03;
 	Diff = 0.223;
 	tau_1 = 1;
 	u0 = 0;
 	Tmax = 1;
 	Cdur = 0.3;
 	Erev = -3.7;
 	v0_block = -20;
 	k_block = 13;
 	nd = 1;
 	diffuse = 1;
 	lamd = 20;
  }
 	_prop->param = _p;
 	_prop->param_size = 255;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[2]._pval = &prop_ion->param[0]; /* eca */
 	_ppvar[3]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[4]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 
#define _tqitem &(_ppvar[5]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _PC_NMDA_NR2B_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 3,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
  _extcall_thread = (Datum*)ecalloc(2, sizeof(Datum));
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 255, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "netsend");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 8;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 PC_NMDA_NR2B /home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/PC_NMDA_NR2B.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define PI _nrnunit_PI[_nrnunit_use_legacy_]
static double _nrnunit_PI[2] = {0x1.921fb54442d18p+1, 3.14159}; /* 3.14159265358979312 */
 static double *_t_MgBlock;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(_threadargsprotocomma_ double);
static int rates(_threadargsprotocomma_ double);
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_rates(_threadargsprotocomma_ double _lv);
 static int _slist1[8], _dlist1[8]; static double *_temp1;
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
for(_i=1;_i<8;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 Trelease = diffusione ( _threadargs_ ) ;
   rb1 = C0_C1_on * Trelease ;
   rb2 = C1_C2_on * Trelease ;
   /* ~ C0 <-> C1 ( rb1 * Q10 , C1_C0_off * Q10 )*/
 f_flux =  rb1 * Q10 * C0 ;
 b_flux =  C1_C0_off * Q10 * C1 ;
 _RHS1( 5) -= (f_flux - b_flux);
 _RHS1( 4) += (f_flux - b_flux);
 
 _term =  rb1 * Q10 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 4 ,5)  -= _term;
 _term =  C1_C0_off * Q10 ;
 _MATELM1( 5 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ C1 <-> C2 ( rb2 * Q10 , C2_C1_off * Q10 )*/
 f_flux =  rb2 * Q10 * C1 ;
 b_flux =  C2_C1_off * Q10 * C2 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  rb2 * Q10 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  C2_C1_off * Q10 ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C2 <-> D1 ( C2_D1_on * Q10 , D1_C2_off * Q10 )*/
 f_flux =  C2_D1_on * Q10 * C2 ;
 b_flux =  D1_C2_off * Q10 * D1 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  C2_D1_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 7 ,3)  -= _term;
 _term =  D1_C2_off * Q10 ;
 _MATELM1( 3 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ C2 <-> D2 ( C2_D2_on * Q10 , D2_C2_off * Q10 )*/
 f_flux =  C2_D2_on * Q10 * C2 ;
 b_flux =  D2_C2_off * Q10 * D2 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 6) += (f_flux - b_flux);
 
 _term =  C2_D2_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 6 ,3)  -= _term;
 _term =  D2_C2_off * Q10 ;
 _MATELM1( 3 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C3 ( C2_C3_on * Q10 , C3_C2_off * Q10 )*/
 f_flux =  C2_C3_on * Q10 * C2 ;
 b_flux =  C3_C2_off * Q10 * C3 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  C2_C3_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  C3_C2_off * Q10 ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C4 ( C2_C4_on * Q10 , C4_C2_off * Q10 )*/
 f_flux =  C2_C4_on * Q10 * C2 ;
 b_flux =  C4_C2_off * Q10 * C4 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  C2_C4_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 1 ,3)  -= _term;
 _term =  C4_C2_off * Q10 ;
 _MATELM1( 3 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ C3 <-> O ( C3_O_on * Q10 , O_C3_off * Q10 )*/
 f_flux =  C3_O_on * Q10 * C3 ;
 b_flux =  O_C3_off * Q10 * O ;
 _RHS1( 2) -= (f_flux - b_flux);
 
 _term =  C3_O_on * Q10 ;
 _MATELM1( 2 ,2)  += _term;
 _term =  O_C3_off * Q10 ;
 _MATELM1( 2 ,0)  -= _term;
 /*REACTION*/
  /* ~ C4 <-> O ( C4_O_on * Q10 , O_C4_off * Q10 )*/
 f_flux =  C4_O_on * Q10 * C4 ;
 b_flux =  O_C4_off * Q10 * O ;
 _RHS1( 1) -= (f_flux - b_flux);
 
 _term =  C4_O_on * Q10 ;
 _MATELM1( 1 ,1)  += _term;
 _term =  O_C4_off * Q10 ;
 _MATELM1( 1 ,0)  -= _term;
 /*REACTION*/
   /* C0 + C1 + C2 + C3 + C4 + D1 + D2 + O = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= O ;
 _MATELM1(0, 6) = 1;
 _RHS1(0) -= D2 ;
 _MATELM1(0, 7) = 1;
 _RHS1(0) -= D1 ;
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
 static double _mfac_rates, _tmin_rates;
  static void _check_rates(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_v0_block;
  static double _sav_k_block;
  if (!usetable) {return;}
  if (_sav_v0_block != v0_block) { _maktable = 1;}
  if (_sav_k_block != k_block) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  - 120.0 ;
   _tmax =  30.0 ;
   _dx = (_tmax - _tmin_rates)/150.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 151; _x += _dx, _i++) {
    _f_rates(_p, _ppvar, _thread, _nt, _x);
    _t_MgBlock[_i] = MgBlock;
   }
   _sav_v0_block = v0_block;
   _sav_k_block = k_block;
  }
 }

 static int rates(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_rates(_p, _ppvar, _thread, _nt);
#endif
 _n_rates(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_rates(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_rates * (_lv - _tmin_rates);
 if (isnan(_xi)) {
  MgBlock = _xi;
  return;
 }
 if (_xi <= 0.) {
 MgBlock = _t_MgBlock[0];
 return; }
 if (_xi >= 150.) {
 MgBlock = _t_MgBlock[150];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 MgBlock = _t_MgBlock[_i] + _theta*(_t_MgBlock[_i+1] - _t_MgBlock[_i]);
 }

 
static int  _f_rates ( _threadargsprotocomma_ double _lv ) {
   MgBlock = 1.0 / ( 1.0 + exp ( - ( _lv - v0_block ) / k_block ) ) ;
    return 0; }
 
static double _hoc_rates(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 
#if 1
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
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
 {int _i; for(_i=0;_i<8;_i++) _p[_dlist1[_i]] = 0.0;}
 Trelease = diffusione ( _threadargs_ ) ;
 rb1 = C0_C1_on * Trelease ;
 rb2 = C1_C2_on * Trelease ;
 /* ~ C0 <-> C1 ( rb1 * Q10 , C1_C0_off * Q10 )*/
 f_flux =  rb1 * Q10 * C0 ;
 b_flux =  C1_C0_off * Q10 * C1 ;
 DC0 -= (f_flux - b_flux);
 DC1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C1 <-> C2 ( rb2 * Q10 , C2_C1_off * Q10 )*/
 f_flux =  rb2 * Q10 * C1 ;
 b_flux =  C2_C1_off * Q10 * C2 ;
 DC1 -= (f_flux - b_flux);
 DC2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> D1 ( C2_D1_on * Q10 , D1_C2_off * Q10 )*/
 f_flux =  C2_D1_on * Q10 * C2 ;
 b_flux =  D1_C2_off * Q10 * D1 ;
 DC2 -= (f_flux - b_flux);
 DD1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> D2 ( C2_D2_on * Q10 , D2_C2_off * Q10 )*/
 f_flux =  C2_D2_on * Q10 * C2 ;
 b_flux =  D2_C2_off * Q10 * D2 ;
 DC2 -= (f_flux - b_flux);
 DD2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> C3 ( C2_C3_on * Q10 , C3_C2_off * Q10 )*/
 f_flux =  C2_C3_on * Q10 * C2 ;
 b_flux =  C3_C2_off * Q10 * C3 ;
 DC2 -= (f_flux - b_flux);
 DC3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> C4 ( C2_C4_on * Q10 , C4_C2_off * Q10 )*/
 f_flux =  C2_C4_on * Q10 * C2 ;
 b_flux =  C4_C2_off * Q10 * C4 ;
 DC2 -= (f_flux - b_flux);
 DC4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C3 <-> O ( C3_O_on * Q10 , O_C3_off * Q10 )*/
 f_flux =  C3_O_on * Q10 * C3 ;
 b_flux =  O_C3_off * Q10 * O ;
 DC3 -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C4 <-> O ( C4_O_on * Q10 , O_C4_off * Q10 )*/
 f_flux =  C4_O_on * Q10 * C4 ;
 b_flux =  O_C4_off * Q10 * O ;
 DC4 -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
   /* C0 + C1 + C2 + C3 + C4 + D1 + D2 + O = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<8;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 Trelease = diffusione ( _threadargs_ ) ;
 rb1 = C0_C1_on * Trelease ;
 rb2 = C1_C2_on * Trelease ;
 /* ~ C0 <-> C1 ( rb1 * Q10 , C1_C0_off * Q10 )*/
 _term =  rb1 * Q10 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 4 ,5)  -= _term;
 _term =  C1_C0_off * Q10 ;
 _MATELM1( 5 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ C1 <-> C2 ( rb2 * Q10 , C2_C1_off * Q10 )*/
 _term =  rb2 * Q10 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  C2_C1_off * Q10 ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C2 <-> D1 ( C2_D1_on * Q10 , D1_C2_off * Q10 )*/
 _term =  C2_D1_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 7 ,3)  -= _term;
 _term =  D1_C2_off * Q10 ;
 _MATELM1( 3 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ C2 <-> D2 ( C2_D2_on * Q10 , D2_C2_off * Q10 )*/
 _term =  C2_D2_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 6 ,3)  -= _term;
 _term =  D2_C2_off * Q10 ;
 _MATELM1( 3 ,6)  -= _term;
 _MATELM1( 6 ,6)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C3 ( C2_C3_on * Q10 , C3_C2_off * Q10 )*/
 _term =  C2_C3_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  C3_C2_off * Q10 ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C4 ( C2_C4_on * Q10 , C4_C2_off * Q10 )*/
 _term =  C2_C4_on * Q10 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 1 ,3)  -= _term;
 _term =  C4_C2_off * Q10 ;
 _MATELM1( 3 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ C3 <-> O ( C3_O_on * Q10 , O_C3_off * Q10 )*/
 _term =  C3_O_on * Q10 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 0 ,2)  -= _term;
 _term =  O_C3_off * Q10 ;
 _MATELM1( 2 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ C4 <-> O ( C4_O_on * Q10 , O_C4_off * Q10 )*/
 _term =  C4_O_on * Q10 ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 0 ,1)  -= _term;
 _term =  O_C4_off * Q10 ;
 _MATELM1( 1 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
   /* C0 + C1 + C2 + C3 + C4 + D1 + D2 + O = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 8;}
 
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
	for (_i=0; _i < 8; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 8, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
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
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 4, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  C4 = C40;
  C3 = C30;
  C2 = C20;
  C1 = C10;
  C0 = C00;
  D2 = D20;
  D1 = D10;
  O = O0;
 {
   rates ( _threadargscomma_ v ) ;
   C0 = 1.0 ;
   C1 = 0.0 ;
   C2 = 0.0 ;
   C3 = 0.0 ;
   C4 = 0.0 ;
   D1 = 0.0 ;
   D2 = 0.0 ;
   O = 0.0 ;
   T = 0.0 ;
   numpulses = 0.0 ;
   gbar_Q10 = pow( Q10_diff , ( ( celsius - 30.0 ) / 10.0 ) ) ;
   Q10 = pow( Q10_channel , ( ( celsius - 30.0 ) / 10.0 ) ) ;
   Mres = 1e3 * ( 1e3 * 1e15 / 6.022e23 * M ) ;
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

#if 0
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
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
  eca = _ion_eca;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   rates ( _threadargscomma_ v ) ;
   g = gmax * gbar_Q10 * O ;
   i = ( 1e-6 ) * g * ( v - Erev ) * MgBlock ;
   ica = ( ( 1e-6 ) * g * ( v - Erev ) * MgBlock ) / 10.0 ;
   ic = i + ica ;
   }
 _current += i;
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
  _ion_dicadv += (_dica - ica)/.001 * 1.e2/ (_nd_area);
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica * 1.e2/ (_nd_area);
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
  eca = _ion_eca;
 {  sparse_thread(&_thread[_spth1]._pvoid, 8, _slist1, _dlist1, _p, &t, dt, kstates, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 8; ++_i) {
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
   _t_MgBlock = makevector(151*sizeof(double));
 _slist1[0] = O_columnindex;  _dlist1[0] = DO_columnindex;
 _slist1[1] = C4_columnindex;  _dlist1[1] = DC4_columnindex;
 _slist1[2] = C3_columnindex;  _dlist1[2] = DC3_columnindex;
 _slist1[3] = C2_columnindex;  _dlist1[3] = DC2_columnindex;
 _slist1[4] = C1_columnindex;  _dlist1[4] = DC1_columnindex;
 _slist1[5] = C0_columnindex;  _dlist1[5] = DC0_columnindex;
 _slist1[6] = D2_columnindex;  _dlist1[6] = DD2_columnindex;
 _slist1[7] = D1_columnindex;  _dlist1[7] = DD1_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/swl/braincell/examples/MC13_golgi_model/golgi_NEURON/mod_gol/PC_NMDA_NR2B.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "\n"
  "NMDA channel, NR2B subunit and  calcium current\n"
  "Modification made by Stefano Masoli PhD based on Nius 2006 and Santucci 2008\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS PC_NMDA_NR2B\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	USEION ca READ eca WRITE ica\n"
  "	\n"
  "	RANGE Q10_diff,Q10_channel\n"
  "	RANGE g , ic, ica\n"
  "	RANGE Cdur,Erev,T,Tmax\n"
  "	RANGE Rb, Ru, Rd, Rr, Ro, Rc,rb1,rb2,gmax,RdRate\n"
  "	RANGE tau_1, tau_rec, tau_facil, U, u0 \n"
  "	RANGE PRE\n"
  "	RANGE Used\n"
  "	RANGE MgBlock,v0_block,k_block\n"
  "	RANGE diffuse,Trelease,lamd, Diff, M, Rd, nd, syntype, y_scale\n"
  "	RANGE C0,C1,C2,C3,C4,D1,D2,O\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)	\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "	(uM) = (micro/liter)\n"
  "	(pS) = (picosiemens)\n"
  "	(nS) = (nanosiemens)\n"
  "	\n"
  "	PI	= (pi)		(1)\n"
  "    }\n"
  "    \n"
  "    PARAMETER {\n"
  "	syntype\n"
  "	: Parametri Presinaptici\n"
  "	gmax		= 5000  	(pS)	: 7e3 : 4e4\n"
  "	Q10_diff	= 1.4\n"
  "	Q10_channel	= 2.2\n"
  "	U 		= 0.2 (1) 	< 0, 1 >\n"
  "	tau_rec 	= 8 (ms) 	< 1e-9, 1e9 > 	 \n"
  "	tau_facil 	= 5 (ms) 	< 0, 1e9 > 	\n"
  "\n"
  "	M	= 21.515	: numero di (kilo) molecole in una vescicola		\n"
  "	Rd	= 1.03 (um)\n"
  "	Diff	= 0.223 (um2/ms)\n"
  "	tau_1 	= 1 (ms) 	< 1e-9, 1e9 >\n"
  "\n"
  "	u0 		= 0 (1) < 0, 1 >\n"
  "	Tmax		= 1  	(mM)\n"
  "\n"
  "	: Postsinaptico, Santucci 2008 scheme\n"
  "	\n"
  "	Cdur	= 0.3	(ms)\n"
  "	\n"
  "	:binding and unbinding\n"
  "	C0_C1_on = 9.06 (/mM /ms)\n"
  "	C1_C0_off = 0.115 (/ms)\n"
  "	C1_C2_on = 4.53 (/mM /ms)\n"
  "	C2_C1_off = 0.23 (/ms)\n"
  "	\n"
  "	:desensitization\n"
  "	C2_D1_on = 1.659 (/ms)\n"
  "	D1_C2_off = 0.245 (/ms)\n"
  "	C2_D2_on = 0.338 (/ms)\n"
  "	D2_C2_off = 0.00274 (/ms)\n"
  "	\n"
  "	:middle closed\n"
  "	C2_C3_on = 8.553 (/ms)\n"
  "	C3_C2_off = 0.528 (/ms)\n"
  "	C2_C4_on = 0.145 (/ms)\n"
  "	C4_C2_off = 0.694 (/ms)\n"
  "	\n"
  "	:open\n"
  "	C3_O_on = 0.145 (/ms)\n"
  "	O_C3_off = 0.694 (/ms)\n"
  "	C4_O_on = 8.553 (/ms)\n"
  "	O_C4_off = 0.528 (/ms)	\n"
  "	\n"
  "	\n"
  "	Erev	= -3.7  (mV)	: 0 (mV)\n"
  "	\n"
  "	v0_block = -20 (mV)	: -16 -8.69 (mV)	: -18.69 (mV) : -32.7 (mV)\n"
  "	k_block  = 13 (mV)\n"
  "	nd	 = 1\n"
  "	kB	 = 0.44	(mM)\n"
  "\n"
  "	: Diffusion			\n"
  "	diffuse	= 1\n"
  "	lamd	= 20 (nm)\n"
  "	celsius (degC)\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)		: postsynaptic voltage\n"
  "	i 		(nA)		: current = g*(v - Erev)\n"
  "	ic 		(nA)		: current = g*(v - Erev)\n"
  "	ica 		(nA)\n"
  "	g 		(pS)		: actual conductance\n"
  "	eca 		(mV)\n"
  "\n"
  "	rb1		(/ms)    : binding\n"
  "	rb2		(/ms)    : binding\n"
  "	\n"
  "	T		(mM)\n"
  "	x \n"
  "	\n"
  "	Trelease	(mM)\n"
  "	tspike[100]	(ms)	: will be initialized by the pointprocess\n"
  "	PRE[100]\n"
  "	Mres		(mM)	\n"
  "	\n"
  "	MgBlock\n"
  "	numpulses\n"
  "	tzero\n"
  "	gbar_Q10 (mho/cm2)\n"
  "	Q10 (1)\n"
  "	:nr2bi (mM)\n"
  "	:y_scale\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	: Channel states (all fractions)\n"
  "	C0		: single bound\n"
  "	C1		: double bound\n"
  "	C2		: closed 2\n"
  "	C3		: closed 3\n"
  "	C4		: closed 4\n"
  "	D1		: desensitized one\n"
  "	D2              : desensitized two\n"
  "	O		: open\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	C0 = 1\n"
  "	C1 = 0\n"
  "	C2 = 0\n"
  "	C3 = 0\n"
  "	C4 = 0\n"
  "	D1 = 0\n"
  "	D2 = 0\n"
  "	O  = 0\n"
  "	T  = 0\n"
  "	numpulses=0\n"
  "\n"
  "	gbar_Q10 = Q10_diff^((celsius-30)/10)\n"
  "	Q10 = Q10_channel^((celsius-30)/10)\n"
  "\n"
  "	Mres = 1e3 * (1e3 * 1e15 / 6.022e23 * M)     : (M) to (mM) so 1e3, 1um^3=1dm^3*1e-15 so 1e15\n"
  "	FROM i=1 TO 100 { PRE[i-1]=0 tspike[i-1]=0 } :PRE_2[500]=0}\n"
  "	tspike[0]=1e12	(ms)\n"
  "	if(tau_1>=tau_rec){ \n"
  "		printf(\"Warning: tau_1 (%g) should never be higher neither equal to tau_rec (%g)!\\n\",tau_1,tau_rec)\n"
  "		tau_rec=tau_1+1e-5\n"
  "		:printf(\"tau_rec has been set to %g\\n\",tau_rec) \n"
  "	} \n"
  "\n"
  "}\n"
  "	FUNCTION imax(a,b) {\n"
  "	    if (a>b) { imax=a }\n"
  "	    else { imax=b }\n"
  "	}\n"
  "	\n"
  "\n"
  "FUNCTION diffusione(){	 \n"
  "	LOCAL DifWave,i,cntc,fi,aaa\n"
  "	DifWave=0\n"
  "	cntc=imax(numpulses-100,0)\n"
  "	FROM i=cntc  TO numpulses{\n"
  "	    fi=fmod(i,100)\n"
  "		tzero=tspike[fi]\n"
  "		if(t>tzero){\n"
  "		    aaa = (-Rd*Rd/(4*Diff*(t-tzero)))\n"
  "		    if(fabs(aaa)<699){\n"
  "			DifWave=DifWave+PRE[fi]*Mres*exp(aaa)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1\n"
  "		    }else{\n"
  "			if(aaa>0){\n"
  "			    DifWave=DifWave+PRE[fi]*Mres*exp(699)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1\n"
  "			}else{\n"
  "			    DifWave=DifWave+PRE[fi]*Mres*exp(-699)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1\n"
  "			}\n"
  "		    }\n"
  "		}\n"
  "	}	\n"
  "	diffusione=DifWave\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	rates(v)\n"
  "	SOLVE kstates METHOD sparse	\n"
  "	\n"
  "	g = gmax * gbar_Q10 * O\n"
  "	\n"
  "	: E' piu' logico spostare * MgBlock * PRE sul calcolo della corrente!\n"
  "	i = (1e-6) * g * (v - Erev) * MgBlock\n"
  "	ica = ((1e-6) * g * (v - Erev) * MgBlock)/10\n"
  "	ic = i + ica\n"
  "    }\n"
  "\n"
  "KINETIC kstates {	\n"
  "	:if ( diffuse && (t>tspike[0]) ) { Trelease= T + diffusione() } else { Trelease=T }\n"
  "	Trelease = diffusione()\n"
  "	rb1 = C0_C1_on * Trelease	\n"
  "	rb2 = C1_C2_on * Trelease	\n"
  "	~ C0 <-> C1	(rb1*Q10,C1_C0_off*Q10) 	\n"
  "	~ C1 <-> C2	(rb2*Q10,C2_C1_off*Q10)	\n"
  "	~ C2 <-> D1	(C2_D1_on*Q10,D1_C2_off*Q10)\n"
  "	~ C2 <-> D2	(C2_D2_on*Q10,D2_C2_off*Q10)\n"
  "	~ C2 <-> C3	(C2_C3_on*Q10,C3_C2_off*Q10)\n"
  "	~ C2 <-> C4	(C2_C4_on*Q10,C4_C2_off*Q10)\n"
  "	~ C3 <-> O	(C3_O_on*Q10,O_C3_off*Q10)\n"
  "	~ C4 <-> O	(C4_O_on*Q10,O_C4_off*Q10)\n"
  "	CONSERVE C0+C1+C2+C3+C4+D1+D2+O = 1\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {\n"
  "	: E' necessario includere DEPEND v0_block,k_block per aggiornare le tabelle!\n"
  "	TABLE MgBlock DEPEND v0_block,k_block FROM -120 TO 30 WITH 150\n"
  "	MgBlock = 1 / ( 1 + exp ( - ( v - v0_block ) / k_block ) )\n"
  "}\n"
  "\n"
  "\n"
  "NET_RECEIVE(weight, on, nspike, tzero (ms),y, z, u, tsyn (ms)) {LOCAL fi\n"
  "\n"
  ": *********** ATTENZIONE! ***********\n"
  ":\n"
  ": Qualora si vogliano utilizzare impulsi di glutammato saturanti e' \n"
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
  "			z = z*exp( - (t - tsyn) / (tau_rec) )	: RESCALED !\n"
  "			z = z + ( y*(exp(-(t - tsyn)/tau_1) - exp(-(t - tsyn)/(tau_rec)))/((tau_1/(tau_rec))-1) ) : RESCALED !\n"
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
  "	if (flag == nspike) { \n"
  "			tzero = t\n"
  "			T = 0\n"
  "			on = 0\n"
  "	}\n"
  "}\n"
  "\n"
  ;
#endif
