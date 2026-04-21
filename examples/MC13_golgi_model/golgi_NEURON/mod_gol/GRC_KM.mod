TITLE Cerebellum Granule Cell Model

COMMENT
        KM channel
   
	Author: A. Fontana
	CoAuthor: T.Nieus Last revised: 20.11.99
	
ENDCOMMENT
 
NEURON { 
	SUFFIX GRC_KM 
	USEION k READ ek WRITE ik 
	RANGE gkbar, ik, g, alpha_n, beta_n 
	RANGE Aalpha_n, Kalpha_n, V0alpha_n
	RANGE Abeta_n, Kbeta_n, V0beta_n
	RANGE V0_ninf, B_ninf
	RANGE n_inf, tau_n 
} 
 
UNITS { 
	(mA) = (milliamp) 
	(mV) = (millivolt) 
} 
 
PARAMETER { 
	Aalpha_n = 0.0033 (/ms)
	Kalpha_n = 40 (mV)

	V0alpha_n = -30 (mV)
	Abeta_n = 0.0033 (/ms)
	Kbeta_n = -20 (mV)

	V0beta_n = -30 (mV)
	V0_ninf = -35 (mV)	:-30
	B_ninf = 6 (mV)		:6:4 rimesso a 6 dopo calibrazione febbraio 2003	
	v (mV) 
	gkbar= 0.00025 (mho/cm2) :0.0001
	ek = -84.69 (mV) 
	celsius = 30 (degC) 
} 

STATE { 
	n 
} 

ASSIGNED { 
	ik (mA/cm2) 
	n_inf 
	tau_n (ms) 
	g (mho/cm2) 
	alpha_n (/ms) 
	beta_n (/ms) 
} 
 
INITIAL { 
	rate(v) 
	n = n_inf 
} 
 
BREAKPOINT { 
	SOLVE states METHOD derivimplicit 
	g = gkbar*n 
	ik = g*(v - ek) 
	alpha_n = alp_n(v) 
	beta_n = bet_n(v) 
} 
 
DERIVATIVE states { 
	rate(v) 
	n' =(n_inf - n)/tau_n 
} 
 
FUNCTION alp_n(v(mV))(/ms) { LOCAL Q10
	Q10 = 3^((celsius-22(degC))/10(degC)) 
	alp_n = Q10*Aalpha_n*exp((v-V0alpha_n)/Kalpha_n) 
} 
 
FUNCTION bet_n(v(mV))(/ms) { LOCAL Q10
	Q10 = 3^((celsius-22(degC))/10(degC)) 
	bet_n = Q10*Abeta_n*exp((v-V0beta_n)/Kbeta_n) 
} 
 
PROCEDURE rate(v (mV)) {LOCAL a_n, b_n 
	TABLE n_inf, tau_n 
	DEPEND Aalpha_n, Kalpha_n, V0alpha_n, 
	       Abeta_n, Kbeta_n, V0beta_n, V0_ninf, B_ninf, celsius FROM -100 TO 30 WITH 13000 
	a_n = alp_n(v)  
	b_n = bet_n(v) 
	tau_n = 1/(a_n + b_n) 
:	n_inf = a_n/(a_n + b_n) 
	n_inf = 1/(1+exp(-(v-V0_ninf)/B_ninf))
} 
