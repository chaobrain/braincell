TITLE h current of deep cerebellar nucleus (DCN) neuron
COMMENT
    Translated from GENESIS by Johannes Luthman and Volker Steuber.
ENDCOMMENT 

NEURON { 
	SUFFIX HCN_SU15_DCN
	NONSPECIFIC_CURRENT ih
	RANGE gbar, ih, eh, m
	GLOBAL qdeltat
} 
 
UNITS { 
	(mA) = (milliamp) 
	(mV) = (millivolt) 
} 
 
PARAMETER { 
    qdeltat = 1
    gbar = 1e-5 (siemens/cm2)
} 

ASSIGNED {
	v (mV)
	ih (mA/cm2) 
	eh (mV)
	minf
	taum (ms)
} 
 
STATE {
	m
} 

INITIAL { 
    rate(v) 
	taum = 400 / qdeltat
    m = minf 
} 
 
BREAKPOINT { 
    SOLVE states METHOD cnexp 
	ih = gbar * m * m * (v - eh)
} 
 
DERIVATIVE states { 
	rate(v) 
	m' = (minf - m) / taum 
} 

PROCEDURE rate(v(mV)) {
	minf = 1 / (1 + exp((v + 80) / 5))
}
