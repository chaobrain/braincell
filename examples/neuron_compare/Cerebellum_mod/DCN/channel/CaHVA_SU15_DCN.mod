TITLE High voltage activated calcium current (CaHVA) of deep cerebellar nucleus (DCN) neuron
COMMENT
    Translated from GENESIS by Johannes Luthman and Volker Steuber. 
    This mechanism and the other calcium channel (CaLVA.mod) are the only channel
    mechanisms of the DCN model that use the GHK mechanism to calculate reversal
    potential. Thus, extracellular Ca concentration is of importance and shall be 
    set from hoc.

    Calcium entering the neuron through this channel is kept track of by CaConc.mod
    and affects the conductance of the SK.mod channel mechanism.
ENDCOMMENT 

NEURON { 
	SUFFIX CaHVA_SU15_DCN 
	USEION ca READ cai, cao WRITE ica
	RANGE perm, ica, m, cai
	GLOBAL qdeltat
} 
 
UNITS { 
	(mA) = (milliamp) 
	(mV) = (millivolt) 
	(molar) = (1/liter)
	(mM) = (millimolar)
} 
 
PARAMETER { 
    qdeltat = 1
    perm = 7.5e-6 (cm/seconds)
} 

ASSIGNED {
	v (mV)
    cai (mM)
    cao (mM)     
	ica (mA/cm2) 
	minf (1)
	taum (ms) 
	celsius (degC)
	T (kelvin)
    A (1)
} 

STATE {
	m
} 

INITIAL { 
    T = 273.15 + celsius
    rate(v)
    m = minf 
} 
 
BREAKPOINT { 
    SOLVE states METHOD cnexp 
    A = getGHKexp(v)
    : "4.47814e6 * v / T" in the following is the simplification of the GHK
    : current equation's (z^2 * F^2 * (0.001) * v) / (R * T). [*(0.001) is to get 
    : volt from NEURON's mV.] Together with the simplification in getGHKexp() 
    : (below), this speeds up the whole DCN simulation (without synapses) by 8%.
    : The division of the calcium concentrations (mM) by 1000 gives molar as 
    : required by the GHK current equation.
    ica = perm * m*m*m * (4.47814e6 * v / T) * ((cai/1000) - (cao/1000) * A) / (1 - A)
} 
 
DERIVATIVE states { 
	rate(v) 
	m' = (minf - m)/taum 
} 

PROCEDURE rate(v(mV)) {
    minf = 1 / (1 + exp((v + 34.5) / -9))
    taum = 1 / ((31.746 * ((exp((v - 5) / -13.89) + 1) ^ -1)) + (3.97e-4 * (v + 8.9)) * ((exp((v + 8.9) / 5) - 1) ^ -1))
    taum = taum / qdeltat
} 

FUNCTION getGHKexp(v(mV)) {
    getGHKexp = exp(-23.20764929 * v / T): =the calculated values of
            : getGHKexp = exp((-z * F * (0.001) * v) / (R * T)).
}
