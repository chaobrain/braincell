TITLE Low voltage activated calcium current (CaLVA) of deep cerebellar nucleus (DCN) neuron
COMMENT
    This mechanism and the other calcium channel (CaHVA.mod) are the only channel
    mechanisms of the DCN model that use the GHK mechanism to calculate reversal
    potential. Thus, extracellular Ca concentration is of importance and shall be
    set from hoc to 2mM, using: "calo0_ca_ion = 2".

    The calcium that this channel lets through feeds into the CalConc.mod mechanism
    while calcium entry via the CaHVA channel is tracked by CalConc.mod.
ENDCOMMENT 

NEURON { 
	SUFFIX CaLVA_SU15_DCN 
	USEION cal READ cali, calo WRITE ical VALENCE 2
	RANGE perm, ical, m, h, cali
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
    perm = 1 (cm/seconds)
} 

ASSIGNED {
    v (mV)
    cali (mM)
    calo (mM)     
	ical (mA/cm2) 
	minf
	hinf
	taum (ms) 
	tauh (ms) 
	celsius (degC)
	T (kelvin)
    A (1)
} 
 
STATE {
	m
    h
} 

INITIAL { 
    T = 273.15 + celsius
    rate(v)
    m = minf 
	h = hinf
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
    ical = perm * m*m * h * (4.47814e6 * v / T) * ((cali/1000) - (calo/1000) * A) / (1 - A)
} 
 
DERIVATIVE states { 
	rate(v) 
	m' = (minf - m)/taum 
	h' = (hinf - h)/tauh 
} 

PROCEDURE rate(v(mV)) {
	TABLE minf, taum, hinf, tauh  FROM -150 TO 100 WITH 300 
	minf = 1 / (1 + exp((v + 56) / -6.2))
	taum = 0.333 / (exp((v + 131) / -16.7) + exp((v + 15.8) / 18.2)) + 0.204
    taum = taum / qdeltat
	hinf = 1 / (1 + exp((v + 80) / 4))
    if (v < -81) {
        tauh = 0.333 * exp((v + 466) / 66)
    } else {
        tauh = 0.333 * exp((v + 21) / -10.5) + 9.32
    }
    tauh = tauh / qdeltat
}

FUNCTION getGHKexp(v(mV)) {
    TABLE DEPEND T FROM -150 TO 100 WITH 300 
    getGHKexp = exp(-23.20764929 * v / T): =the calculated values of
            : getGHKexp = exp((-z * F * (0.001) * v) / (R * T)).
}
