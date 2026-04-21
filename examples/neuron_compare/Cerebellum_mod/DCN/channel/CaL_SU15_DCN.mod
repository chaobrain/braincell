TITLE LVA calcium current (CaLVA) of deep cerebellar nucleus (DCN) neuron

COMMENT
    Translated from GENESIS by Johannes Luthman and Volker Steuber.
ENDCOMMENT

NEURON {
	SUFFIX CaL_SU15_DCN
	USEION call READ calli, callo WRITE icall VALENCE 2
	RANGE gbar, m, h, icall
	GLOBAL qdeltat
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
    qdeltat = 1
    gbar = 1e-5 (siemens/cm2)
    calli (mM)
    callo (mM)  
   	carev (mV)

}

ASSIGNED {
	v (mV)
	icall (mA/cm2)
	minf
	hinf
	taum (ms)
	tauh (ms)
}

STATE {
	m
	h
}

INITIAL {
    rate(v)
    m = minf
	h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    carev = 139 : (1e3) * (R*(celsius+273.15))/(2*FARADAY) * log (cao/cai)

	icall = gbar * m*m * h * (v - carev)
}

DERIVATIVE states {
	rate(v)
	m' =(minf - m)/taum
	h' =(hinf - h)/tauh
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