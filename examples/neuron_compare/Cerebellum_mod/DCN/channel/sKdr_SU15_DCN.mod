TITLE Slow delayed rectifier (sKdr) of deep cerebellar nucleus (DCN) neuron
COMMENT
    Translated from GENESIS by Johannes Luthman and Volker Steuber.
ENDCOMMENT

NEURON {
	SUFFIX sKdr_SU15_DCN
	USEION k READ ek WRITE ik
	RANGE gbar, m, ik
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
	ek (mV)
	ik (mA/cm2)
	minf
	taum (ms)
}

STATE {
	m
}

INITIAL {
    rate(v)
    m = minf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	ik = gbar * m*m*m*m * (v - ek)
}

DERIVATIVE states {
	rate(v)
	m' = (minf - m) / taum
}

PROCEDURE rate(v(mV)) {
	TABLE minf, taum FROM -150 TO 100 WITH 300
    minf = 1 / (1 + exp((v + 50) / -9.1))
    taum = 14.95 / (exp((v + 50) / 21.74) + exp((v + 50) / -13.91)) + 0.05
    taum = taum / qdeltat
}
