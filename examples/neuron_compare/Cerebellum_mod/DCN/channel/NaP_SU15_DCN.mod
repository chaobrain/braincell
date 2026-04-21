TITLE Persistent sodium current (NaP) of deep cerebellar nucleus (DCN) neuron
COMMENT
    Translated from GENESIS by Johannes Luthman and Volker Steuber.
ENDCOMMENT

NEURON {
	SUFFIX NaP_SU15_DCN
	USEION na READ ena WRITE ina
	RANGE gbar, ina, m, h
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
    ena (mV)
	ina (mA/cm2)
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
	taum = 50 / qdeltat
    m = minf
	h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	ina = gbar * m*m*m * h * (v - ena)
}

DERIVATIVE states {
	rate(v)
	m' =(minf - m)/taum
	h' =(hinf - h)/tauh
}

PROCEDURE rate(v(mV)) {
	TABLE minf, hinf, tauh FROM -150 TO 100 WITH 300
	minf = 1 / (1 + exp((v + 70) / -4.1))
    hinf = 1 / (1 + exp((v + 80) / 4))
	tauh = (1750 / (1 + exp((v + 65) / -8))) + 250
    tauh = tauh / qdeltat
}
