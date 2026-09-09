TITLE Fast sodium current (NaF) of deep cerebellar nucleus (DCN) neuron

COMMENT
    Translated from GENESIS by Johannes Luthman and Volker Steuber.
ENDCOMMENT

NEURON {
	SUFFIX NaF_SU15_DCN
	USEION na READ ena WRITE ina
	RANGE gbar, m, h, ina
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
	minf = 1 / (1 + exp((v + 45) / -7.3))
	taum = 5.83 / (exp((v - (6.4)) / -9) + exp((v + 97) / 17)) + 0.025
    taum = taum / qdeltat
    hinf = 1 / (1 + exp((v + 42) / 5.9))
	tauh = 16.67 / (exp((v - 8.3) / -29) + exp((v + 66) / 9)) + 0.2
    tauh = tauh / qdeltat
}
