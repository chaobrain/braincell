TITLE Simplified Hodgkin-Huxley example for AST and IR extraction

NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    RANGE gnabar, gkbar, gl, el
    GLOBAL minf, hinf, ninf, mtau, htau, ntau
    NONSPECIFIC_CURRENT il
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gnabar = 0.12 (S/cm2)
    gkbar = 0.036 (S/cm2)
    gl = 0.0003 (S/cm2)
    el = -54.3 (mV)
}

STATE {
    m
    h
    n
}

ASSIGNED {
    v (mV)
    ena (mV)
    ek (mV)
    ina (mA/cm2)
    ik (mA/cm2)
    il (mA/cm2)
    minf
    hinf
    ninf
    mtau (ms)
    htau (ms)
    ntau (ms)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
    n = ninf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h * (v - ena)
    ik = gkbar * n*n*n*n * (v - ek)
    il = gl * (v - el)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    n' = (ninf - n) / ntau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v + 40) / 10))
    hinf = 1 / (1 + exp((v + 62) / 10))
    ninf = 1 / (1 + exp(-(v + 53) / 16))
    mtau = 0.2
    htau = 1
    ntau = 1
}
