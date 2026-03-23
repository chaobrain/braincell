TITLE Simplified Kv example for BrainCell template generation

NEURON {
    SUFFIX Kv
    USEION k READ ek WRITE ik
    RANGE gbar, i, v12, q
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.0   (S/cm2)
    Ra   = 0.02  (/mV/ms)
    Rb   = 0.006 (/mV/ms)
    v12  = 25    (mV)
    q    = 9     (mV)
}

ASSIGNED {
    v    (mV)
    i    (mA/cm2)
    ik   (mA/cm2)
    ek   (mV)
    ninf
    ntau (ms)
    tadj
}

STATE { n }

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar * n * (v - ek)
    ik = i
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n) / ntau
}

INITIAL {
    rates(v)
    n = ninf
}

FUNCTION rateconst(v (mV), r (/mV/ms), th (mV), q (mV)) (/ms) {
    rateconst = r * (v - th) / (1 - exp(-(v - th)/q))
}

PROCEDURE rates(v (mV)) {
    tadj = 1
    ninf = 1 / (1 + exp(-(v - v12) / q))
    ntau = 1 / (tadj * (rateconst(v, Ra, v12, q) + rateconst(v, -Rb, v12, -q)))
}
