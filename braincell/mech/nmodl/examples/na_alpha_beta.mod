TITLE Simple alpha/beta sodium example for one_ion_hh_ohmic

NEURON {
    SUFFIX naab
    USEION na READ ena WRITE ina
    RANGE gnabar
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S)  = (siemens)
}

PARAMETER {
    gnabar = 0.12 (S/cm2)
}

STATE {
    m
    h
}

ASSIGNED {
    v (mV)
    ena (mV)
    ina (mA/cm2)
    alpha_m (/ms)
    beta_m (/ms)
    alpha_h (/ms)
    beta_h (/ms)
}

INITIAL {
    rates(v)
    m = alpha_m / (alpha_m + beta_m)
    h = alpha_h / (alpha_h + beta_h)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h * (v - ena)
}

DERIVATIVE states {
    rates(v)
    m' = alpha_m * (1 - m) - beta_m * m
    h' = alpha_h * (1 - h) - beta_h * h
}

PROCEDURE rates(v (mV)) {
    alpha_m = 0.1 * exp(-(v + 40) / 10)
    beta_m = 4 * exp(-(v + 65) / 18)
    alpha_h = 0.07 * exp(-(v + 65) / 20)
    beta_h = 1 / (1 + exp(-(v + 35) / 10))
}
