TITLE Passive potassium leak for BrainCell's dynamic potassium comparisons
COMMENT
    ik = g * (v - ek), outward positive as in NEURON. A pool that WRITEs ki
    and ko (K_conc.mod) makes NEURON compute ek from them every step.
ENDCOMMENT

NEURON {
    SUFFIX kleak
    USEION k READ ek WRITE ik
    RANGE g
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    g = 0.001 (S/cm2)
}

ASSIGNED {
    v  (mV)
    ek (mV)
    ik (mA/cm2)
}

BREAKPOINT {
    ik = g * (v - ek)
}
