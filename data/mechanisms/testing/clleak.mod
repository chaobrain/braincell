TITLE Passive chloride leak for BrainCell's Cl_Leak comparison
COMMENT
    icl = g * (v - ecl), outward positive as in NEURON. The mechanism only
    reads ecl, so NEURON does not compute it from cli/clo unless the caller
    sets ion_style("cl_ion", 1, 2, 1, 1, 1); otherwise ecl stays 0 mV.
ENDCOMMENT

NEURON {
    SUFFIX clleak
    USEION cl READ ecl WRITE icl VALENCE -1
    RANGE g
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    g = 0.0001 (S/cm2)
}

ASSIGNED {
    v   (mV)
    ecl (mV)
    icl (mA/cm2)
}

BREAKPOINT {
    icl = g * (v - ecl)
}
