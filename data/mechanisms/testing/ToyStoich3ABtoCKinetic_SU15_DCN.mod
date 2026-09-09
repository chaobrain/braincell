TITLE Minimal stoichiometric kinetic toy in deep cerebellar nucleus (DCN)
COMMENT
    This toy exists to validate higher-order stoichiometry handling in the
    BrainCell KineticIon path against a minimal NEURON KINETIC example.

    It models:

        3a + b <-> c
ENDCOMMENT

NEURON {
    SUFFIX ToyStoich3ABtoCKinetic_SU15_DCN
    USEION ca WRITE cai VALENCE 2
    RANGE cai, a, b, c, kf, kb
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
}

PARAMETER {
    kf = 2 (/ms mM3)
    kb = 0.5 (/ms)
}

STATE {
    cai (mM)
    a (mM)
    b (mM)
    c (mM)
}

INITIAL {
    cai = 0.1
    a = 1.0
    b = 1.0
    c = 0.0
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    ~ cai <-> cai (0, 0)
    ~ a + a + a + b <-> c (kf, kb)
}
