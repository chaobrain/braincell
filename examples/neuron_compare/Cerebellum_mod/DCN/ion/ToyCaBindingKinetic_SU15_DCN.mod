TITLE Minimal reversible calcium-binding kinetic toy in deep cerebellar nucleus (DCN)
COMMENT
    This deliberately minimal mechanism exists only to validate the BrainCell
    KineticIon import path against a small NMODL KINETIC example before
    attempting larger DCN or GoC calcium-pool mechanisms.

    It models one reversible buffering step:

        cai + b <-> bc

    with a conserved buffer pool:

        b + bc = Btot
ENDCOMMENT

NEURON {
    SUFFIX ToyCaBindingKinetic_SU15_DCN
    USEION ca WRITE cai VALENCE 2
    RANGE cai, b, bc, kf, kb, Btot
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
}

PARAMETER {
    kf = 2 (/ms mM)
    kb = 0.5 (/ms)
    Btot = 1 (mM)
}

STATE {
    cai (mM)
    b (mM)
    bc (mM)
}

INITIAL {
    cai = 0.1
    bc = 0
    b = Btot - bc
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    ~ cai + b <-> bc (kf, kb)
    CONSERVE b + bc = Btot
}
