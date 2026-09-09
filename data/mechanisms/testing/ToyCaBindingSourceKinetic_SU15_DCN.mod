TITLE Minimal reversible calcium-binding kinetic toy with constant source in deep cerebellar nucleus (DCN)
COMMENT
    This deliberately minimal mechanism extends the source-free toy
    ToyCaBindingKinetic_SU15_DCN with one constant source term on cai.

    It models:

        cai + b <-> bc

    with conserved buffer pool:

        b + bc = Btot

    and a constant calcium influx:

        cai' += ci_source
ENDCOMMENT

NEURON {
    SUFFIX ToyCaBindingSourceKinetic_SU15_DCN
    USEION ca WRITE cai VALENCE 2
    RANGE cai, b, bc, kf, kb, Btot, ci_source
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
}

PARAMETER {
    kf = 2 (/ms mM)
    kb = 0.5 (/ms)
    Btot = 1 (mM)
    ci_source = 0.002 (mM/ms)
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
    ~ cai << (ci_source)
    CONSERVE b + bc = Btot
}
