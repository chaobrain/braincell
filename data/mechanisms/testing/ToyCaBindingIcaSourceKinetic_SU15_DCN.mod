TITLE Minimal reversible calcium-binding kinetic toy with ica-driven source in deep cerebellar nucleus (DCN)
COMMENT
    This toy extends the source-free reversible binding toy with a calcium
    current driven source term.

    It models:

        cai + b <-> bc

    with conserved buffer pool:

        b + bc = Btot

    and a current-driven calcium influx:

        cai' += -(kCa / depth) * ica * 1e4
ENDCOMMENT

NEURON {
    SUFFIX ToyCaBindingIcaSourceKinetic_SU15_DCN
    USEION ca READ ica WRITE cai VALENCE 2
    RANGE cai, b, bc, kf, kb, Btot, kCa, depth
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
    (mA) = (milliamp)
}

PARAMETER {
    kf = 2 (/ms mM)
    kb = 0.5 (/ms)
    Btot = 1 (mM)
    kCa = 3.45e-7 (1/coulomb)
    depth = 0.2 (micron)
}

ASSIGNED {
    ica (mA/cm2)
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
    ~ cai << (-(kCa / depth) * ica * 1e4)
    CONSERVE b + bc = Btot
}
