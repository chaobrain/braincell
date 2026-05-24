TITLE Minimal factor-crossing calcium pump toy in deep cerebellar nucleus (DCN)
COMMENT
    This toy validates BrainCell KineticIon factor handling with two explicit
    compartments:

      - cai in a cytosolic volume
      - pump states on a membrane-area-like pool

    Dynamics:

        cai + pumpfree <-> pumpbound
        pumpbound -> pumpfree

    Conservation:

        pumpfree + pumpbound = PumpTot * pump_area

    Current-driven calcium source:

        cai' += -(kCa / depth) * ica * 1e4
ENDCOMMENT

NEURON {
    SUFFIX ToyCaPumpFactorKinetic_SU15_DCN
    USEION ca READ ica WRITE cai VALENCE 2
    RANGE cai, pumpfree, pumpbound, kf, kb, k_rel, PumpTot, kCa, depth
    RANGE cyt_volume, pump_area
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
    (mA) = (milliamp)
    (um) = (micron)
}

PARAMETER {
    kf = 2 (/ms mM)
    kb = 0.5 (/ms)
    k_rel = 0.05 (/ms)
    PumpTot = 1 (mM um)
    kCa = 3.45e-7 (1/coulomb)
    depth = 0.2 (micron)
    cyt_volume = 3 (um3)
    pump_area = 3 (um2)
}

ASSIGNED {
    ica (mA/cm2)
}

STATE {
    cai (mM)
    pumpfree (mM um)
    pumpbound (mM um)
}

INITIAL {
    cai = 0.1
    pumpbound = 0
    pumpfree = PumpTot
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    COMPARTMENT cyt_volume {cai}
    COMPARTMENT pump_area {pumpfree pumpbound}

    ~ cai + pumpfree <-> pumpbound (kf * pump_area, kb * pump_area)
    ~ pumpbound <-> pumpfree (k_rel * pump_area, 0)
    ~ cai << (-(kCa / depth) * ica * 1e4 * cyt_volume)
    CONSERVE pumpfree + pumpbound = PumpTot * pump_area
}
