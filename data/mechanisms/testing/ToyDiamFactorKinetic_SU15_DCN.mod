TITLE Minimal diameter-driven factor reaction toy in deep cerebellar nucleus (DCN)
COMMENT
    This toy validates geometry-derived factors in the BrainCell import path
    while still supporting ordinary reaction-network alignment against NEURON.

    The comparison notebook sets:

        pump_area = PI * diam
        cyto = PI * diam * depth

    and uses one reversible binding step:

        cai + pumpfree <-> pumpbound

    together with the conserved pool:

        pumpfree + pumpbound = PumpTot * pump_area
ENDCOMMENT

NEURON {
    SUFFIX ToyDiamFactorKinetic_SU15_DCN
    USEION ca WRITE cai VALENCE 2
    RANGE cai, pumpfree, pumpbound, kf, kb, PumpTot, pump_area, cyto
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
    (um) = (micron)
    PI = (pi) (1)
}

PARAMETER {
    kf = 2 (/ms mM)
    kb = 0.5 (/ms)
    PumpTot = 1 (mM um)
    pump_area = 62.83185307179586 (um)
    cyto = 62.83185307179586 (um2)
}

STATE {
    cai (mM)
    pumpfree (mM um)
    pumpbound (mM um)
}

INITIAL {
    cai = 0.2
    pumpbound = 0
    pumpfree = PumpTot - pumpbound
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    COMPARTMENT cyto {cai}
    COMPARTMENT pump_area {pumpfree pumpbound}
    ~ cai + pumpfree <-> pumpbound (kf * pump_area, kb * pump_area)
    CONSERVE pumpfree + pumpbound = PumpTot * pump_area
}
