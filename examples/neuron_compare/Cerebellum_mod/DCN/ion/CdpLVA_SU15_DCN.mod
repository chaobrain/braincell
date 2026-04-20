TITLE Intracellular calcium concentration from the CaLVA channel in deep cerebellar nucleus (DCN) neuron
COMMENT
    This mechanism keeps track of intracellular calcium entering the cell through
    the CaLVA channel, with the sole purpose of setting the conductance of the 
    channel which uses the GHK equation to calculate current flow.
    
    The mechanism is a copy of the CaConc.mod and thus uses the same method of
    tracking Ca concentration merely in a hypothetical shell below the membrane
    of the cell.
ENDCOMMENT

NEURON {
    SUFFIX CdpLVA_SU15_DCN
    USEION cal READ ical WRITE cali VALENCE 2    
    RANGE cali, kCal, depth
    GLOBAL tauCal
}

UNITS {
    (molar) = (1 / liter)
    (mM) = (millimolar)
    (mV) = (millivolt)
    (mA) = (milliamp)
    PI = (pi) (1)
}

PARAMETER {
    : qdeltat has been skipped here and is used via the division of this NMODL's
    : tauCal when it's inserted (in hoc).

    : kCa in the following is given for the soma and shall be adjusted
    : from hoc when the CaConc model is inserted in a dendrite: 1.04e-6
    : in the dendritic compartments units. kCa is here given as 1 / coulombs
    : instead of moles / coulomb as in the GENESIS code since mole
    : in NEURON simply equals the number 6.022 * 10e23
    kCal = 3.45e-7 (1/coulomb)
    tauCal = 70 (ms)
    caliBase = 50e-6 (mM) : the resting intracellular calcium conc =50 nM
    depth = 0.2 (micron)
}

ASSIGNED {
    C (kilo / m3 / s)
    D (kilo / m3 / s)
    ical (mA/cm2)
}

STATE {
    cali (mM)
}

INITIAL {
    cali = caliBase
}

BREAKPOINT {
    SOLVE states METHOD cnexp
}

DERIVATIVE states {
    C = (cali - caliBase) / tauCal
	D = - kCal / depth * ical * (1e4)
    cali' = D - C
}
