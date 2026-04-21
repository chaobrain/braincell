TITLE Intracellular calcium concentration in deep cerebellar nucleus (DCN) neuron
COMMENT
    This mechanism keeps track of intracellular calcium entering the cell through
    the CaHVA channel. The calcium concentration is for a hypothetical shell
    below the membrane of the cell, and affects the conductance level of the SK.mod
    channel mechanism. In addition, it affects the conductance of the CaHVA channel
    which uses the GHK equation to calculate current flow.
    Translated from GENESIS by Johannes Luthman and Volker Steuber.
ENDCOMMENT

NEURON {
    SUFFIX CdpHVA_SU15_DCN
    USEION ca READ ica WRITE cai
    RANGE cai, kCa, depth
    GLOBAL tauCa
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
    : tauCa when it's inserted (in hoc).

    : kCa in the following is given for the soma and shall be adjusted
    : from hoc when the CaConc model is inserted in a dendrite: 1.04e-6
    : in the dendritic compartments units. kCa is here given as 1 / coulombs
    : instead of moles / coulomb as in the GENESIS code since mole
    : in NEURON simply equals the number 6.022 * 10e23
    kCa = 3.45e-7 (1/coulomb)
    tauCa = 70 (ms)
    caiBase = 50e-6 (mM) : the assumed resting level of intracellular calcium
    depth = 0.2 (micron)
}

ASSIGNED {
    C (kilo / m3 / s)
    D (kilo / m3 / s)
    ica (mA/cm2)
}

STATE {
    cai (mM)
}

INITIAL {
    cai = caiBase
}

BREAKPOINT {
    SOLVE states METHOD cnexp
}

DERIVATIVE states {
    C = (cai - caiBase) / tauCa
	D = - kCa / depth * ica * (1e4)
    cai' = D - C
}
