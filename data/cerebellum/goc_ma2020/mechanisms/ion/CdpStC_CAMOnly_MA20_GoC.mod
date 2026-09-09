TITLE Calcium accumulation with calmodulin-only subnetwork in Golgi cell model

NEURON {
  SUFFIX CdpStC_CAMOnly_MA20_GoC
  USEION ca READ cao, ica WRITE cai VALENCE 2
  RANGE Nannuli, vrat
  RANGE CAM0, CAM1C, CAM2C, CAM1N2C, CAM1N, CAM2N, CAM2N1C, CAM1C1N, CAM4
}

UNITS {
    (molar) = (1/liter)
    (mM)    = (millimolar)
    (um)    = (micron)
    (mA)    = (milliamp)
    PI      = (pi) (1)
}

PARAMETER {
    Nannuli = 10.9495 (1)
    celsius (degC)

    cainull = 45e-6 (mM)

    CAM_start = 0.03 (mM)

    K1Coff = 0.04 (/ms)
    K1Con = 5.4 (/ms mM)
    K2Coff = 0.00925 (/ms)
    K2Con = 15.0 (/ms mM)
    K1Noff = 2.5 (/ms)
    K1Non = 142.5 (/ms mM)
    K2Noff = 0.75 (/ms)
    K2Non = 175.0 (/ms mM)
}

ASSIGNED {
    diam   (um)
    ica    (mA/cm2)
    cai    (mM)
    cao    (mM)
    vrat   (1)
    dsq    (um2)
    dsqvol (um2)
}

STATE {
    ca       (mM)
    CAM0     (mM)
    CAM1C    (mM)
    CAM2C    (mM)
    CAM1N2C  (mM)
    CAM1N    (mM)
    CAM2N    (mM)
    CAM2N1C  (mM)
    CAM1C1N  (mM)
    CAM4     (mM)
}

BREAKPOINT {
    dsq = diam*diam
    dsqvol = dsq*vrat
    SOLVE state METHOD sparse
}

INITIAL {
    factors()

    ca = cainull

    CAM0 = CAM_start
    CAM1C = 0
    CAM2C = 0
    CAM1N2C = 0
    CAM1N = 0
    CAM2N = 0
    CAM2N1C = 0
    CAM1C1N = 0
    CAM4 = 0

    cai = ca
}

PROCEDURE factors() {
    LOCAL r, dr2
    r = 1/2
    dr2 = r/(Nannuli-1)/2
    vrat = PI*(r-dr2/2)*2*dr2
    r = r - dr2
}

KINETIC state {
    COMPARTMENT diam*diam*vrat {ca}

    dsq = diam*diam
    dsqvol = dsq*vrat

    : Calmodulin
    : C-lobe
    ~ ca + CAM0 <-> CAM1C (K1Con*dsqvol, K1Coff*dsqvol)
    ~ ca + CAM1C <-> CAM2C (K2Con*dsqvol, K2Coff*dsqvol)
    ~ ca + CAM2C <-> CAM1N2C (K1Non*dsqvol, K1Noff*dsqvol)
    ~ ca + CAM1N2C <-> CAM4 (K2Non*dsqvol, K2Noff*dsqvol)

    : N-lobe
    ~ ca + CAM0 <-> CAM1N (K1Non*dsqvol, K1Noff*dsqvol)
    ~ ca + CAM1N <-> CAM2N (K2Non*dsqvol, K2Noff*dsqvol)
    ~ ca + CAM2N <-> CAM2N1C (K1Con*dsqvol, K1Coff*dsqvol)
    ~ ca + CAM2N1C <-> CAM4 (K2Con*dsqvol, K2Coff*dsqvol)

    : Mixed C and N lobes
    ~ ca + CAM1C <-> CAM1C1N (K1Non*dsqvol, K1Noff*dsqvol)
    ~ ca + CAM1N <-> CAM1C1N (K1Con*dsqvol, K1Coff*dsqvol)
    ~ ca + CAM1C1N <-> CAM1N2C (K2Con*dsqvol, K2Coff*dsqvol)
    ~ ca + CAM1C1N <-> CAM2N1C (K2Non*dsqvol, K2Noff*dsqvol)

    cai = ca
}
