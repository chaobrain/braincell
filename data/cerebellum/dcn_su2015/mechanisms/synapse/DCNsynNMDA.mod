COMMENT by Johannes Luthman:

Based on DCNsyn.mod. Changes made compared to that file: 
 # variables MgFactor, gamma added, as PARAMETERs.
 # magnesium-block (variable C below) added, giving fraction of mg-block
 # of the channel

The magnesium-block is based on how GENESIS does it in its built-in 
Mg_block function
(http: //www.genesis-sim.org/GENESIS/Hyperdoc/Manual-26.html#Mg_block),
explained in The book of GENESIS, section 19.5 "NMDA Channels".

ENDCOMMENT

NEURON {
	POINT_PROCESS DCNsynNMDA
	NONSPECIFIC_CURRENT i
	RANGE g, i, e, tauRise, tauFall, MgFactor, gamma, C
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(mM) = (milli/liter)
}

PARAMETER {
	tauRise = 1 (ms)
	tauFall = 1 (ms)
	e = 0 (mV)
    MgFactor = 0.1 (mM)
    gamma = 0.1
}

ASSIGNED {
	C 
	v (mV)
	i (nA)
	g (microsiemens)
	factor
}

STATE {
	A (microsiemens)
	B (microsiemens)
}

INITIAL {
	LOCAL tp
	if (tauRise/tauFall > .9999) {
		tauRise = .9999*tauFall
	}
	A = 0
	B = 0
	tp = (tauRise*tauFall)/(tauFall - tauRise) * log(tauFall/tauRise)
	factor = -exp(-tp/tauRise) + exp(-tp/tauFall)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	C = 1 / (1 + MgFactor * exp(-gamma * v))
	g = (B - A) * C
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tauRise
	B' = -B/tauFall
}

NET_RECEIVE(weight (microsiemens)) {
	state_discontinuity(A, A + weight*factor)
	state_discontinuity(B, B + weight*factor)
}
