COMMENT by Johannes Luthman: 
Based on NEURON 6.0's built-in exp2syn.mod.
Changes made to the original: 
* tau1 renamed tauRise; tau2, tauFall
* restructuring of NEURON block
* microsiemens changed to siemens for consistency with the other NMODLs.


Original comment: 
Two state kinetic scheme synapse described by rise time tauRise,
and decay time constant tauFall. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tauRise and 1/tauFall is
 A = a*exp(-t/tauRise) and
 G = a*tauFall/(tauFall-tauRise)*(-exp(-t/tauRise) + exp(-t/tauFall))
	where tauRise < tauFall

If tauFall-tauRise -> 0 then we have a alphasynapse.
and if tauRise -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS DCNsyn
	NONSPECIFIC_CURRENT i
	RANGE g, i, e, tauRise, tauFall
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
}

PARAMETER {
	tauRise = 1 (ms)
	tauFall = 1 (ms)
	e = 0 (mV)
}

ASSIGNED {
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
	g = B - A
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
