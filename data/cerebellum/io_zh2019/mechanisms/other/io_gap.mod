COMMENT
A gap junction mechanism that is under blockade from GABAergic input
ENDCOMMENT

NEURON {
	POINT_PROCESS iogap
	POINTER vgap
	RANGE tau1, tau2, g, i, noise
	RANGE cc, maxblock
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1 = 60 (ms) <1e-9,1e9>
	tau2 = 200 (ms) <1e-9,1e9>
	g = 1.0 (uS)
	noise = 0 (nA)
	maxblock = 0.9
}

ASSIGNED {
	i (nA)
	v (millivolt)
	vgap (millivolt)
	factor
	cc
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	cc = 1
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	cc = 1 - maxblock*tanh(B - A)
	i = g * cc * (v - vgap) + noise
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight) {
	A = A + weight*factor
	B = B + weight*factor
}
