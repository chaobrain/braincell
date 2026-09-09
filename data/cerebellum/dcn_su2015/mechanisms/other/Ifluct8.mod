TITLE Ornstein-Uhlenbeck fluctuating current
 
: no checking for tau = 0

UNITS { 
	(nA) = (nanoamp) 
	(mV) = (millivolt) 
} 
 

NEURON { 
	POINT_PROCESS Ifluct8
	NONSPECIFIC_CURRENT i
	RANGE i0, i1, tau, std, deriv
} 
 

PARAMETER { 
	i0 = 0.0 		(nA)
	tau = 2		(ms)
	std = 0.2		(nA)
}	


ASSIGNED {
	v (mV) 
	i (nA)
	deriv (nA/ms)
}


STATE {
	i1 (nA)
}


INITIAL {
	i1 = i0
	i = i0
}


BREAKPOINT { 
	SOLVE states METHOD cnexp
	i = i1
} 


DERIVATIVE states { 
	deriv = (i0 - i1)/tau + sqrt(2*std^2/tau) * normrand(0,1)
	i1' = deriv
}


PROCEDURE new_seed(seed) {
	set_seed(seed)
	VERBATIM
	  printf("Setting random generator with seed = %g\n", _lseed);
	ENDVERBATIM
}
