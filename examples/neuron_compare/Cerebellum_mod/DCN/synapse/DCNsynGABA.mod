COMMENT by Johannes Luthman:

    This mechanism is based on DCNsyn.mod in this project. What's added here is
    paired-pulse depression of the synapse's current, based on Shin et al, 2007
    (PLOSone issue 5, e485, page 2), on which the changes in terminology
    compared to DCNsyn.mod are based.
    The depression is implemented via the change from [g = B - A] in DCNsyn.mod
    to [g = (B - A) * deprLevel] here, and the calculation of deprLevel on each
    input (NETRECEIVE).

ENDCOMMENT

NEURON {
	POINT_PROCESS DCNsynGABA
	NONSPECIFIC_CURRENT i
	RANGE g, i, e, tauRise, tauFall, startDeprLevel, deprLevel
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
}

PARAMETER {
	tauRise = 1 (ms)
	tauFall = 1 (ms)
	e = 0 (mV)
	startDeprLevel = 1 : set this in hoc to the depression level reached at 
	        : steady state (use the equation for relProbSS, below) by the 
	        : used GABAergic input frequency.
}

ASSIGNED {
    relProbSS (1) : This corresponds to Rss in the article (given in COMMENT at top).
    relProb[2] (1) : This corresponds to Rn and Rn-1 in the article.
    freq (1/s) : This corresponds to r in the article.
    tau (ms)
    tSpikes[2] (ms)
    ISI (ms)
    deprLevel (1) : level of synaptic depression. The conductance is
                  : multiplied by this factor in BREAKPOINT.
    notFirstSpike (1) : boolean used to set up values of previous step on first
                      : call to this mechanism.

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

    notFirstSpike = 0
}

BREAKPOINT {
    : Here the conductance is updated each time step, while the NET_RECEIVE block
    : is only invoked by being contacted by a NetCon object.
  	SOLVE state METHOD cnexp
   	g = (B - A) * deprLevel
   	i = g*(v - e)   	
}

DERIVATIVE state {
	A' = -A/tauRise
	B' = -B/tauFall
}

NET_RECEIVE(weight (microsiemens)) {
    deprLevel = giveFractionG()
    state_discontinuity(A, A + weight*factor)
	state_discontinuity(B, B + weight*factor)
}

FUNCTION giveFractionG() {
	if (notFirstSpike) {
        : Set the current spike to the present time, and calculate ISI as the
        : difference in time from the last pass through here.
        tSpikes[0] = tSpikes[1]
        tSpikes[1] = t
        ISI = tSpikes[1] - tSpikes[0]
        freq = 1000 / ISI
        
        :relProbSS = 0.08 + 0.60*exp(-2.84*freq) + 0.32*exp(-0.02*freq)
        if (freq <=10) {
        	relProbSS=0.06 +0.95*exp(-0.053*freq)
        }
        if (freq >10 && freq <=50)	{
        relProbSS= 0.647*exp(-0.007*freq)
        }
        if (freq >50 && freq <=100 )	{
        relProbSS= ((-0.28/50)*freq)+0.73
        }
        
        if (freq>100) {
        	relProbSS = 0.1
        }
        
        tau = 2 + 2500*exp(-0.274*freq) + 100*exp(-0.022*freq)
        relProb[1] = relProb[0] + (relProbSS - relProb[0]) * (1-exp(-ISI/tau))
        relProb[0] = relProb[1]

        giveFractionG = relProb[1]
	} else {
	    tSpikes[1] = t
	    relProb[0] = startDeprLevel
	    notFirstSpike = 1
	    giveFractionG = relProb[0]
	}
}
