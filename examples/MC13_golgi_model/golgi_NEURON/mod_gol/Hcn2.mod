TITLE Cerebellum Golgi Cell Model

COMMENT

Author:L. Forti & S. Solinas
Data from: Santoro et al. J Neurosci. 2000
Last revised: April 2006

From Golgi_hcn2 to HCN2

ENDCOMMENT

NEURON {

        SUFFIX HCN2
        
	NONSPECIFIC_CURRENT ih
        
	RANGE Q10_diff,Q10_channel,gbar_Q10, ic

	RANGE o_fast_inf, o_slow_inf, tau_f, tau_s, gbar, ehcn2, g, o
        
	:GLOBAL o_fast_inf, o_slow_inf
}       
        
UNITS {
        
        (mA) = (milliamp)
        
	(mV) = (millivolt)
        
	(S)  = (siemens)
        
}


PARAMETER {
        
	celsius  (degC)
	gbar = 8e-5   (S/cm2)   < 0, 1e9 >
	Q10_diff	= 1.5
	Q10_channel	= 3
        ehcn2 = -20 (mV)
	
	Ehalf = -81.95 (mV)
	c = 0.1661 (/mV)
	
	rA = -0.0227 (/mV)
        rB = -1.4694 (1)
        tCf = 0.0269 (1)
        tDf = -5.6111 (mV)
	tEf = 2.3026 (/mV)
	tCs = 0.0152 (1)
        tDs = -5.2944 (mV)
	tEs = 2.3026 (/mV)
}

ASSIGNED {
	ih		(mA/cm2)
        v               (mV)
	g		(S/cm2)
	o_fast_inf
        o_slow_inf
        tau_f           (ms)
	tau_s           (ms)
	gbar_Q10 (mho/cm2)
	Q10     (1)
	ic
	o
    }

INITIAL {
    
    gbar_Q10 = gbar*(Q10_diff^((celsius-23)/10))
    rate(v)
    o_fast = o_fast_inf
    o_slow = o_slow_inf
}

STATE {	o_fast o_slow }		: fraction of fast and slow open channels


BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gbar_Q10 * (o_fast + o_slow)
        ih = g * (v - ehcn2)
	ic = ih
	o = o_fast + o_slow
}

DERIVATIVE state {	
	rate(v)
	o_fast' = (o_fast_inf - o_fast) / tau_f
	o_slow' = (o_slow_inf - o_slow) / tau_s
}


FUNCTION r(potential (mV),r1,r2)  { 	:fraction of fast component in double exponential
        UNITSOFF
	if (potential >= -64.70)  {
			r = 0
	} else{ 
	    if (potential <= -108.70)  {
		r = 1
	    } else{ 
		r =  (r1 * potential) + r2
	    }
	}
	UNITSON
}

FUNCTION tau_fast(potential (mV),t1,t2,t3) (ms) { 
	UNITSOFF
	Q10 = Q10_channel^((celsius -23) / 10)
        tau_fast = exp(t3 * ((t1 * potential) - t2)) / Q10
	UNITSON
}

FUNCTION tau_slow(potential (mV) ,t1,t2,t3) (ms) { 
	UNITSOFF
	Q10 = Q10_channel^((celsius -23) / 10)
        tau_slow = exp(t3 * ((t1 * potential) - t2)) / Q10
	UNITSON
}

FUNCTION o_inf(potential (mV),Ehalf,c)  { 
	UNITSOFF
        o_inf = 1 / (1 + exp((potential - Ehalf) * c))
        UNITSON
}

PROCEDURE rate(v (mV)) { 
	TABLE o_fast_inf, o_slow_inf, tau_f, tau_s
	DEPEND celsius FROM -100 TO 30 WITH 13000
	o_fast_inf = r(v,rA,rB) * o_inf(v,Ehalf,c)
        o_slow_inf = (1 - r(v,rA,rB)) * o_inf(v,Ehalf,c)
	tau_f =  tau_fast(v,tCf,tDf,tEf)
	tau_s =  tau_slow(v,tCs,tDs,tEs)
}
