COMMENT
Ca channel from Manor (Rinzel, Segev, Yarom) 1997
Channel can cause sub-threshold oscillations in interplay with the leak

B. Torben-Nielsen @ HUJI, 7-10-2010
ENDCOMMENT

NEURON {
       SUFFIX Ca_ZH19_IO
       :USEION ca WRITE ica
       NONSPECIFIC_CURRENT i
       RANGE mMidV,gbar,g,i,minf,hinf,tauh,m,h,ecas : now i can access these variables
}

UNITS {
      (S) = (siemens)
      (mS) = (millisiemens)
      (mV) = (millivolt)
      (mA) = (milliamp)
}

PARAMETER {
	  ecas = 120 (mV)
	  gbar = 0.4 (mS/cm2)
      mMidV=-61 (mV) : -61 default from mnaor. can be set to other values, e.g., to test 'windowness' of the current
}

ASSIGNED {
	 v (mV)
	 i (mA/cm2)
	 g (mS/cm2)
	 minf 
	 hinf 
	 tauh (ms)
}

STATE {
      m 
      h
}

INITIAL {
	rates(v)
	h = hinf
	m = minf
}

BREAKPOINT {
	   SOLVE states METHOD cnexp
	   g = gbar *minf*h
	   i = g * (v - ecas)*(0.001)
	   :ica = i
}

DERIVATIVE states {
	   rates(v)
	   h' = (hinf -h)/tauh
}

PROCEDURE rates(v (mV)) {
	  : updates formulas with earlier activation, e_r=-70, e_l=-78 (i.e., at lower amplitudes)
	  :hinf =1/( 1+exp( (v+100.5)/8.6 ) )
	  :minf = 1/(  (1+exp((-75.5-v)/4.2)) *(1+exp((-75.5-v)/4.2))* (1+exp((-75.5-v)/4.2)) )
	  :tauh=55+30*(1/( 1+exp((v+99.0)/7.3) ))*exp((v+175.0)/30.0)

	  : below the original ones
	  UNITSOFF
	  hinf =1/( 1+exp( (v+85.5)/8.6 ) ) :85.5
	  tauh=40+30*(1/( 1+exp((v+84.0)/7.3) ))*exp((v+160.0)/30.0)
 	  minf = 1/(  (1+exp((mMidV-v)/4.2)) *(1+exp((mMidV-v)/4.2))* (1+exp((mMidV-v)/4.2)) )
	  m = minf
	  UNITSON
}
