COMMENT
Somatic h channel from Schweighofer et al., 1999
Xu Zhang @ UConn, 6-22-2018
ENDCOMMENT

NEURON {
       SUFFIX HCN_ZH19_IO
       NONSPECIFIC_CURRENT ih
       RANGE gbar,g,ih,qinf,tauq,q,eh : now i can access these variables
}

UNITS {
      (S) = (siemens)
      (mS) = (millisiemens)
      (mV) = (millivolt)
      (mA) = (milliamp)
}

PARAMETER {
	  eh = -43 (mV)
	  gbar = 0.15 (mS/cm2)
}

ASSIGNED {
	 v (mV)
	 ih (mA/cm2)
	 g (mS/cm2)
	 qinf
	 tauq (ms)
}

STATE {
      q
}

INITIAL {
	rates(v)
	q = qinf
}

BREAKPOINT {
	   SOLVE states METHOD cnexp
	   g = gbar *q
	   ih = g * (v - eh)*(0.001) : 0.001 for converting mS to S
}

DERIVATIVE states {
	   rates(v)
	   q' = (qinf-q)/tauq
}

PROCEDURE rates(v (mV)) {
	  UNITSOFF
	  qinf = 1/(1+exp((v+75)/5.5))
	  tauq = 1/(exp(-0.086*v-14.6) + exp(0.07*v-1.87))
	  UNITSON
}
