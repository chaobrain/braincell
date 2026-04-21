COMMENT
Na channel from Schweighofer et al 1999.
The referred model is an inferior olive neuron

B. Torben-Nielsen @ HUJI, 21-10-2010
ENDCOMMENT

NEURON {
       SUFFIX Na_ZH19_IO
       USEION na READ ena WRITE ina
       RANGE gbar,g,i,minf,hinf,tauh,m : now i can access these variables
}

UNITS {
      (S) = (siemens)
      (mS) = (millisiemens)
      (mV) = (millivolt)
      (mA) = (milliamp)
}

PARAMETER {
	  qdeltat = 1
	  gbar = 70 (mS/cm2)
	  ena = 55 (mV)
}

ASSIGNED {
	 v (mV)
	 ina (mA/cm2)
	 i (mA/cm2)
	 g (mS/cm2)
	 minf 
	 hinf 
	 tauh (ms)
}

STATE {
      m h
}

INITIAL {
	rates(v)
	h = hinf
  	m = minf
}

BREAKPOINT {
	   rates(v)
	   SOLVE states METHOD cnexp
	   g = gbar *m*m*m*h
	   i = g * (v - ena)*(0.001)
	   ina = i
}

DERIVATIVE states {
	h' = (hinf -h)/tauh
    m' = (minf - m)/0.001
}

PROCEDURE rates(v (mV)) {
	  LOCAL a_h, b_h, a_m,b_m
	  UNITSOFF
	  if(fabs(v+41.0) < 1e-6) {
	    a_m=(0.1*(v+41.000001)) / ( 1-exp( -(v+41.000001)/10 ) )
	  } else {
	    a_m=(0.1*(v+41)) / ( 1-exp( -(v+41)/10 ) )
	  }
	  b_m=9.0*exp( -(v+66)/20 )
	  minf=a_m/(a_m+b_m)
	  
	  a_h=5.0*exp( -(v+60)/15 )
    if (fabs(v+50.0) < 1e-6) {
        b_h = (v+50.000001) / (1 - exp(-(v+50.000001)/10))
    } else {
        b_h = (v+50) / (1 - exp(-(v+50)/10))
    }
	  hinf=a_h/(a_h+b_h)
	  tauh=250/( a_h+b_h ) : was 170, 250
	  UNITSON
}

