COMMENT
K_dr channel from Schweighofer et al 1999.
The referred model is an inferior olive neuron

B. Torben-Nielsen @ HUJI, 21-10-2010
ENDCOMMENT

NEURON {
       SUFFIX Kdr_ZH19_IO
       USEION k READ ek WRITE ik
       RANGE gbar,g,i,ninf,taun,n : now i can access these variables
}

UNITS {
      (S) = (siemens)
      (mS) = (millisiemens)
      (mV) = (millivolt)
      (mA) = (milliamp)
}

PARAMETER {
	  gbar = 18 (mS/cm2)
	  ek = -75 (mV)
}

ASSIGNED {
	 v (mV)
	 ik (mA/cm2)
	 i (mA/cm2)
	 g (mS/cm2)
	 ninf 
	 taun (ms)
}

STATE { n }

INITIAL {
	rates(v)
	n = ninf
}

BREAKPOINT {
	   rates(v)
	   SOLVE states METHOD cnexp
	   g = gbar *n*n*n*n
	   i = g * (v - ek)*(0.001)
	   ik = i
}

DERIVATIVE states {
	   n' = (ninf -n)/taun
}

PROCEDURE rates(v (mV)) {
	  LOCAL a_n , b_n
	  UNITSOFF
	  if(fabs(v+41.0) < 1e-6) {
	  	    : printf("v=%g\n", v)
	  	    a_n=(v+41.00001)/( 1-exp( -(v+41.00001)/10 ) )
	  } else {
	    a_n=(v+41)/( 1-exp( -(v+41)/10 ) )
	  }
	  b_n=12.5*exp( -(v+51)/80 )
	  ninf=a_n/(a_n+b_n)
	  taun=10/( a_n+b_n ) : was 5
	  UNITSON
}
