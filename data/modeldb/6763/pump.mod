TITLE PUMP
 

UNITS {
       (molar) = (1/liter)
        (pA) = (picoamp)
	(mV) =	(millivolt)
        (uS) = (micromho)
	(mA) =	(milliamp)
	(mM) =	(millimolar)
}


INDEPENDENT {v FROM -100 TO 50 WITH 50 (mV)}

NEURON {
	SUFFIX pump
	USEION na READ nai  WRITE ina
	USEION k  WRITE ik
	RANGE  inapump,ipumpmax,n,km
 
}


PARAMETER {
        dt (ms)
        nai   (mM)
        ipumpmax  = 0.04   (mA/cm2)
        km = 10.0        (mM)
        n=1.5

        nainit = 4  (mM)
        celsius = 35  (degC)
        
 
}

ASSIGNED { 
           ina		(mA/cm2)
           ik		(mA/cm2)
        inapump (mA/cm2)
}


BREAKPOINT {
        inapump = ipumpmax*(1/(1 + pow(km/nai,n)))
	ina = 3.0*inapump
	ik = -2.0*inapump
}


COMMENT
INITIAL{
       nai = nainit}
ENDCOMMENT
