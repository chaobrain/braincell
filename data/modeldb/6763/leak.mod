TITLE LEAK
 

UNITS {
        (pA) = (picoamp)
        (molar) = (1/liter)
	(mV) =	(millivolt)
        (uS) = (micromho)
	(mA) =	(milliamp)
	(mM) =	(millimolar)
        F = (faraday) (coulomb)
        R = (mole k) (mV-coulomb/degC)
}


INDEPENDENT {v FROM -100 TO 50 WITH 50 (mV)}

NEURON {
	SUFFIX leak
	USEION ca  WRITE ica
	USEION na READ nai,ena  WRITE ina
	USEION k  WRITE ik
	USEION cl  WRITE icl VALENCE 1
	RANGE  ileak,ina,ica,ik,gnabar,gkbar,nai,nainit,ggabaa
 
}


PARAMETER {
        dt (ms)
        gcabar =  0.0 (uS/cm2)
        gnabar = 16.3 (uS/cm2)
        gkbar = 83.7 (uS/cm2)
        ggabaa = 0.0 (uS/cm2)
        eca =  20 (mV)
        ecl =  -65 (mV)
        ek =  -100 (mV)
        ena     (mV)
        nao = 145 (mM)
        nai   (mM)
        nainit = 4  (mM)
        celsius = 35  (degC)
        
 
}

ASSIGNED { 
           ica		(mA/cm2)
           ina		(mA/cm2)
           ik		(mA/cm2)
           icl		(mA/cm2)
        ileak (mA/cm2)
}


BREAKPOINT {
        ena = R*(celsius+273.15)/F*log(nao/nai)
	ica = (0.000001)*gcabar*(v-eca)
	ina = (0.000001)*gnabar*(v-ena)
	icl = (0.000001)*ggabaa*(v-ecl)
	ik = (0.000001)*gkbar*(v-ek)
        ileak= ica + ina + ik 
}


COMMENT
INITIAL{
       nai = nainit
        ena = R*(celsius+273.15)/F*log(nao/nai)}
ENDCOMMENT
