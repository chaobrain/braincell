TITLE Cerebellum Granule Cell Model

COMMENT
Based on Raman 13 state model. Adapted from Magistretti et al, 2006.
ENDCOMMENT

NEURON {
	SUFFIX Nav_MA20_GrC
	USEION na READ ena WRITE ina
	RANGE gnabar, ina, g
	RANGE gamma, delta, epsilon, Con, Coff, Oon, Ooff
	RANGE Aalfa, Valfa, Abeta, Vbeta, Ateta, Vteta, Agamma, Adelta, Aepsilon, ACon, ACoff, AOon, AOoff
	RANGE n1,n2,n3,n4, alpha_d, beta_d, teta_d
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
	celsius = 32  	(degC)
	ena = 87.39		(mV)
	gnabar = 0.013	(mho/cm2)
	
	Aalfa = 353.91 ( /ms)
	Valfa = 13.99 ( /mV) 
	Abeta = 1.272  ( /ms)
	Vbeta = 13.99 ( /mV)
	Agamma = 150 ( /ms)
	Adelta = 40  ( /ms)
	Aepsilon = 1.75 ( /ms)
	Ateta = 0.0201 ( /ms)
	Vteta = 25
	
	ACon = 0.005    ( /ms) 
	ACoff = 0.5     ( /ms)
	AOon = 0.75     ( /ms)
	AOoff = 0.005   ( /ms)
	
	n1 = 5.422
	n2 = 3.279
	n3 = 1.83
	n4 = 0.738
}

ASSIGNED {
	ina  (mA/cm2)
	g   (mho/cm2)
	
	gamma
	delta
	epsilon
	Con
	Coff
	Oon
	Ooff
	a
	b
	Q10
	:alpha_d
	:beta_d
	:teta_d	
}

STATE {
	C1
	C2
	C3
	C4
	C5
	O
	OB
	I1
	I2
	I3
	I4
	I5
	I6
}


INITIAL {
	C1=1
	C2=0
	C3=0
	C4=0
	C5=0
	O=0
	OB=0
	I1=0
	I2=0
	I3=0
	I4=0
	I5=0
	I6=0
	Q10 =3^((celsius-20(degC))/10 (degC))
	gamma = Q10 * Agamma
	delta = Q10 * Adelta
	epsilon = Q10 * Aepsilon
	Con = Q10 * ACon
	Coff = Q10 * ACoff
	Oon = Q10 * AOon
	Ooff = Q10 * AOoff
	a = (Oon/Con)^0.25
	b = (Ooff/Coff)^0.25

}

BREAKPOINT {
	SOLVE kstates METHOD sparse
	g = gnabar * O	      	: (mho/cm2)
	ina = g * (v - ena)  	: (mA/cm2)
	:alpha_d = alfa(v) 
	:beta_d = beta(v) 
	:teta_d = teta(v) 
}


FUNCTION alfa(v(mV))(/ms){ 
	alfa = Q10*Aalfa*exp(v/Valfa) 
}

FUNCTION beta(v(mV))(/ms){ 
	beta = Q10*Abeta*exp(-v/Vbeta) 
}

FUNCTION teta(v(mV))(/ms){ 
	teta = Q10*Ateta*exp(-v/Vteta) 
}
 

KINETIC kstates {
	: 1 riga
	~ C1 <-> C2 (n1*alfa(v),n4*beta(v))
	~ C2 <-> C3 (n2*alfa(v),n3*beta(v))
	~ C3 <-> C4 (n3*alfa(v),n2*beta(v))
	~ C4 <-> C5 (n4*alfa(v),n1*beta(v))
	~ C5 <-> O  (gamma,delta)
	~  O <-> OB (epsilon,teta(v))
	
	: 2 riga
	~ I1 <-> I2	(n1*alfa(v)*a,n4*beta(v)*b)
	~ I2 <-> I3	(n2*alfa(v)*a,n3*beta(v)*b)
	~ I3 <-> I4	(n3*alfa(v)*a,n2*beta(v)*b)
	~ I4 <-> I5 (n4*alfa(v)*a,n1*beta(v)*b)
	~ I5 <-> I6 (gamma,delta)
	
	: connette 1 riga con 2 riga
	~ C1 <-> I1 (Con,Coff)
	~ C2 <-> I2 (Con*a,Coff*b)
	~ C3 <-> I3 (Con*a^2,Coff*b^2)
	~ C4 <-> I4 (Con*a^3,Coff*b^3)
	~ C5 <-> I5 (Con*a^4,Coff*b^4)
	~  O <-> I6 (Oon,Ooff)
	
	CONSERVE C1+C2+C3+C4+C5+O+OB+I1+I2+I3+I4+I5+I6=1
}
