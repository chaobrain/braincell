TITLE Cerebellum Granule Cell Model

COMMENT
basato sul modello di Raman a 13 stati. genera corrente di sodio transiente, persistente e risorgente
with Long-Term Inactivation States L3,L4,L5,L6.

#modified by Stefano Masoli PhD to be the channel it was supposed to be from the beginning.
ENDCOMMENT

NEURON {
	SUFFIX NaFHF_MA20_GrC
	USEION na READ ena WRITE ina
	RANGE gnabar, ina, g
	RANGE gamma, delta, epsilon, Con, Coff, Oon, Ooff, Lon, Loff
	RANGE Aalfa, Valfa, Abeta, Vbeta, Ateta, Vteta, Agamma, Adelta, Aepsilon, ACon, ACoff, AOon, AOoff
	RANGE n1,n2,n3,n4
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
	
	ACon = 0.025    ( /ms) 
	ACoff = 0.5     ( /ms)
	AOon = 0.75     ( /ms)
	AOoff = 0.002   ( /ms)
	
	n1 = 5.422
	n2 = 3.279
	n3 = 1.83
	n4 = 0.738
	
	:L3 to L6
	ALon = 0.001   ( /ms)
	ALoff = 0.5    ( /ms) :0.15
	
	c = 20
	d = 0.075	
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
	Lon
	Loff
	a
	b
	Q10
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
	L3
	L4
	L5
	L6
}


INITIAL {
	Q10 =3^((celsius-20(degC))/10 (degC))
	gamma = Q10 * Agamma
	delta = Q10 * Adelta
	epsilon = Q10 * Aepsilon
	Con = Q10 * ACon
	Coff = Q10 * ACoff
	Oon = Q10 * AOon
	Ooff = Q10 * AOoff
	
	Lon = Q10 * ALon
	Loff = Q10 * ALoff
	a = (Oon/Con)^0.25
	b = (Ooff/Coff)^0.25
	SOLVE seqinitial

}

BREAKPOINT {
	SOLVE kstates METHOD sparse
	g = gnabar * O	      	: (mho/cm2)
	ina = g * (v - ena)  	: (mA/cm2)
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
	
	: 3 riga
	~ L3 <-> L4 (n3*alfa(v)*c,n2*alfa(v)*d)
	~ L4 <-> L5 (n4*alfa(v)*c,n1*alfa(v)*d)
	~ L5 <-> L6 (gamma,delta)
	
	: connette 1 riga con 2 riga
	~ C1 <-> I1 (Con,Coff)
	~ C2 <-> I2 (Con*a,Coff*b)
	~ C3 <-> I3 (Con*a*a,Coff*b*b)
	~ C4 <-> I4 (Con*a*a*a,Coff*b*b*b)
	~ C5 <-> I5 (Con*a*a*a*a,Coff*b*b*b*b)
	~  O <-> I6 (Oon,Ooff)
	
	: connette 1 riga con 3 riga
	~ C3 <-> L3 (Lon,Loff)
	~ C4 <-> L4 (Lon*c,Loff*d)
	~ C5 <-> L5 (Lon*c*c,Loff*d*d)
	~  O <-> L6 (Lon*c*c,Loff*d*d)
	
	CONSERVE C1+C2+C3+C4+C5+O+OB+I1+I2+I3+I4+I5+I6+L3+L4+L5+L6=1
}

LINEAR seqinitial {
	~          I1*Coff + C2*n4*beta(v) - C1*(Con+n1*alfa(v)) = 0
	~ C1*n1*alfa(v) + I2*Coff*b + C3*n3*beta(v) - C2*(n4*beta(v)+Con*a+n2*alfa(v)) = 0
	~ C2*n2*alfa(v) + I3*Coff*b*b + C4*n2*beta(v) + L3*Loff - C3*(n3*beta(v)+Con*a*a+n3*alfa(v)+Lon) = 0
	~ C3*n3*alfa(v) + I4*Coff*b*b*b + C5*n1*beta(v) + L4*Loff*d - C4*(n2*beta(v)+Con*a*a*a+n4*alfa(v)+Lon*c) = 0
	~ C4*n4*alfa(v) + I5*Coff*b*b*b*b + O*delta + L5*Loff*d*d - C5*(n1*beta(v)+Con*a*a*a*a+gamma+Lon*c*c) = 0
	~ C5*gamma + OB*teta(v) + I6*Ooff + L6*Loff*d*d - O*(delta+epsilon+Oon+Lon*c*c) = 0
	~ O*epsilon - OB*teta(v) = 0

	~          C1*Con + I2*n4*beta(v)*b - I1*(Coff+n1*alfa(v)*a) = 0
	~ I1*n1*alfa(v)*a + C2*Con*a + I3*n3*beta(v)*b - I2*(n4*beta(v)*b+Coff*b+n2*alfa(v)*a) = 0
	~ I2*n2*alfa(v)*a + C3*Con*a*a + I4*n2*beta(v)*b - I3*(n3*beta(v)*b+Coff*b*b+n3*alfa(v)*a) = 0
	~ I3*n3*alfa(v)*a + C4*Con*a*a*a + I5*n1*beta(v)*b - I4*(n2*beta(v)*b+Coff*b*b*b+n4*alfa(v)*a) = 0
	~ I4*n4*alfa(v)*a + C5*Con*a*a*a*a + I6*delta - I5*(n1*beta(v)*b+Coff*b*b*b*b+gamma) = 0

	~          C3*Lon + L4*n2*alfa(v)*d - L3*(Loff+n3*alfa(v)*c) = 0
	~ L3*n3*alfa(v)*c + C4*Lon*c + L5*n1*alfa(v)*d - L4*(n2*alfa(v)*d+Loff*d+n4*alfa(v)*c) = 0
	~ L4*n4*alfa(v)*c + C5*Lon*c*c + L6*delta - L5*(n1*alfa(v)*d+Loff*d*d+gamma) = 0
	~ L5*gamma + O*Lon*c*c - L6*(delta+Loff*d*d) = 0

	~ C1+C2+C3+C4+C5+O+OB+I1+I2+I3+I4+I5+I6+L3+L4+L5+L6=1
}
