:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 1367 $]
:[$Date: 2010-03-26 15:17:59 +0200 (Fri, 26 Mar 2010) $]
:[$Author: rajnish $]
:Comment :
:Reference :Molecular identification of a component of delayed rectifier current in gastrointestinal smooth muscles. Am. J. Physiol., 1998, 274, G901-11

NEURON	{
	SUFFIX Kv2p2_0010_MA20_GrC
	USEION k READ ek WRITE ik
	RANGE gKv2_2bar, gKv2_2, ik, BBiD 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gKv2_2bar = 0.00001 (S/cm2) 
	BBiD = 10 
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gKv2_2	(S/cm2)
	mInf
	mTau
	hInf
	hTau
}

STATE	{ 
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gKv2_2 = gKv2_2bar*m*h
	ik = gKv2_2*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
	UNITSOFF 
		mInf = 1/(1+exp(((v -(5.000))/(-12.000)))) 
		mTau = 130.000/(1+exp(((v -(-46.560))/(-44.140)))) 
		hInf = 1/(1+exp(((v -(-16.300))/(4.800)))) 
		hTau = 10000.000/(1+exp(((v -(-46.560))/(-44.140))))
	UNITSON
}
