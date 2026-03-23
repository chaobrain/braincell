COMMENT

NMDA channel, NR2B subunit and  calcium current
Modification made by Stefano Masoli PhD based on Nius 2006 and Santucci 2008

ENDCOMMENT

NEURON {
	POINT_PROCESS PC_NMDA_NR2B
	NONSPECIFIC_CURRENT i
	USEION ca READ eca WRITE ica
	
	RANGE Q10_diff,Q10_channel
	RANGE g , ic, ica
	RANGE Cdur,Erev,T,Tmax
	RANGE Rb, Ru, Rd, Rr, Ro, Rc,rb1,rb2,gmax,RdRate
	RANGE tau_1, tau_rec, tau_facil, U, u0 
	RANGE PRE
	RANGE Used
	RANGE MgBlock,v0_block,k_block
	RANGE diffuse,Trelease,lamd, Diff, M, Rd, nd, syntype, y_scale
	RANGE C0,C1,C2,C3,C4,D1,D2,O
}

UNITS {
	(nA) = (nanoamp)	
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
	(uM) = (micro/liter)
	(pS) = (picosiemens)
	(nS) = (nanosiemens)
	
	PI	= (pi)		(1)
    }
    
    PARAMETER {
	syntype
	: Parametri Presinaptici
	gmax		= 5000  	(pS)	: 7e3 : 4e4
	Q10_diff	= 1.4
	Q10_channel	= 2.2
	U 		= 0.2 (1) 	< 0, 1 >
	tau_rec 	= 8 (ms) 	< 1e-9, 1e9 > 	 
	tau_facil 	= 5 (ms) 	< 0, 1e9 > 	

	M	= 21.515	: numero di (kilo) molecole in una vescicola		
	Rd	= 1.03 (um)
	Diff	= 0.223 (um2/ms)
	tau_1 	= 1 (ms) 	< 1e-9, 1e9 >

	u0 		= 0 (1) < 0, 1 >
	Tmax		= 1  	(mM)

	: Postsinaptico, Santucci 2008 scheme
	
	Cdur	= 0.3	(ms)
	
	:binding and unbinding
	C0_C1_on = 9.06 (/mM /ms)
	C1_C0_off = 0.115 (/ms)
	C1_C2_on = 4.53 (/mM /ms)
	C2_C1_off = 0.23 (/ms)
	
	:desensitization
	C2_D1_on = 1.659 (/ms)
	D1_C2_off = 0.245 (/ms)
	C2_D2_on = 0.338 (/ms)
	D2_C2_off = 0.00274 (/ms)
	
	:middle closed
	C2_C3_on = 8.553 (/ms)
	C3_C2_off = 0.528 (/ms)
	C2_C4_on = 0.145 (/ms)
	C4_C2_off = 0.694 (/ms)
	
	:open
	C3_O_on = 0.145 (/ms)
	O_C3_off = 0.694 (/ms)
	C4_O_on = 8.553 (/ms)
	O_C4_off = 0.528 (/ms)	
	
	
	Erev	= -3.7  (mV)	: 0 (mV)
	
	v0_block = -20 (mV)	: -16 -8.69 (mV)	: -18.69 (mV) : -32.7 (mV)
	k_block  = 13 (mV)
	nd	 = 1
	kB	 = 0.44	(mM)

	: Diffusion			
	diffuse	= 1
	lamd	= 20 (nm)
	celsius (degC)
}


ASSIGNED {
	v		(mV)		: postsynaptic voltage
	i 		(nA)		: current = g*(v - Erev)
	ic 		(nA)		: current = g*(v - Erev)
	ica 		(nA)
	g 		(pS)		: actual conductance
	eca 		(mV)

	rb1		(/ms)    : binding
	rb2		(/ms)    : binding
	
	T		(mM)
	x 
	
	Trelease	(mM)
	tspike[100]	(ms)	: will be initialized by the pointprocess
	PRE[100]
	Mres		(mM)	
	
	MgBlock
	numpulses
	tzero
	gbar_Q10 (mho/cm2)
	Q10 (1)
	:nr2bi (mM)
	:y_scale
}

STATE {
	: Channel states (all fractions)
	C0		: single bound
	C1		: double bound
	C2		: closed 2
	C3		: closed 3
	C4		: closed 4
	D1		: desensitized one
	D2              : desensitized two
	O		: open
}

INITIAL {
	rates(v)
	C0 = 1
	C1 = 0
	C2 = 0
	C3 = 0
	C4 = 0
	D1 = 0
	D2 = 0
	O  = 0
	T  = 0
	numpulses=0

	gbar_Q10 = Q10_diff^((celsius-30)/10)
	Q10 = Q10_channel^((celsius-30)/10)

	Mres = 1e3 * (1e3 * 1e15 / 6.022e23 * M)     : (M) to (mM) so 1e3, 1um^3=1dm^3*1e-15 so 1e15
	FROM i=1 TO 100 { PRE[i-1]=0 tspike[i-1]=0 } :PRE_2[500]=0}
	tspike[0]=1e12	(ms)
	if(tau_1>=tau_rec){ 
		printf("Warning: tau_1 (%g) should never be higher neither equal to tau_rec (%g)!\n",tau_1,tau_rec)
		tau_rec=tau_1+1e-5
		:printf("tau_rec has been set to %g\n",tau_rec) 
	} 

}
	FUNCTION imax(a,b) {
	    if (a>b) { imax=a }
	    else { imax=b }
	}
	

FUNCTION diffusione(){	 
	LOCAL DifWave,i,cntc,fi,aaa
	DifWave=0
	cntc=imax(numpulses-100,0)
	FROM i=cntc  TO numpulses{
	    fi=fmod(i,100)
		tzero=tspike[fi]
		if(t>tzero){
		    aaa = (-Rd*Rd/(4*Diff*(t-tzero)))
		    if(fabs(aaa)<699){
			DifWave=DifWave+PRE[fi]*Mres*exp(aaa)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1
		    }else{
			if(aaa>0){
			    DifWave=DifWave+PRE[fi]*Mres*exp(699)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1
			}else{
			    DifWave=DifWave+PRE[fi]*Mres*exp(-699)/((4*PI*Diff*(1e-3)*lamd)*(t-tzero)) : ^nd nd =1
			}
		    }
		}
	}	
	diffusione=DifWave
}

BREAKPOINT {
	rates(v)
	SOLVE kstates METHOD sparse	
	
	g = gmax * gbar_Q10 * O
	
	: E' piu' logico spostare * MgBlock * PRE sul calcolo della corrente!
	i = (1e-6) * g * (v - Erev) * MgBlock
	ica = ((1e-6) * g * (v - Erev) * MgBlock)/10
	ic = i + ica
    }

KINETIC kstates {	
	:if ( diffuse && (t>tspike[0]) ) { Trelease= T + diffusione() } else { Trelease=T }
	Trelease = diffusione()
	rb1 = C0_C1_on * Trelease	
	rb2 = C1_C2_on * Trelease	
	~ C0 <-> C1	(rb1*Q10,C1_C0_off*Q10) 	
	~ C1 <-> C2	(rb2*Q10,C2_C1_off*Q10)	
	~ C2 <-> D1	(C2_D1_on*Q10,D1_C2_off*Q10)
	~ C2 <-> D2	(C2_D2_on*Q10,D2_C2_off*Q10)
	~ C2 <-> C3	(C2_C3_on*Q10,C3_C2_off*Q10)
	~ C2 <-> C4	(C2_C4_on*Q10,C4_C2_off*Q10)
	~ C3 <-> O	(C3_O_on*Q10,O_C3_off*Q10)
	~ C4 <-> O	(C4_O_on*Q10,O_C4_off*Q10)
	CONSERVE C0+C1+C2+C3+C4+D1+D2+O = 1
}

PROCEDURE rates(v(mV)) {
	: E' necessario includere DEPEND v0_block,k_block per aggiornare le tabelle!
	TABLE MgBlock DEPEND v0_block,k_block FROM -120 TO 30 WITH 150
	MgBlock = 1 / ( 1 + exp ( - ( v - v0_block ) / k_block ) )
}


NET_RECEIVE(weight, on, nspike, tzero (ms),y, z, u, tsyn (ms)) {LOCAL fi

: *********** ATTENZIONE! ***********
:
: Qualora si vogliano utilizzare impulsi di glutammato saturanti e' 
: necessario che il pulse sia piu' corto dell'intera simulazione
: altrimenti la variabile on non torna al suo valore di default.

INITIAL {
	y = 0
	z = 0
	u = u0
	tsyn = t
	nspike = 1
}
   if (flag == 0) { 
		: Qui faccio rientrare la modulazione presinaptica
		nspike = nspike + 1
		if (!on) {
			tzero = t
			on = 1				
			z = z*exp( - (t - tsyn) / (tau_rec) )	: RESCALED !
			z = z + ( y*(exp(-(t - tsyn)/tau_1) - exp(-(t - tsyn)/(tau_rec)))/((tau_1/(tau_rec))-1) ) : RESCALED !
			y = y*exp(-(t - tsyn)/tau_1)			
			x = 1-y-z
				
			if (tau_facil > 0) { 
				u = u*exp(-(t - tsyn)/tau_facil)
				u = u + U * ( 1 - u )							
			} else { u = U }
			
			y = y + x * u
			
			T=Tmax*y
			fi=fmod(numpulses,100)
			PRE[fi]=y	: PRE[numpulses]=y
			
			:PRE=1	: Istruzione non necessaria ma se ommesso allora le copie dell'oggetto successive alla prima non funzionano!
			:}
			: all'inizio numpulses=0 !			
			
			tspike[fi] = t
			numpulses=numpulses+1
			tsyn = t
			
		}
		net_send(Cdur, nspike)	 
    }
	if (flag == nspike) { 
			tzero = t
			T = 0
			on = 0
	}
}

