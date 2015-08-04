

//__device__ double energy(int x, int y, int dim_x){
//
//	//  Density of nonthermal electrons in e-/cm3
//	      RESTD(CEL)=FDNONTH*DENSTY(IXA,IYA)*EXTDEN/ELMASS
//
//    //  Energy of nonthermal electrons per unit of volume in ergs/cm3
//	      ENEVOL=FDNONTH*DENSTY(IXA,IYA)*EXTDEN*FENONTH*EPS(IXA,IYA)*CLIGHT*CLIGHT;
//
//	//  Electron density per energy (to appear in the power law)
//	      ELDENG(CEL)=(ENEVOL*(GAMSP-2.D0)/(1.D0-RATENE**(2.D0-GAMSP)))**
//	                                                        (GAMSP-1.D0)
//	            *((1.D0-RATENE**(1.D0-GAMSP))/RESTD(CEL)/(GAMSP-1.D0))**
//	                                                        (GAMSP-2.D0);
//
//	//  Minimum energy of the nonthermal electrons in ergs
//	      EMINI(CEL)=FENONTH*EPS(IXA,IYA)*ELMASS*CLIGHT*CLIGHT*
//	                (GAMSP-2.D0)*(1.D0-RATENE**(1.D0-GAMSP))/
//	                (GAMSP-1.D0)/(1.D0-RATENE**(2.D0-GAMSP));
//
//
//}
//
//
__device__ void energy(int cell,float *restd,float *eldeng,float *emini ,float density, float eps){

	//  Get values from HADES

	float c = %(CLIGHT)s;
	float e_mass= %(E_MASS)s;
	float external_density = %(EXT_DEN)s;
	float fdnonth = %(FDNONTH)s;
	float fenonth = %(FENONTH)s;
	float gamsp = %(GAM_SP)s;
	float ratene = %(RATE_ENE)s;

	//	 Density of nonthermal electrons in e-/cm3
	      *restd=fdnonth*density*external_density/e_mass;


   //  Energy of nonthermal electrons per unit of volume in ergs/cm3

    float enevol=fdnonth*density*external_density*fenonth*eps*c*c;

    //	//  Electron density per energy (to appear in the power law)
    //	      ELDENG(CEL)=(ENEVOL*(GAMSP-2.D0)/(1.D0-RATENE**(2.D0-GAMSP)))**
    //	                                                        (GAMSP-1.D0)
    //	            *((1.D0-RATENE**(1.D0-GAMSP))/RESTD(CEL)/(GAMSP-1.D0))**
    //	                                                        (GAMSP-2.D0);


	     *eldeng=pow((enevol*(-2 + gamsp))/(1 - pow(ratene,2 - gamsp)),-1 + \
	    		 gamsp)*pow((1 - pow(ratene,1 - gamsp))/((-1 + gamsp)*(*restd)),-2 \
	    		 + gamsp);

	//  Minimum energy of the nonthermal electrons in ergs

	     *emini=(pow(c,2)*e_mass*fenonth*(-2 + gamsp)*(1 - pow(ratene,1 - \
	    		 gamsp))*eps)/((-1 + gamsp)*(1 - pow(ratene,2 - gamsp)));

}





__device__ void coeff(int i,float freqi, float *kap1,float *kap2, float *em1, float *em2,float *besselx,float *besself,
		              float *besselg, float *eldeng, float *emini,float *mfield, float *ang,int *error_test,float *res){

	// Get values from HADES

	float gamsp = %(GAM_SP)s;
	float ratene = %(RATE_ENE)s;
	float c_emiss = %(C_EMISS)s;
	float c_absorb = %(C_ABS)s;
	float c1 = %(C1)s;

//	  Limits values
	float xmax= freqi/mfield[i]/sin(ang[i])/emini[i]/emini[i]/c1;
	float xmin= xmax/ratene/ratene;


	*res=xmin;
//	error_test[0]=xmax;

}



__device__ void difsum(int cell, float *restd, float *eldeng, float *emini, float *deltao
		, float *mfield, float *ang, float *chih, float *rmds,float *res,float *besselx,float *besself,float *besselg,int *error_test){


	float c = %(CLIGHT)s;
	float ds= %(DS)s;
	float rshift= %(REDSHIFT)s;
	float obfreq= %(OBS_FREQ)s;

//	  Cosmological correction in 'ds', just for integrating the emission

	    float  dscl=ds/(1.0+rshift);

//     Initializating values to zero

	     float sia =0.0;
	     float sib =0.0;
	     float suab=0.0;
	     float tau=0.0  ;
	     float rmea=0.0  ;
	     float botf=0.0   ;
	     //  Loop along the integration column

	     for ( int i = 0; i < cell-1 ; i++ ) {



//	    	   Transformation to obtain the integration frequency
	    	 float freqi=obfreq/deltao[i];


	    	 float kap1=2.0;
	    	 float kap2=2.0;
	    	 float em1=1.0;
	    	 float em2=1.0;
	    	 coeff(i,freqi,&kap1,&kap2, &em1, &em2,besselx,besself,besselg, eldeng, emini,mfield,ang,error_test,res);


//	    	 *res=  kap1;

//	   	    Relativistic transformations of the absorption and emission coefficients

	    	 kap1=kap1/deltao[i];
	    	 kap2=kap2/deltao[i];
	    	 float kap0=0.5*(kap1+kap2);
	    	 em2 = em2*deltao[i]*deltao[i];

//	    	 Wavelength
	    	 float lambda=c/freqi;

//	        Faraday rotation angle per unit of ds
	    	 float dxfds=rmds[i]*lambda*lambda;

//	      	 Set angles
//
	    	 float schi=sin(chih[i]);
	    	 float cchi=cos(chih[i]);
	    	 float schi2=schi*schi;
	    	 float schi4=schi2*schi2;
	    	 float cchi2=cchi*cchi;
	    	 float cchi4=cchi2*cchi2;
	    	 float s2chi=2.0*schi*cchi;
	    	 float s2chi2=s2chi*s2chi;


	    	 //Initialize stuff of the loop

	    	 float dsint=0;
	    	 //CUIDADO CON ESTOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
	    	 for ( int k = 0; k < 10+0*int(dscl*kap0)*10 ; k++ ) {
//
	    		dsint=dscl/(int(dscl*kap0)*10.0+1.0);

	    		sia=sia+dsint*(sia*(-kap1*schi4-kap2*cchi4-0.5E0*kap0*s2chi2)+suab*(0.25E0*(kap1-kap2)*s2chi+dxfds)+em1*schi2+em2*cchi2);
	    		sib=sib+dsint*(sib*(-kap1*cchi4-kap2*schi4-0.5E0*kap0*s2chi2)+suab*(0.25E0*(kap1-kap2)*s2chi-dxfds)+em1*cchi2+em2*schi2);
	    	  suab=suab+dsint*(sia*(0.5E0*(kap1-kap2)*s2chi-2.E0*dxfds)+sib*(0.5E0*(kap1-kap2)*s2chi+2.E0*dxfds)-kap0*suab-(em1-em2)*s2chi);
	    	 }

//	    	   Integrate optical depth

	    	   tau=tau+dscl*kap0;

//	    	     Rotation measure (Cosm. corrections do not affect to rotation measure)

	    	   rmea=rmea+rmds[i]*ds;

//	    	    Mean Doppler boosting factor
	    	   botf=botf+deltao[i];

//	    	    Mean Dbf

	    	   botf=botf/cell;





	     }





}


