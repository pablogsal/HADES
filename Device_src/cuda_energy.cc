

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
__device__ void energy(int cell,double *restd,double *eldeng,double *emini ,float density, float eps,double *res){

	//  Get values from HADES

	double c = %(CLIGHT)s;
	double e_mass= %(E_MASS)s;
	double external_density = %(EXT_DEN)s;
	double fdnonth = %(FDNONTH)s;
	double fenonth = %(FENONTH)s;
	double gamsp = %(GAM_SP)s;
	double ratene = %(RATE_ENE)s;

	//	 Density of nonthermal electrons in e-/cm3
	      *restd=fdnonth*density*external_density/e_mass;


   //  Energy of nonthermal electrons per unit of volume in ergs/cm3

    double enevol=fdnonth*density*external_density*fenonth*eps*c*c;

    //	//  Electron density per energy (to appear in the power law)
    //	      ELDENG(CEL)=(ENEVOL*(GAMSP-2.D0)/(1.D0-RATENE**(2.D0-GAMSP)))**
    //	                                                        (GAMSP-1.D0)
    //	            *((1.D0-RATENE**(1.D0-GAMSP))/RESTD(CEL)/(GAMSP-1.D0))**
    //	                                                        (GAMSP-2.D0);


	     *eldeng=pow((enevol*(-2 + gamsp))/(1 - pow(ratene,2 - gamsp)),-1 + \
	    		 gamsp)*pow((1 - pow(ratene,1 - gamsp))/((-1 + gamsp)*(*restd)),-2 \
	    		 + gamsp);

	//  Minimum energy of the nonthermal electrons in ergs

	     *emini=(c*c*e_mass*fenonth*(-2 + gamsp)*(1 - pow(ratene,1 - \
	    		 gamsp))*eps)/((-1 + gamsp)*(1 - pow(ratene,2 - gamsp)));

//	     *res=eps
}





__device__ void coeff(int i,float freqi, double *kap1,double *kap2, double *em1, double *em2,float *besselx,float *besself1,float *besself2,float *besselg1,float *besselg2,
		double *eldeng, double *emini,double *mfield, double *ang,int *error_test,double *res){

	// Get values from HADES

	float gamsp = %(GAM_SP)s;
	float ratene = %(RATE_ENE)s;
	float c_emiss = %(C_EMISS)s;
	float c_absorb = %(C_ABS)s;
	float c1 = %(C1)s;

//	  Limits values
	float xmax= freqi/mfield[i]/sin(ang[i])/emini[i]/emini[i]/c1;
	float xmin= xmax/ratene/ratene;


	int k;
	float infg1;
	float sepmin;
	float sep;

	float infgn;


//	  Calculating the index of the minimum and maximum values for the
//	  integrations
//	  Minimum value

	if(xmin > 30.0){
		xmin=30.0;
		infg1 = 162;
	}
	else{

		sepmin=1.0E3;
		k=1;

		sep=abs(xmin-besselx[k]);
		 while ( sep < sepmin ) {
			 sep=abs(xmin-besselx[k]);
		      k++;
		      sepmin=sep;
		  }

		 infg1=k-2;


	}

//	Maximum value

	if(xmax < 1.0E-9){

		xmax=1.0E-9;
		infgn=3;

	}
	else{

		sepmin=1.0E3;
		k=164;
		sep=abs(xmax-besselx[k]);

		 while ( sep < sepmin ) {
			 	 	 sep=abs(xmax-besselx[k]);
				      k--;
				      sepmin=sep;
				  }

		 infgn=k+1;

		 if(infgn>164){
			 infgn=164;
		 }

	}

//	infg1=1;
//	infgn=164;

	// End of evil magic calculating min and max value.

	float xint_f1=0;
	float xint_f2=0;
	float xint_g1=0;
	float xint_g2=0;


//	  Calculate the 2*VALUE of the integration of the Bessel functions
//	  First step integration


	xint_f1 = besself1[int(infg1)];
	xint_f2 = besself2[int(infg1)];
	xint_g1 = besselg1[int(infg1)];
	xint_g2 = besselg2[int(infg1)];


//   Subsequent steps


	for(int l=infg1+1; l<infgn-1; l++){

		xint_f1 = xint_f1 + besself1[l];
		xint_f2 = xint_f2 + besself2[l];
		xint_g1 = xint_g1 + besselg1[l];
		xint_g2 = xint_g2 + besselg2[l];


	}

//  Last sum



	xint_f1 = xint_f1 + besself1[int(infgn)];
	xint_f2 = xint_f2 + besself2[int(infgn)];
	xint_g1 = xint_g1 + besselg1[int(infgn)];
	xint_g2 = xint_g2 + besselg2[int(infgn)];


//	  Emission coefficients

	double emiss =  c_emiss * eldeng[i] * pow( double( abs(mfield[i] * sin(ang[i]))  ) , double( (gamsp+1.E0)/2.E0 ) ) * pow(  double( freqi ),  double( (1.E0-gamsp)/2.E0 )  );
	*em1 = emiss* ( xint_f1  + xint_g1  ) * 0.5E0;
	*em2 = emiss* ( xint_f1  - xint_g1  ) * 0.5E0;

//	  Absortion coefficients

	double absort =  c_absorb * eldeng[i] * pow( double( abs(mfield[i] * sin(ang[i])) ) , double( (gamsp+2.E0)/2.E0 ) ) * pow( double( freqi ) ,  double( (-4.E0-gamsp)/2.E0 )  );
	*kap1 = absort* ( xint_f2  + xint_g2  ) * 0.5E0 ;
	*kap2 = absort* ( xint_f2  - xint_g2  ) * 0.5E0 ;

//	*res= emiss;
//	error_test[0]=xmax;

}



__device__ void difsum(int cell, double *restd, double *eldeng, double *emini, double *deltao
		, double *mfield, double *ang, double *chih, double *rmds,double *res,float *besselx,float *besself1,float *besself2,float *besselg1,float *besselg2,int *error_test,
		double *sia, double *sib,double *suab,double *tau,double *rmea,double *botf){


	double c = %(CLIGHT)s;
	double ds= %(DS)s;
	double rshift= %(REDSHIFT)s;
	double obfreq= %(OBS_FREQ)s;

//	  Cosmological correction in 'ds', just for integrating the emission

	    double  dscl=ds/(1.0+rshift);

//     Initializating values to zero

	     double sia_temp =0.0;
	     double sib_temp =0.0;
	     double suab_temp=0.0;
	     double tau_temp=0.0;
	     double rmea_temp=0.0;
	     double botf_temp=0.0;
    	 double kap1=0.0;
    	 double kap2=0.0;
    	 double em1=0.0;
    	 double em2=0.0;
    	 double kap0=0.0;
    	 double freqi=0.0;
    	 double lambda=0.0;
    	 double dxfds=0.0;
    	 double schi=0.0;
    	 double cchi=0.0;
    	 double schi2=0.0;
    	 double schi4=0.0;
    	 double cchi2=0.0;
    	 double cchi4=0.0;
    	 double s2chi=0.0;
    	 double s2chi2=0.0;
    	 double dsint=0.0;


	     //  Loop along the integration column

	     for ( int i = 0; i < cell-1 ; i++ ) {



//	    	   Transformation to obtain the integration frequency
	    	 freqi=obfreq/deltao[i];





	    	 coeff(i,freqi,&kap1,&kap2, &em1, &em2,besselx,besself1,besself2,besselg1,besselg2, eldeng, emini,mfield,ang,error_test,res);




////	   	    Relativistic transformations of the absorption and emission coefficients
	    	 kap1=kap1/deltao[i];
	    	 kap2=kap2/deltao[i];
	    	 kap0=0.5E0*(kap1+kap2);
	    	 em1 = em1*deltao[i]*deltao[i];
	    	 em2 = em2*deltao[i]*deltao[i];

//
//	    	 Wavelength
	    	 lambda=c/freqi;

//	        Faraday rotation angle per unit of ds
	    	 dxfds=rmds[i]*lambda*lambda/dscl;




//	      	 Set angles
//
	    	 schi=sin(chih[i]);
	    	 cchi=cos(chih[i]);
	    	 schi2=schi*schi;
	    	 schi4=schi2*schi2;
	    	 cchi2=cchi*cchi;
	    	 cchi4=cchi2*cchi2;
	    	 s2chi=2.0*schi*cchi;
	    	 s2chi2=s2chi*s2chi;





//	    	 //Initialize stuff of the loop

	    	 dsint=0;

//	    	 for ( int iter = 0; iter <  1000; iter++ ) {
	    	  dsint=dscl/(int(dscl*kap0)*10.0+1.0);
	    	  sia_temp=sia_temp+dsint*(sia_temp*(-kap1*schi4-kap2*cchi4-0.5E0*kap0*s2chi2)+suab_temp*(0.25E0*(kap1-kap2)*s2chi+dxfds)+(em1*schi2+em2*cchi2));
	    	  sib_temp=sib_temp+dsint*(sib_temp*(-kap1*cchi4-kap2*schi4-0.5E0*kap0*s2chi2)+suab_temp*(0.25E0*(kap1-kap2)*s2chi-dxfds)+(em1*cchi2+em2*schi2));
	    	  suab_temp=suab_temp+dsint*(sia_temp*(0.5E0*(kap1-kap2)*s2chi-2.E0*dxfds)+sib_temp*(0.5E0*(kap1-kap2)*s2chi+2.E0*dxfds)-kap0*suab_temp-(em1-em2)*s2chi);

//	    	 }





//	    	   Integrate optical depth

	    	   tau_temp=tau_temp+dscl*kap0;
//
//	    	     Rotation measure (Cosm. corrections do not affect to rotation measure)

	    	   rmea_temp=rmea_temp+rmds[i]*ds;

//	    	    Mean Doppler boosting factor
	    	  botf_temp=botf_temp+deltao[i];

//	    	    Mean Dbf

	    	   botf_temp=botf_temp/cell;


	     }

	     *sia=sia_temp;
	     *sib=sib_temp;
	     *suab=suab_temp;
	     *tau=tau_temp;
	     *rmea=rmea_temp;
	     *botf=botf_temp;
//	     *res=sia_temp+sib_temp;


}


