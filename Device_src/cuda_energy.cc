

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

	//  Density of nonthermal electrons in e-/cm3

	float c = 2.9979e10;
	float e_mass=9.1094e-28;
	float external_density = 1.0;
	float fdnonth = 1.0;
	float fenonth = 1.0;
	float gamsp = 5.345;
	float rate_energy = 1.0;

	//	 Density of nonthermal electrons in e-/cm3
	      *restd=fdnonth*density*external_density/e_mass;

   //  Energy of nonthermal electrons per unit of volume in ergs/cm3

    float enevol=fdnonth*density*external_density*fenonth*eps*c*c;

    //	//  Electron density per energy (to appear in the power law)
    //	      ELDENG(CEL)=(ENEVOL*(GAMSP-2.D0)/(1.D0-RATENE**(2.D0-GAMSP)))**
    //	                                                        (GAMSP-1.D0)
    //	            *((1.D0-RATENE**(1.D0-GAMSP))/RESTD(CEL)/(GAMSP-1.D0))**
    //	                                                        (GAMSP-2.D0);


    	 float gam_2 =(2.e0-gamsp);
	     float gam_1 =(1.e0-gamsp);
		 float gam_m1 =(gamsp-1.e0);
    	 float gam_m2=(gamsp-2.e0);

    	 float p1=enevol * gam_m2 / (1-  pow(rate_energy, gam_2 )     );
    	 float p2=(1.e0 -  pow( rate_energy, gam_1)   )   / *restd /  gam_m1 ;


	     *eldeng=pow(p1,gam_m1)*pow( p2, gam_m2 ) ;



}







