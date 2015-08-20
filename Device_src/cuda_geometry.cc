

// Declaring test function. This is only to show how to use different funtions
// in a Kernel. We must declare the function first (first line) and the we can 
// define what it does in the second line.



//__device__ double dot(double a, double b);
//
//__device__ double cosa(float *x_min, float *x_max){
//
//    *x_min = 2;
//
//    *x_max = 3;
//
//}


__device__ void x_limits(float *x_min, float *x_max,int dim_x_old, int dim_y_old, float theta,float y_cell,
float z_cell)
{


    // Renaming things from input_par to easy reading
    // Notice that for the RMHD plane we are using the variable names:
    //
    // x= rho
    // y= z
float pi= 3.14159265358979323846264338328;
int scalex = %(SCALEX)s;
int scaley = %(SCALEY)s;
// CHAGE THIS!!!

int dim_x=dim_x_old*scalex;
int dim_y=dim_y_old*scaley;

    // As when we revolve the RMHD rho-z plane around the z axis we
    // are constructing a cylinder, the limits in the x coordinate
    // (perpendicular to the observer plane) change as we change the z
    // coordinate because we are slicing the cylinder at different heights.
    //
    // To take account of these we must calculate the limits in the RMHD slice
    // when doing the cylinder thing. This is easy when using the cylinder eq:
    //
    // X_max^2 = x^2 + z^2   (x,y,z is the LAB frame)
    //
    // So x_lim= Sqrt( x_lim^2-z^2)

float x_lim = sqrt(dim_x*dim_x-z_cell*z_cell);


    // Now, the limits for the y variable in the observer map change as the x_lim change. We must recalculate them
    // as we did in the image class.

float y_min = -sin(theta)*x_lim;
float y_max = sin(theta)*x_lim+cos(theta)*dim_y;

if(y_cell >= y_min && y_cell <= y_max){

    // Now we can calculate the x limits. These are a little tricky. There are two easy cases: theta=90,0. We
    // take account of these fist.

    	if( theta == pi/2.0){

      	  *x_min=0;
      	  *x_max=dim_y;

   	 	}

   	    if( theta == 0.0){

        *x_min=-round(x_lim);
        *x_max= round(x_lim);

	    }



    // Now for the serious business. The x lims are the point where the line y=y_cell cuts the RMHD rectangle.
    // These limits are a little tricky because it depend if the first crossing point is in the z-axis (parallel to) of the RMHD
    // rectangle of in the rho-axis (parallel to). So first we must calculate if the crossing point is in the z-axis (long one) or
    // in the rho axis (short) one. If the crossing is in the z axis (parallel to), we must use trigonometry to
    // calculate these
    // crossing points. The easiest way is to start with the line in the observer frame -> l=(t,y0), then using the
    // rotation matrix calculate this line in the LAB frame as l' = R.transpose * (t,y0) and now calculate the
    // crossing point of l' with the line (x_lim,t)->z axis or the line (t,dim_y) -> rho-axis. Then rotate the
    // crossing point to the observer frame and you are done.
    //
    // In this program we first calculate the crossing point with the z axis and then we check if we must calculate
    // the other one.

   	 if(theta != 0.0 && theta != pi/2){

        // X_max calculation

        // Cross with z axis in lab-frame
        float z_cross =  1.0/cos(theta) * ( y_cell + x_lim * sin(theta) );

            if (0 < z_cross && z_cross < dim_y){
            //This checks if the cross is long away the RMHD (x_lim,dim_y) coordinate.

            *x_max = 1.0/(cos(theta)) * (x_lim + y_cell * sin(theta));

            // Cross with z axis in obs-frame
           }
           else{

            *x_max = ( dim_y - y_cell * cos(theta) ) / sin(theta); // Cross with rho axis in OBS-frame
           }




       // X_min calculation

        z_cross = y_cell /cos(theta) - x_lim * tan(theta); // Cross with z axis in lab-frame

        if (0 < z_cross && z_cross < dim_y){

         //This checks if the cross is long away the RMHD (x_lim,dim_y) coordinate.

            *x_min =  - x_lim /cos(theta) + y_cell * tan(theta);
            // Cross with z axis in obs-frame
        }
        else{
            *x_min = -y_cell / tan(theta);
            // Cross with rho axis in OBS-frame

        }

	}


    //Hurray! Now we have to round the calculations.


		*x_max=round(*x_max);

		if (*x_min > *x_max){

			*x_min= *x_max;

		}


}
else{

*x_min = 0.0001;
*x_max = -0.0001;
}





} //End of function










__device__ void obs_to_rmhd(float *rho,float *zeta,float x,float y,float z,float theta,int dim_x,int dim_y){

    // Renaming things from input_par to easy reading
    // Notice that for the RMHD plane we are using the variable names:
    //
    // x= rho
    // y= z

//    float pi= 3.14159265358979323846264338328;
    // The x,y,z in lab frame (a rotation of x,y,z)

	int scalex = %(SCALEX)s;
	int scaley = %(SCALEY)s;

    float x_lab= cos(theta) * x - sin(theta) * y;
    float y_lab= cos(theta) * y + sin(theta) * x;
    float z_lab = z;

     // The rho and z coordinates

    *rho=round(sqrt(x_lab * x_lab + z_lab * z_lab )/scalex);
    *zeta= round(y_lab/scaley);

    //Beware of rho=-1 and rho > dim_x!!

    if(*rho<0){
    	*rho=0;
    }
    if(*rho>=dim_x)
    {
    	*rho=dim_x-1;
    }


    if(*zeta<=0){
    	*zeta=0;
    }
    if(*zeta>=dim_y)
    {
    	*rho=dim_y-1;
    }


}



__device__ void lorenz_calculator(int cell,int x_cell_out,int y_cell_out,int z_cell_out,float theta, float vx, float vy, float vz, float bx, float by, float bz,double *res
								 ,double *deltao, double *mfield , double *ang, double *chih,double *rmds, float restd){





	// Problems with x_cell, y_cell and z_cell =0;

	double x_cell= x_cell_out;
	double y_cell= y_cell_out;
	double z_cell= z_cell_out;

	if(x_cell == 0){
		x_cell = 1;
	}



	//Construct reusable numberic variables

	double cos_theta = cos(theta);
	double sin_theta = sin(theta);




	// Get variable values


	double rshift= %(REDSHIFT)s;
	double pi= 3.14159265358979323846264338328;


	//  Modulus of the velocity

	double beta = sqrt( vx*vx+vy*vy+vz*vz  );

	//  Lorentz factor
	double w =1.0/sqrt(1.0-beta*beta);

//	  Transformation of the cell's coordinates between the observer's and
//	  the source system where:
//	  Observer's system: x-axis towards the line of sight
//	  Observer's to Source's system: Negative rotation through the z-axis by THETA
//	  (I don't need now CSOUY)


	   double   csoux=cos_theta*x_cell-sin_theta*y_cell;
	   double   csouz=z_cell;


//	     Cosinus and sinus of the inclination angle of the grid frame plane
//	     (which contains the y-axis in the source's system and the cell) with
//	     respect to the plane z=0 in the source system (positive towards the
//	     source's system z-axis)

	   double      cphi=csoux/sqrt(csoux*csoux+csouz*csouz);
	   double      sphi=csouz/sqrt(csoux*csoux+csouz*csouz);


//	     Components of the velocity vector in the observer's frame, transforming
//	     from grid to source to observer


	   double    vobs_x= vx*cphi*cos_theta+vy*sin_theta-vz*sphi*cos_theta;
	   double    vobs_y=-vx*cphi*sin_theta+vy*cos_theta+vz*sphi*sin_theta;
	   double    vobs_z= vx*sphi+vz*cphi;


//	     Doppler boosting factor. Cosinus angle between velocity and line of sight
//	     is equal to escalar product of velocity and l.o.s (unity along x axis in
//	     observer's frame)
	   *deltao=1.0/w/(1.0-beta*(vobs_x/beta));

//	     Red-shift correction in the Doppler boosting factor
	   *deltao=*deltao/(1.0+rshift);

//		  Orientation of the cell's velocity vector in the observer's frame
	   	   // Posible bug in azicell!!!!!

		   double   azicell=atan2(vobs_y,vobs_x);
		   double   elecell=atan2(float(vobs_z),sqrt(float(vobs_x*vobs_x+vobs_y*vobs_y)));


		   //Get reusable values

		   double cos_azicell = cos(azicell);
		   double sin_azicell = sin(azicell);
		   double cos_elecell = cos(elecell);
		   double sin_elecell = sin(elecell);


//		     Cross product (v x B)
		   double v_por_b_x=vy*bz-vz*by;
		   double v_por_b_y=vz*bx-vx*bz;
		   double v_por_b_z=vx*by-vy*bx;

//		     Cros product [v x (v x B)]
		   double  v_por_vb_x=vy*v_por_b_z-vz*v_por_b_y;
		   double  v_por_vb_y=vz*v_por_b_x-vx*v_por_b_z;
		   double  v_por_vb_z=vx*v_por_b_y-vy*v_por_b_x;

//		     Escalar product (v.B)
		   double  v_esc_b=vx*bx+vy*by+vz*bz;

//		     Magnetic field in the fluid comovil frame, but with the orientation of
//		     the grid frame

		   double bf_x=w*(bx+v_por_vb_x)-w*w/(w+1.0)*v_esc_b*vx;
	       double bf_y=w*(by+v_por_vb_y)-w*w/(w+1.0)*v_esc_b*vy;
	       double bf_z=w*(bz+v_por_vb_z)-w*w/(w+1.0)*v_esc_b*vz;


//	         Total magnetic field strength in a frame comovil with the fluid

	        *mfield =sqrt(float(bf_x*bf_x+bf_y*bf_y+bf_z*bf_z));

//	          Components of the magnetic field comovil with the fluid, but rotated to
//	          the observer's frame (rotating from from grid to source to observer)

	       double bfo_x = bf_x*cphi*cos_theta+bf_y*sin_theta-bf_z*sphi*cos_theta;
	       double bfo_y = -bf_x*cphi*sin_theta+bf_y*cos_theta+bf_z*sphi*sin_theta;
	       double bfo_z = bf_x*sphi+bf_z*cphi;

//	          Now we introduce another rotation to the fluid frame (as seen in the
//	         observer's frame using azicell and elecell)

	       double bfl_x = bfo_x*cos_azicell*cos_elecell+bfo_y*sin_azicell*cos_elecell+bfo_z*sin_elecell;
	       double bfl_y = -bfo_x*sin_azicell+bfo_y*cos_azicell;
	       double bfl_z = -bfo_x*cos_azicell*sin_elecell-bfo_y*sin_azicell*sin_elecell+bfo_z*cos_elecell;


//	         ---> Light aberration
//	         Transformation of the line of sight from the observer's frame to the
//	        fluid's frame using the light aberration formulae.
//	         [We should also consider corrections by Hubble velocity (using redshift).]

	       double  raiz = sqrt(float(w*w*pow(beta - cos_azicell*cos_elecell, double(2.0) ) + cos_elecell*cos_elecell*sin_azicell*sin_azicell ));


//	         New value of ELECELL corrected by light aberration

	       double selecellp = sin_elecell/(w*(1.0-beta*cos_elecell*cos_azicell));
	       double celecellp = raiz/(w*(1.0-beta*cos_elecell*cos_azicell));
	       double elecellp =  atan2(float(selecellp),float(celecellp));

		   double sin_elecellp = sin(elecellp);
		   double cos_elecellp = cos(elecellp);


//	         New value of AZICELL corrected by light aberration

	         double sazicellp = cos_elecell*sin_azicell/raiz;
	       	 double cazicellp = w*(cos_elecell*cos_azicell-beta)/raiz;
	       	 double azicellp = atan2(float(sazicellp),float(cazicellp));

			   //Get reusable values


			   double sin_azicellp = sin(azicellp);
			   double cos_azicellp = cos(azicellp);



//	       	  Components of the line of sight in the fluid's frame corrected by
//	       	  light aberration
	       	double lfab_x = cos_azicellp*cos_elecellp;
	       	double lfab_y = -sin_azicellp;
	       	double lfab_z = -cos_azicellp*sin_elecellp;

//	       	  Angle between the line of sight and the magnetic field in the fluid's frame
//	       	  corrected by light aberration
	       	double arg =double(   (bfl_x*lfab_x+bfl_y*lfab_y+bfl_z*lfab_z)/(*mfield)  );

	       	*ang=acos(arg);




//	       	 Components of the total magnetic field in a frame comoving with the fluid
//	       	  but with the orientation of the observer corrected by light
//	       	  aberration. See e.g. Blandford & Konigl 1979 or Lyutikov et al. 2003, for
//	       	  the rotation of the polarization.

	       	  double bfl_obla_x = cos_elecellp*cos_azicellp*bfl_x-sin_azicellp*bfl_y-sin_elecellp*cos_azicellp*bfl_z;
	       	  double bfl_obla_y = cos_elecellp*sin_azicellp*bfl_x+cos_azicellp*bfl_y-sin_elecellp*sin_azicellp*bfl_z;
	       	  double bfl_obla_z = sin_elecellp*bfl_x+cos_elecellp*bfl_z;


//	          Angle between the projection of the total magnetic field in the plane
//	       	  x=0 in the above defined system (that is, projection in the map of
//	       	  observation -corrected by light aberration), with respect to the z-axis
//	       	  (usually also defined as a-axis in the papers).
//	       	  This is defined to follow the usual VLBI notation, with zero to the
//	       	  north, and counter-clockwise. This determines the polarization.

	       	*chih= atan2(float(bfl_obla_z),float(bfl_obla_y))-(90.0*pi/180.0);

//	       	  Rotation measure per cell's length per observing wavelength**2 in c.g.s.
//	       	  This is a quantity to be used in the integration of the transfer in the
//	       	  OBSERVER's frame, therefore we need to put all quantities in this frame.
//	       	  # Electron density: only the THERMAL population, and transformed in the
//	       	    observer's frame (it's initially defined in the fluid frame, as given
//	       	    by the RMHD code) by dividing by W.

	       	      *rmds=2.63e-17*(restd/w)*bfl_obla_x;

// This is a tester!!!!

//	if(*ang != *ang){
//	*res = *ang  ;
//	}



}











