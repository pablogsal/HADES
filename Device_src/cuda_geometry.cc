

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


__device__ void x_limits(float *x_min, float *x_max,int dim_x, int dim_y, float theta,float y_cell,
float z_cell)
{


    // Renaming things from input_par to easy reading
    // Notice that for the RMHD plane we are using the variable names:
    //
    // x= rho
    // y= z
float pi= 3.14159265358979323846264338328;

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

if(y_cell > y_min && y_cell < y_max){

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

    float pi= 3.14159265358979323846264338328;
    // The x,y,z in lab frame (a rotation of x,y,z)

    float x_lab= cos(theta) * x - sin(theta) * y;
    float y_lab= cos(theta) * y + sin(theta) * x;
    float z_lab = z;

     // The rho and z coordinates

    *rho=round(sqrt(x_lab * x_lab + z_lab * z_lab ));
    *zeta= round(y_lab);

    //Beware of rho=-1 and rho > dim_x!!

    if(*rho<0){
    	*rho=0;
    }
    if(*rho>=dim_x)
    {
    	*rho=159;
    }


}











