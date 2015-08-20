
// Declaring the Kernel function. 

__global__ void MatrixMulKernel(float *density,float *eps,float *velx,float *vely,float *velz,
                               float *bx,float *by,float *bz,
							   int *jet_limits , float *besselx,float *besself1,float *besself2,float *besselg1,float *besselg2,int *error_test,double *sia_out, double *sib_out,double *suab_out,double *tau_out,double *rmea_out,double *botf_out)
{
    
     
     
     //Getting the matrix size as dictionary substitution when calling the Kernel
	
	 // Getting the observer image dimensions
     
     int wA = %(MATRIX_SIZE)s ;
     int wB = %(MATRIX_SIZE2)s ;
	 
	 // Getting the observer image min max limits in y and z direction
	 
     int image_y_min = %(Y_MIN)s ;
//	 int image_y_max = %(Y_MAX)s ; Not actually necessary
	 int image_z_min = %(Z_MIN)s ;
//	 int image_z_max = %(Z_MAX)s ; Not actually necessary
	 
	 
	 // Getting the dimensions of the RMHD data
	 
     int dim_x= %(DIM_X)s;
     int dim_y= %(DIM_Y)s;
	 
	 // Getting the theta angle
	 
	 float theta = %(ANGLE)s;

     
     //Constructing the indexes. The coodinates x and y
     //are the number of threads at the left and above
     //the current thread.
     //
     //The offset value is then the unique index
     //constructed using the x and y values.
     
     int x = threadIdx.x + blockIdx.x * blockDim.x;
     int y = threadIdx.y + blockIdx.y * blockDim.y;
     int offset = x + y * blockDim.x * gridDim.x ;

//     double flag=0;

     // Initialize the matrix values to get out

	double sia =0.0;
	double sib =0.0;
	double suab=0.0;
	double tau=0.0;
	double rmea=0.0;
	double botf=0.0;

         
     // If we are out of range, we do nothing. This is important if we
     // want more threads than needed.    
          
     if ( y < wA and x < wB ) {

		 // Obtain z_cell and y_cell coordinate. Notice that this is because the x and y indexes are thread index,
		 // so its values only goes from 0 to wA/wB. As we need the actual value of the z/y coordinate, we must convert
		 // these values. This is easy as we only need to sum the actual value to the minimum physical value.

          int y_cell=image_y_min+x;
          int z_cell=image_z_min+y;
		  

    

          // Initialize x_min and x_max for this cell.

          float x_min = 0.0 ;
          float x_max = 0.0 ;


		  // Calculate x_min and x_max for this cell. Notice that the function gets a pointer to the
		  // x_min and x_max values and updates them. dim_y here is the horizontal dimension of the
          // observer map and dim_z is the vertical. As we need the horizontal, we pass dim_y.

          x_limits( &x_min, &x_max,dim_x, dim_y, theta, y_cell ,z_cell);



		  
	      // Initialize rho and zeta values. These are the values of the point x,y,z in the RMHD grid.
		
			
      	  float rho = 0.0 ;
      	  float zeta = 0.0 ;
		
          // If we have enought cells to integrate:

          if( x_max > x_min){


        	 double restd[ %(MAX_CELLS)s ];
        	 double eldeng[ %(MAX_CELLS)s ];
        	 double emini[ %(MAX_CELLS)s ];
        	 double deltao[ %(MAX_CELLS)s ];
        	 double mfield[ %(MAX_CELLS)s ];
        	 double ang[ %(MAX_CELLS)s ];
        	 double chih[ %(MAX_CELLS)s ];
        	 double rmds[ %(MAX_CELLS)s ];


        	  //Initialize the number of cells accumulator.

                  int cell=0;

                // Main for loop in the x cells. The +1 in the x_min is because we start in 0. POSIBLE BUG!
                for ( int x_cell = round(x_min)+1; x_cell < round(x_max); x_cell++ ) {


                    //cont=jet_limits[int(y_cell)];
           			//Beware of z being 600 because jet_limits goes from 0 to 599!!!! POSIBLE BUG!
                	// This function is for obatining the rho and zeta coordinates (dim_x and dim_y respectively).

       		  		obs_to_rmhd( &rho, &zeta, x_cell, y_cell, z_cell, theta,dim_x,dim_y);


       		  		//If we are inside the jet:
           		 	if(round(rho)<jet_limits[int(round(zeta))]){
				



           		 			int index_for_rmhd = get_data_index(zeta, rho, dim_y);

           		 			double res;

           		 			energy(cell,&(restd[cell]),&(eldeng[cell]),&(emini[cell]),  density[ index_for_rmhd ] , eps[ index_for_rmhd ],&res);




            			    // Call lorentz factor function

            			    lorenz_calculator( cell, x_cell, y_cell, z_cell,theta,  velx[ index_for_rmhd ],  velz[ index_for_rmhd ],  vely[ index_for_rmhd ]
            			                      ,  bx[ index_for_rmhd ],  bz[ index_for_rmhd ],  by[ index_for_rmhd ],&res, &(deltao[cell]), &(mfield[cell])
            			                      ,  &(ang[cell]), &(chih[cell]) ,&(rmds[cell]), restd[cell] );

            			    cell++;


            			   //Control stuff
//            				flag=res;

			
           			 } //End of ?inside tracer value


                } // En of x_cell loop

                //Continue after x cell loop

                if( cell > 1){

                // Must integrate transfer equations

                double res=0;

                 difsum( cell,  restd,  eldeng,  emini,  deltao,  mfield,  ang,  chih,  rmds,&res,besselx,besself1,besself2,besselg1,besselg2,error_test, &sia, &sib,&suab,&tau,&rmea,&botf);



//				  Control stuff to debugging
//                flag=sia+sib;

                }





           } // End of ?x_max > x_min
		
        else{ // If not x_max > x_min

       	 	rho=0;
       	 	zeta=0;

        } //Continue outside x_max > x_min
		

          //Get the values into the output arrays

          sia_out[offset] = sia;
          sib_out[offset] = sib;
          suab_out[offset] = suab;
          tau_out[offset] = tau;
          rmea_out[offset] = rmea;
          botf_out[offset] = botf;





      }
     else{ //What to do if we are out of thread.
		 

          // This printf is only for fun
   
//           printf("I am out of thread (%% i,%% i) \\n ",x,y);
      }
}
