
// Declaring the Kernel function. 

__global__ void MatrixMulKernel(double *density,double *eps,double *velx,double *vely,double *velz,
                               double *bx,double *by,double *bz,
							   double *jet_limits ,double *a, double *b, double *c)
{
    
     
     
     //Getting the matrix size as dictionary substitution when calling the Kernel
	
	 // Getting the observer image dimensions
     
     int wA = %(MATRIX_SIZE)s ;
     int wB = %(MATRIX_SIZE2)s ;
	 
	 // Getting the observer image min max limits in y and z direction
	 
     int image_y_min = %(Y_MIN)s ;
	 int image_y_max = %(Y_MAX)s ;
	 int image_z_min = %(Z_MIN)s ;
	 int image_z_max = %(Z_MAX)s ;
	 
	 
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
     
	 
         
     // If we are out of range, we do nothing. This is important if we
     // want more threads than needed.    
          
     if ( y < wA and x < wB ) {

		 // Obtain z_cell and y_cell coordinate. Notice that this is because the x and y indexes are thread index,
		 // so its values only goes from 0 to wA/wB. As we need the actual value of the z/y coordinate, we must convert
		 // these values. This is easy as we only need to sum the actual value to the minimum physical value.

          int y_cell=image_y_min+x;
          int z_cell=image_z_min+y;
		  
		  
		  //Test stuff (to test if the matrix are being readed)
		  
          double Aelement = velx[13];
          double Belement = b[offset];
    

          // Initialize x_min and x_max for this cell.

          float x_min = 0.0 ;
          float x_max = 0.0 ;


		  // Calculate x_min and x_max for this cell. Notice that the function gets a pointer to the
		  // x_min and x_max values and updates them. 

          x_limits( &x_min, &x_max,dim_x, dim_y, theta, y_cell ,z_cell);

			

		  
	      // Initialize rho and zeta values. These are the values of the point x,y,z in the RMHD grid.
		
			
      	  float rho = 0.0 ;
      	  float zeta = 0.0 ;




         float sum=0;
		
          if( x_max > x_min){

//        	  float *restd = new float[int(x_max-x_min)+3];
              float eldeng[3000];
              float emini[3000];

        	  //Initialize the number of cells accumulator.

                  int cell=0;


                for ( int x_cell = round(x_min)+1; x_cell < round(x_max); x_cell++ ) {


                      //cont=jet_limits[int(y_cell)];
           			  //Beware of z being 600 because jet_limits goes from 0 to 599!!!!

       		  		obs_to_rmhd( &rho, &zeta, x_cell, y_cell, z_cell, theta,dim_x,dim_y);

           		 	if(round(rho)<jet_limits[int(round(zeta))]){
				
           		 				int index_for_rmhd = get_data_index(zeta, rho, dim_y);


           		 			//index_for_rmhd can contain outside values -> posible bug.
//            			    energy(cell,&(restd[cell]),&(eldeng[cell]),&(emini[cell]),  density[ index_for_rmhd ] , eps[ index_for_rmhd ]);

            				sum = sum + density[index_for_rmhd];

//            				b[offset] = restd[cell];

            				cell++;



			
			
           			 } //End of ?inside tracer value

                } // En of x_cell loop


           } // End of ?x_max > x_min
		
        else{

       	 	rho=0;
       	 	zeta=0;

       	 b[offset] = 1.111;

        }
		
		

		//Continue after x cell loop

// PRUEBA DE OVERFLOW!!
//		  float test[10];
//		  float test5[6000];
//     	  float num;
//		  for ( int i = 0; i < 10; i++ ) {
//			  num=pow(  float(2.0)  ,  float(0.345)  );
//			  test[i]= num;
//
//
//
//		  }
// PRUEBA DE QUE EL JET SALE BIEN EN LA IMAGEN!!
//          int yy= y * blockDim.x * gridDim.x;
//
//          if(y<dim_x){
//
//        	  yy= (dim_x-y-1)* blockDim.x * gridDim.x;
//          }
//          else{
//        	  yy= (y-dim_x)* blockDim.x * gridDim.x;
//          }
//
//
//             c[offset] =density[ x + yy];
//
          c[offset] = sum;

		  
      }
     else{ //What to do if we are out of thread.
		 
    	 c[offset] =-12;
          // This printf is only for fun
   
          // printf("I am out of thread (%% i,%% i) \\n ",x,y);
      }
}
