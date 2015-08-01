#include <stdio.h>
#include <math.h>
#include <cuda.h>


__device__ int get_data_index(int x, int y, int dim_x){

 return x + y * dim_x ;

}

__device__ float over_flow(){
// PRUEBA DE OVERFLOW!!
		  float test[ %(MAX_CELLS)s ];
		  float test2[ %(MAX_CELLS)s ];
     	  float num;
		  for ( int i = 0; i < %(MAX_CELLS)s ; i++ ) {
			  num=pow(  float(2.0)  ,  float(0.345)  );
			  test[i]= num;
			  test2[i]= 2*test[i];



		  }
		  num=0;
		  for ( int i = 0; i < %(MAX_CELLS)s ; i++ ) {
			  num= i;



		  }

		  return num;
}
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
