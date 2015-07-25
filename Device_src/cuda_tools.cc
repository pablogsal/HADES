#include <stdio.h>
#include <math.h>
#include <cuda.h>


__device__ int get_data_index(int x, int y, int dim_x){

 return x + y * dim_x ;

}


