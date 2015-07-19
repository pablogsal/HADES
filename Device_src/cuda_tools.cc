#include <stdio.h>
#include <math.h>

__device__ double get_data_index(int x, int y, int dim_x){

 return x + y * dim_x ;

}


