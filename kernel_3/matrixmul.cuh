#include <math.h>
#include <stdio.h>

using namespace std;

template <typename data_type, typename datasize_type>
__global__  void GEMM_Kernel_3_2DTiling(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
   int gid, gidx, gidy;
   int indexRefB; 
   data_type sum;
   // unrolling loop with pragma lead to slower FLOPS 
   for (int i = 0; i < TX; i++) {
       gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x + (i*blockDim.x); 
       for (int j = 0; j < TY; j++) {
           gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y + (j*blockDim.y);
           gid = (Arows * gidy) + gidx; 
           if (gidx < Arows && gidy < Bcols) { 
               indexRefB = gidy * AcolsBrows;
    	       sum = 0.0;
	       for (int k = 0; k < AcolsBrows; k++)  {
	           sum+=(d_A[gidx + k * Arows] * d_B[k+indexRefB]);
               }
               d_C[gid] = alpha * sum + beta * d_C[gid];	
           }

       }
   
   }

}

