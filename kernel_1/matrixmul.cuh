#include <math.h>
#include <stdio.h>

using namespace std;

template<typename data_type, typename datasize_type>
__global__  void naive_GEMM_Kernel_1(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   if (gid < (Arows * Bcols)) { 
        int indexW = gid % Arows;
	int index = gid / Arows;
	int indexRefB = index * AcolsBrows;
    	data_type sum = 0.0;
	for (int i = 0; i < AcolsBrows; i++)  {
	   sum+=(d_A[indexW + i * Arows] * d_B[i+indexRefB]);
       	}
        d_C[gid] = alpha * sum + beta * d_C[gid];	
	 
   }

}
