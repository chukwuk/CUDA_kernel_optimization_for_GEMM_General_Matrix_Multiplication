#include <math.h>
#include <stdio.h>

using namespace std;


#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define TX 2
#define TY 2
#define NUM_OF_RESULT_PER_THREAD TX*TY

#define c(x) #x
#define stringify(x) c(x)

#define t(s1,s2) s1##s2
#define tg(s1,s2) t(s1,s2)

#define tgg(s1,s2,s3) tg(tg(s1,s2),s3)
#define sum(s2,s3)  tgg(sum,s2,s3)  

#define tggg(s1,s2,s3,s4) tg(tgg(s1,s2,s3),s4)


using data_type = float;
using datasize_type = int;




//template <typename data_type, typename datasize_type>
__global__  void naive2D_GEMM_Kernel_2(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
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
