#include <math.h>
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace std;


template <typename data_type, typename datasize_type>
__global__  void 
__launch_bounds__(256, 1)
naive2D_GEMM_Kernel_2(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
   int gid, gidx, gidy;
   gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x;
   gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y;
   gid = (Arows * gidy) + gidx;
  extern __shared__ float AB_smem[];
  float sum[NUM_OF_RESULT_PER_THREAD];
  for (int i = 0; i < TX; i++) {
        for (int j = 0; j < TY; j++) {
            int ele = i * TY + j;
            sum[ele] = 0.0;
	}
  }
  for (int k = 0; k < AcolsBrows; k+=B_BATCHSIZE)  {          

      #pragma unroll	  
      for (int i = 0; i < TX; i++) {
	 #pragma unroll 
	 for (int j = 0; j < NUM_OF_TRIPS_FOR_A; j++) {
           gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x + (i*blockDim.x); 
           //AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)] = d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows];  	  
           __pipeline_memcpy_async(&AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)], &d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows], sizeof(float)); 
	 }       	     
      }	            
      __syncthreads();
            
     float Reg_A[TX * A_BATCHSIZE];
     #pragma unroll 
      for (int m = 0; m < B_BATCHSIZE; m++) {
          #pragma unroll
	  for (int i = 0; i < TX; i++) {
	      int ind = i * B_BATCHSIZE + m;
	      Reg_A[ind] = AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)];  
          }
      } 


      #pragma unroll 
      for (int j = 0; j < TY; j++) {
	 gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y + (j*blockDim.y);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         __pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + threadIdx.x + k], sizeof(float));

      }
      __syncwarp();
       
      
      float Reg_B;
      #pragma unroll
      for (int m = 0; m < B_BATCHSIZE; m++) {
         #pragma unroll
          for (int j = 0; j < TY; j++) {
	     Reg_B = AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]; 
	     #pragma unroll
              for (int i = 0; i < TX; i++) {
		  int index = i * B_BATCHSIZE + m;
		  int ele = i * TY + j;
                  sum[ele]+=(Reg_A[index] * Reg_B);  
	      }
	  }
       }
       
      __syncthreads();                  

  }
  for (int i = 0; i < TX; i++) {
      gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x + (i*blockDim.x); 
      for (int j = 0; j < TY; j++) {
         gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y + (j*blockDim.y);
         int index = i * TY + j;
         gid = (Arows * gidy) + gidx; 
         d_C[gid] = alpha * sum[index] + beta * d_C[gid];	
      
      } 
  }
    

} 



/*
template <typename data_type, typename datasize_type>
__global__  void naive2D_GEMM_Kernel_2(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
   int gid, gidx, gidy;
   gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x;
   gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y;
   gid = (Arows * gidy) + gidx; 

  __shared__ float A_smem[32][32];
  __shared__ float B_smem[32][32];
  float sum = 0.0;
  for (int k = 0; k < AcolsBrows; k+=B_BATCHSIZE)  {
     
      A_smem[threadIdx.y][threadIdx.x] = d_A[gidx + (k + threadIdx.y) * Arows];  
      __syncthreads();            
      B_smem[threadIdx.y][threadIdx.x] = d_B[gidy * AcolsBrows + threadIdx.x + k];
      __syncwarp();      
	  
      for (int i = 0; i < B_BATCHSIZE; i++) {
          //sum+=(A_smem[threadIdx.x + i * A_BATCHSIZE] * B_smem[i + threadIdx.y * B_BATCHSIZE]);
           
          sum+=(A_smem[i][threadIdx.x] * B_smem[threadIdx.y][i]);
      }
      __syncthreads();            
  }
  
  d_C[gid] = alpha * sum + beta * d_C[gid];	

}
*/

/*
   int gid, gidx, gidy;
   int indexRefB; 
   //data_type sum;
   // unrolling loop with pragma lead to slower FLOPS 
   for (int i = 0; i < TX; i++) {
       gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x + (i*blockDim.x); 
       for (int j = 0; j < TY; j++) {
           gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y + (j*blockDim.y);
           gid = (Arows * gidy) + gidx; 
           if (gidx < Arows && gidy < Bcols) { 
               indexRefB = gidy * AcolsBrows;
    	       float sum(i,j) = 0.0;
	       for (int k = 0; k < AcolsBrows; k++)  {
	           sum(i,j)+=(d_A[gidx + k * Arows] * d_B[k+indexRefB]);
               }
               d_C[gid] = alpha * sum(i,j) + beta * d_C[gid];	
           }

       }
   
   }
*/
