#include <math.h>
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace std;


template <typename data_type, typename datasize_type, const int blocksizex, const int blocksizey, const int fake_blocksizey>
__global__  void 
__launch_bounds__(256, 1)
WarpSpec_GEMM_Kernel_20(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
  extern __shared__ float AB_smem[];
  
  int threadIDX = threadIdx.x % blocksizex;
  int threadIDY = (threadIdx.x / blocksizex);
  int threadIDY2 = (threadIdx.x / blocksizex) % 4;
  int thread_fake_id = (threadIDX % 8)*4;
  int trip_number = threadIDX / 8;
  
  int thread_fake_id_2 = (threadIDX % 8) * 2; 
  bool prod_consum = threadIdx.x < 128 ? true : false;
  
  float sum[NUM_OF_RESULT_PER_THREAD];
  float Reg_A[TX];
  float Reg_B[TY];   
   
  for (int i = 0; i < NUM_OF_RESULT_PER_THREAD; i++) {
        sum[i] = 0.0;
  }
  
  int COPY_OFFSET = AB_SMEM_SIZE;
  int COMPUTE_OFFSET = 0;
  

  if (!prod_consum) {
           
      #pragma unroll	  
      for (int i = 0; i < TX; i++) {
          int gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex); 
          __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id + (32 * (threadIDY2 + trip_number * fake_blocksizey)) + (i * A_SMEM_BATCHSIZE)])[0])
  ,  &(reinterpret_cast<float4 *>(&d_A[gidx + (threadIDY2 + (trip_number * fake_blocksizey)) * Arows])[0]) , sizeof(float4)); 
     
      }

      #pragma unroll 
      for (int j = 0; j < TY; j+=4) {
          
	 int gidy = (blockIdx.y * blocksizey * TY) + threadIDY2 + ((j + trip_number) * blocksizey);
         __pipeline_memcpy_async( &(reinterpret_cast<float2 *> (&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + (threadIDY2 * B_BATCHSIZE * TY) + ((j + trip_number) * B_BATCHSIZE)])[0]),  &(reinterpret_cast<float2 *>   ( &d_B[gidy * AcolsBrows + thread_fake_id_2])[0]), sizeof(float2));    
     
      }
      
      #pragma unroll 
      for (int j = 0; j < TY; j+=4) {
             
	 int gidy = (blockIdx.y * blocksizey * TY) + (threadIDY2 + 4 ) + ((j + trip_number) * blocksizey);  
         __pipeline_memcpy_async( &(reinterpret_cast<float2 *> (&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + ((threadIDY2 + 4) * B_BATCHSIZE * TY) + ((j + trip_number) * B_BATCHSIZE)])[0]),  &(reinterpret_cast<float2 *>   ( &d_B[gidy * AcolsBrows + thread_fake_id_2])[0]), sizeof(float2));

      } 
      

  } 

  for (int k = B_BATCHSIZE; k < AcolsBrows; k+=B_BATCHSIZE)  {          
             	  
      __syncthreads();
      
      if (!prod_consum) { 
         
      #pragma unroll	  
      for (int i = 0; i < TX; i++) {

         int gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex); 
	 __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[COPY_OFFSET + thread_fake_id + (32 * (threadIDY2 + trip_number * fake_blocksizey)) + (i * A_SMEM_BATCHSIZE)])[0])
 ,  &(reinterpret_cast<float4 *>(&d_A[gidx + (k + threadIDY2 + (trip_number * fake_blocksizey)) * Arows])[0]) , sizeof(float4)); 
      }
         
      #pragma unroll 
      for (int j = 0; j < TY; j+=4) {

	 int gidy = (blockIdx.y * blocksizey * TY) + threadIDY2 + ((j + trip_number) * blocksizey);
         __pipeline_memcpy_async( &(reinterpret_cast<float2 *> (&AB_smem[COPY_OFFSET + A_SMEM_SIZE + thread_fake_id_2 + (threadIDY2 * B_BATCHSIZE * TY) + ((j + trip_number) * B_BATCHSIZE)])[0]),  &(reinterpret_cast<float2 *>   ( &d_B[gidy * AcolsBrows + thread_fake_id_2 + k])[0]), sizeof(float2));
          	
	 gidy = (blockIdx.y * blocksizey * TY) + (threadIDY2 + 4) + ((j + trip_number) * blocksizey);
         __pipeline_memcpy_async( &(reinterpret_cast<float2 *> (&AB_smem[COPY_OFFSET + A_SMEM_SIZE + thread_fake_id_2 + ((threadIDY2 + 4) * B_BATCHSIZE * TY) + ((j + trip_number) * B_BATCHSIZE)])[0]),  &(reinterpret_cast<float2 *>   ( &d_B[gidy * AcolsBrows + thread_fake_id_2 + k])[0]), sizeof(float2));
      
      } 
              
    } 

    #pragma unroll 
    for (int m = 0; m < B_BATCHSIZE; m++) {
         #pragma unroll
         for (int j = 0; j < TY; j++) {
            Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + j * B_BATCHSIZE + m + ( threadIDY * B_BATCHSIZE * TY )]; 
         }
          
         #pragma unroll
	for (int i = 0; i < TX; i++) {
 	    Reg_A[i] = AB_smem[COMPUTE_OFFSET + threadIDX + m * 32 + (i * A_SMEM_BATCHSIZE)];  
	}
	   
        #pragma unroll
	for (int j = 0; j < TY; j++) {
            #pragma unroll 
            for (int i = 0; i < TX; i++) {                   
		int ind = i * TY + j;
                  sum[ind] = Reg_A[i] * Reg_B[j] + sum[ind];  

	    }

	}	 
       
    }   
    
    COPY_OFFSET = (COPY_OFFSET != 0) ? 0 : AB_SMEM_SIZE;
    COMPUTE_OFFSET = (COMPUTE_OFFSET != 0) ? 0 : AB_SMEM_SIZE;    
  }
  
 __syncthreads();
  
   #pragma unroll 
   for (int m = 0; m < B_BATCHSIZE; m++) {
        #pragma unroll
        for (int j = 0; j < TY; j++) {
            Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + j * B_BATCHSIZE + m + ( threadIDY * B_BATCHSIZE * TY )];
       	}
          
        #pragma unroll
        for (int i = 0; i < TX; i++) {
 	     Reg_A[i] = AB_smem[COMPUTE_OFFSET + threadIDX + m * 32 + (i * A_SMEM_BATCHSIZE)];  
	}
	   
        #pragma unroll
	for (int j = 0; j < TY; j++) {
            #pragma unroll 
            for (int i = 0; i < TX; i++) {                   
		int ind = i * TY + j;
                sum[ind] = Reg_A[i] * Reg_B[j] + sum[ind];  

	    }

	}	 
       
   }   

  #pragma unroll
  for (int i = 0; i < TX; i++) {
      int gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex); 
      #pragma unroll
      for (int j = 0; j < TY; j+=8) {

          int gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + trip_number) * blocksizey);
	  int gid = (Arows * gidy) + gidx; 
    __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id + COPY_OFFSET + threadIDY*256 + trip_number * 32])[0]),  
     &(reinterpret_cast<float4 *>(&d_C[Arows * gidy + thread_fake_id + (i * blocksizex) +  (blockIdx.x * blocksizex * TX)])[0]) , sizeof(float4)); 
         
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + trip_number + 4) * blocksizey);
	 gid = (Arows * gidy) + gidx;

       __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id  + COPY_OFFSET + threadIDY*256 + (trip_number + 4) * 32 ])[0])
   ,  &(reinterpret_cast<float4 *>(&d_C[Arows * gidy + thread_fake_id + (i * blocksizex) +  (blockIdx.x * blocksizex * TX) ])[0]) , sizeof(float4)); 
 
	  __syncwarp();
         
         #pragma unroll
	 for (int k = 0; k < 8; k++) {
             gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
             gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + k) * blocksizey);
             gid = (Arows * gidy) + gidx;
	    int ind = i * TY + (j+k);
           d_C[gid] = alpha * sum[ind] + beta *  AB_smem [ COPY_OFFSET + threadIDY*256 +  k * 32  + threadIDX];
     	         
	 }

      } 
  }
 

} 
