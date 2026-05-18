#include <math.h>
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace std;


template <typename data_type, typename datasize_type, const int blocksizex, const int blocksizey>
__global__  void 
__launch_bounds__(256, 1)
naive2D_GEMM_Kernel_2(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
   int gid, gidx, gidy;
   //gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x;
   //gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y;
   //gid = (Arows * gidy) + gidx;
  extern __shared__ float AB_smem[];
  float sum[NUM_OF_RESULT_PER_THREAD];
  float Reg_A[TX];
  float Reg_B[TY];
  //int Reg_refB[B_BATCHSIZE];
  //int Reg_refA[B_BATCHSIZE];
  int threadIDX = threadIdx.x % blocksizex;
  int threadIDY = (threadIdx.x / blocksizex) % 4;
  //int thread_fake_id = (threadIDX % 8)*4;
  //int trip_number = threadIDX / 8;
  int thread_fake_id_2 = threadIDX >= 16 ? threadIDX - 16 : threadIDX ;
  int j_fake = threadIDX >= 16 ? 1 : 0;

  
   #pragma unroll 
   for (int i = 0; i < NUM_OF_RESULT_PER_THREAD; i++) {
       sum[i] = 0.0;
   }
  const int store = A_SMEM_SIZE + threadIDY * B_BATCHSIZE;
  
  int k = 0;

  if (threadIdx.x >= 128) {
      
       
      
      #pragma unroll	  
      for (int i = 0; i < TX; i++) {
	#pragma unroll 
	 for (int j = 0; j < 4; j++) {
           gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
           //AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)] = d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows];  	  
           __pipeline_memcpy_async(&AB_smem[threadIDX + (32 * (threadIDY + j * blocksizey)) + (i * A_SMEM_BATCHSIZE)], &d_A[gidx + (k + threadIDY + (j * blocksizey)) * Arows], sizeof(float)); 
	 }       	     
      }	            
      
      
         
      #pragma unroll 
      for (int j = 0; j < TY; j+=2) {
	 
	 gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
         __pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + (threadIDY * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
	      

      }
      



  } 

  int COPY_OFFSET = AB_SMEM_SIZE;
  int COMPUTE_OFFSET = 0;
  
  
  for (k = B_BATCHSIZE; k < AcolsBrows; k+=B_BATCHSIZE)  {          

      
     // __syncwarp();
      
      __syncthreads();
      if (threadIdx.x < 128) {      
       #pragma unroll 
       for (int m = 0; m < B_BATCHSIZE; m++) {
           #pragma unroll
           for (int j = 0; j < TY; j++) {
             //Reg_B[j] = AB_smem[COMPUTE_OFFSET + store + m + (B_SMEM_BATCHSIZE * j)];
               
             //Reg_B[j] = AB_smem[COMPUTE_OFFSET + store + m + (B_SMEM_BATCHSIZE * j)];
	      
              Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + m + threadIDY * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)];
	      //Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + m + (threadIDY * B_BATCHSIZE * TY) + (B_BATCHSIZE * j)]; 
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

		
     } else {
       
      
      #pragma unroll	  
      for (int i = 0; i < TX; i++) {
	#pragma unroll 
	 for (int j = 0; j < 4; j++) {
           gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
           //AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)] = d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows];  	  
           __pipeline_memcpy_async(&AB_smem[COPY_OFFSET + threadIDX + (32 * (threadIDY + j * blocksizey)) + (i * A_SMEM_BATCHSIZE)], &d_A[gidx + (k + threadIDY + (j * blocksizey)) * Arows], sizeof(float)); 
	 }       	     
      }	            


         
      #pragma unroll 
      for (int j = 0; j < TY; j+=2) {
         
	 gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
         __pipeline_memcpy_async(&AB_smem[COPY_OFFSET + A_SMEM_SIZE + thread_fake_id_2 + (threadIDY * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
	  
	   
      }
   

    COPY_OFFSET = (COPY_OFFSET != 0) ? 0 : AB_SMEM_SIZE;
    COMPUTE_OFFSET = (COMPUTE_OFFSET != 0) ? 0 : AB_SMEM_SIZE;
      
    
    
    /*
      if (!(k&(2-1))) {	  
         __syncthreads();
         //__syncwarp();

      }
       */
  }
 // __syncwarp();
  
 
 __syncthreads();
 
 
 if (threadIdx.x < 128) {       
  
       #pragma unroll 
       for (int m = 0; m < B_BATCHSIZE; m++) {
           #pragma unroll
           for (int j = 0; j < TY; j++) {
             //Reg_B[j] = AB_smem[COMPUTE_OFFSET + store + m + (B_SMEM_BATCHSIZE * j)];
	      
              Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + m + threadIDY * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)];
	      //Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + m + (threadIDY * B_BATCHSIZE * TY) + (B_BATCHSIZE * j)]; 
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

      __syncwarp(); 

  #pragma unroll
  for (int i = 0; i < TX; i++) {
      gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
      #pragma unroll
      for (int j = 0; j < TY; j++) {
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + (j * blocksizey);
         int ind = i * TY + j;
         gid = (Arows * gidy) + gidx;
	 __syncwarp();
         //d_C[gid] = sum[ind];	
         d_C[gid] = alpha * sum[ind] + beta * d_C[gid];	
      } 
  }
  
 }
 
} 
