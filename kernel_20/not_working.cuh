#include <math.h>
#include <stdio.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace std;


template <typename data_type, typename datasize_type, const int blocksizex, const int blocksizey, const int fake_blocksizey>
__global__  void 
__launch_bounds__(256, 1)
naive2D_GEMM_Kernel_2(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
  extern __shared__ float AB_smem[];
  
  int threadIDX = threadIdx.x % blocksizex;
  int threadIDY = (threadIdx.x / blocksizex);
  int threadIDY2 = (threadIdx.x / blocksizex) % 4;
  int thread_fake_id = (threadIDX % 8)*4;
  int trip_number = threadIDX / 8;
  
  int thread_fake_id_2 = threadIDX >= 16 ? threadIDX - 16 : threadIDX ;
  int j_fake = threadIDX >= 16 ? 1 : 0;
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
         //AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)] = d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows];  	  
         //_cuda::memcpy_async __pipeline_memcpy_async
          __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id + (32 * (threadIDY2 + trip_number * fake_blocksizey)) + (i * A_SMEM_BATCHSIZE)])[0])
  ,  &(reinterpret_cast<float4 *>(&d_A[gidx + (threadIDY2 + (trip_number * fake_blocksizey)) * Arows])[0]) , sizeof(float4)); 
     
     
      }
            

      

      #pragma unroll 
      for (int j = 0; j < TY; j+=2) {
          
	 int gidy = (blockIdx.y * blocksizey * TY) + threadIDY2 + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
         __pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + (threadIDY2 * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2], sizeof(float));

         
          
     
     //    gidy = (blockIdx.y * blocksizey * TY) + threadIDY2 + 4 + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
       // __pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + ((threadIDY2 + 4) * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2], sizeof(float));
      }

      
    __syncwarp();
      
      #pragma unroll 
      for (int j = 0; j < TY; j+=2) {
          
	 int gidy = (blockIdx.y * blocksizey * TY) + threadIDY2 + 4 + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
         __pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + ((threadIDY2 + 4) * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2], sizeof(float));


      } 
      

  }
   
  __syncwarp();

 //__syncthreads();
  

  for (int k = B_BATCHSIZE; k < AcolsBrows; k+=B_BATCHSIZE)  {          
       
      	  
      __syncthreads();
      
      if (!prod_consum) { 
         
      #pragma unroll	  
      for (int i = 0; i < TX; i++) {

         int gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex); 
         //AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)] = d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows];  	  
	 __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[COPY_OFFSET + thread_fake_id + (32 * (threadIDY2 + trip_number * fake_blocksizey)) + (i * A_SMEM_BATCHSIZE)])[0])
 ,  &(reinterpret_cast<float4 *>(&d_A[gidx + (k + threadIDY2 + (trip_number * fake_blocksizey)) * Arows])[0]) , sizeof(float4)); 
      }
         
      #pragma unroll 
      for (int j = 0; j < TY; j+=2) {
 
	 int gidy = (blockIdx.y * blocksizey * TY) + threadIDY2 + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
         __pipeline_memcpy_async(&AB_smem[COPY_OFFSET + A_SMEM_SIZE + thread_fake_id_2 + (threadIDY2 * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
	  
        
        gidy = (blockIdx.y * blocksizey * TY) + (threadIDY2 + 4) + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
        __pipeline_memcpy_async(&AB_smem[COPY_OFFSET + A_SMEM_SIZE + thread_fake_id_2 + ((threadIDY2 + 4) * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
	  
      
      }
      
       
     /* 
      #pragma unroll 
      for (int j = 0; j < TY; j+=2) {
           
	 int gidy = (blockIdx.y * blocksizey * TY) + (threadIDY2 + 4) + ((j + j_fake) * blocksizey);
         //AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
         //__pipeline_memcpy_async(&AB_smem[A_SMEM_SIZE + thread_fake_id_2 + threadIDY * B_BATCHSIZE + ((j + j_fake) * B_SMEM_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
        __pipeline_memcpy_async(&AB_smem[COPY_OFFSET + A_SMEM_SIZE + thread_fake_id_2 + ((threadIDY2 + 4) * B_BATCHSIZE * TY) + ((j + j_fake) * B_BATCHSIZE)], &d_B[gidy * AcolsBrows + thread_fake_id_2 + k], sizeof(float));
      }
    */      
              
    } 
   
     


    #pragma unroll 
    for (int m = 0; m < B_BATCHSIZE; m++) {
         #pragma unroll
         for (int j = 0; j < TY; j++) {
            Reg_B[j] = AB_smem[COMPUTE_OFFSET + A_SMEM_SIZE + threadIDY * B_BATCHSIZE + m + (B_SMEM_BATCHSIZE * j)]; 
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
    
        
    
    
    /*
      if (!(k&(2-1))) {	  
         __syncthreads();
         //__syncwarp();

      }
       */
  }
  
 __syncthreads();
  
   #pragma unroll 
   for (int m = 0; m < B_BATCHSIZE; m++) {
        #pragma unroll
        for (int j = 0; j < TY; j++) {
           Reg_B[j] = AB_smem[COMPUTE_OFFSET +  A_SMEM_SIZE + threadIDY * B_BATCHSIZE + m + (B_SMEM_BATCHSIZE * j)]; 
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
/*
  #pragma unroll
  for (int i = 0; i < TX; i++) {
      int gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
      #pragma unroll
      for (int j = 0; j < TY; j++) {
         int gidy = (blockIdx.y * blocksizey * TY) + threadIDY + (j * blocksizey);
         int ind = i * TY + j;
         int gid = (Arows * gidy) + gidx;
         __syncwarp();
         //d_C[gid] = sum[ind];	
         d_C[gid] = alpha * sum[ind] + beta * d_C[gid];	
      } 
  }
  */
    
  #pragma unroll
  for (int i = 0; i < TX; i++) {
     // int gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
      int gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex); 
      #pragma unroll
      for (int j = 0; j < TY; j+=8) {
	 /*
         int gidy = (blockIdx.y * blocksizey * TY) + threadIDY + (j * blocksizey);
         int gid = (Arows * gidy) + gidx;
	 AB_smem[COPY_OFFSET + threadIDY*256 + 0 * 32 + threadIDX] = d_C[gid];
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j +1) * blocksizey);
         gid = (Arows * gidy) + gidx;
	 AB_smem[COPY_OFFSET + threadIDY*256 + 1 * 32 + threadIDX] = d_C[gid];
	 gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + 2) * blocksizey);
         gid = (Arows * gidy) + gidx;
	 AB_smem[ COPY_OFFSET + threadIDY*256 + 2 * 32 + threadIDX] = d_C[gid];
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + 3) * blocksizey);
         gid = (Arows * gidy) + gidx;
         AB_smem[COPY_OFFSET + threadIDY*256 + 3 * 32  + threadIDX] = d_C[gid];
	
	  
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + 4) * blocksizey);
         gid = (Arows * gidy) + gidx;
	 AB_smem[COPY_OFFSET + threadIDY*256 + 4 * 32 + threadIDX] = d_C[gid];
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j +5) * blocksizey);
         gid = (Arows * gidy) + gidx;
	 AB_smem[COPY_OFFSET + threadIDY*256 + 5 * 32 + threadIDX] = d_C[gid];
	 gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + 6) * blocksizey);
         gid = (Arows * gidy) + gidx;
	 AB_smem[ COPY_OFFSET + threadIDY*256 + 6 * 32 + threadIDX] = d_C[gid];
         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + 7) * blocksizey);
         gid = (Arows * gidy) + gidx;
         AB_smem[COPY_OFFSET + threadIDY*256 + 7 * 32  + threadIDX] = d_C[gid];
	 */

          int gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + trip_number) * blocksizey);
   //       gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex);
	  int gid = (Arows * gidy) + gidx;
        //__pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id + COPY_OFFSET + threadIDY*256 + trip_number * 32 ])[0])
 // ,  &(reinterpret_cast<float4 *>(&d_C[gid])[0]) , sizeof(float4)); 
      
    __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id + COPY_OFFSET + threadIDY*256 + trip_number * 32])[0]),  
     &(reinterpret_cast<float4 *>(&d_C[Arows * gidy + thread_fake_id + (i * blocksizex) +  (blockIdx.x * blocksizex * TX)])[0]) , sizeof(float4)); 

         gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + trip_number + 4) * blocksizey);
  //      gidx = (blockIdx.x * blocksizex * TX) + thread_fake_id + (i * blocksizex); 

	 gid = (Arows * gidy) + gidx;

//      __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id  + COPY_OFFSET + threadIDY*256 + (trip_number + 4) * 32 ])[0])
//   ,  &(reinterpret_cast<float4 *>(&d_C[gid])[0]) , sizeof(float4)); 
       __pipeline_memcpy_async( &(reinterpret_cast<float4 *>(&AB_smem[thread_fake_id  + COPY_OFFSET + threadIDY*256 + (trip_number + 4) * 32 ])[0])
   ,  &(reinterpret_cast<float4 *>(&d_C[Arows * gidy + thread_fake_id + (i * blocksizex) +  (blockIdx.x * blocksizex * TX) ])[0]) , sizeof(float4)); 
 
	  __syncwarp();
         
	 //d_C[gid] = sum[ind];
	 /*
         float Reg_C[4];  	
	 #pragma unroll 
	 for (int k = 0; k < 4; k++) {
	     
             gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + k) * blocksizey);
             gid = (Arows * gidy) + gidx;
             Reg_C[k] = AB_smem [ COPY_OFFSET + threadIDY*128 +  k * 32  + threadIDX];	
	 }
	 */
         #pragma unroll
	 for (int k = 0; k < 8; k++) {
             gidx = (blockIdx.x * blocksizex * TX) + threadIDX + (i * blocksizex); 
             gidy = (blockIdx.y * blocksizey * TY) + threadIDY + ((j + k) * blocksizey);
             gid = (Arows * gidy) + gidx;
	    int ind = i * TY + (j+k);
           d_C[gid] = alpha * sum[ind] + beta *  AB_smem [ COPY_OFFSET + threadIDY*256 +  k * 32  + threadIDX];
     	         
          // d_C[gid] = alpha * sum[ind] + beta *  Reg_C[k];
	 }

      } 
  }
  
 
 
} 
