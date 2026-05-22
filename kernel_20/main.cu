#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define TX 8
#define TY 16
#define BLOCKSIZEX 32
#define BLOCKSIZEY 8
#define NUM_OF_THREADS_IN_BLOCK BLOCKSIZEY * BLOCKSIZEX
#define NUM_OF_WARP_IN_A_BLOCK NUM_OF_THREADS_IN_BLOCK/32
#define NUM_OF_TRIPS_FOR_A  32/NUM_OF_WARP_IN_A_BLOCK
#define NUM_OF_RESULT_PER_THREAD TX*TY
#define B_BATCHSIZE 16
#define A_BATCHSIZE 16
#define B_SMEM_BATCHSIZE BLOCKSIZEY * B_BATCHSIZE 
#define A_SMEM_BATCHSIZE BLOCKSIZEX * A_BATCHSIZE 
#define A_SMEM_SIZE BLOCKSIZEX * A_BATCHSIZE * TX
#define B_SMEM_SIZE BLOCKSIZEY * B_BATCHSIZE * TY

#define AB_SMEM_SIZE A_SMEM_SIZE + B_SMEM_SIZE



#define c(x) #x
#define stringify(x) c(x)

#define t(s1,s2) s1##s2
#define tg(s1,s2) t(s1,s2)

#define tgg(s1,s2,s3) tg(tg(s1,s2),s3)
#define sum(s2,s3)  tgg(sum,s2,s3)  

#define tggg(s1,s2,s3,s4) tg(tgg(s1,s2,s3),s4)



using namespace std;

using data_type = float;
using datasize_type = int;



#include "matrixmul.cuh"


inline
cudaError_t checkCudaErrors(cudaError_t result, string functioncall = "")
{
//#if defined(DEBUG) || defined(_DEBUG)
  //fprintf(stderr, "CUDA Runtime Error: %d\n", result);
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error for this function call ( %s ) : %s\n", 
            functioncall.c_str(), cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}



int
main( int argc, char* argv[ ] )
{
  //srand(time(0));
//  fprintf (stderr, "Amount of data transfered to the device is %lld GB\n", bytes4euc/1000000000);
     
   fprintf (stderr, "Amount of A transfered to the device is %d Bytes\n", NUM_OF_TRIPS_FOR_A);
  
   const datasize_type Arows = 4096; // 16384; // 4096;// 30336;  // number of A rows
   const datasize_type Bcols = 4096; // 16384; // 4096; //30336;  // number of B columns
   const datasize_type AcolsBrows = 4096; // 16384; //4096; // 10000;  // number of A columns and B rows
   //const datasize_type ldArows = 30336; // leading dimension for A is number of rows for A
   //const datasize_type ldBrows = 10000; // leading dimension for B is number of rows for B
   //const datasize_type ldCrows = 30336; // leading dimension for C is number of roes for C 
    
    
   datasize_type Asize = sizeof(data_type) * Arows * AcolsBrows;
   datasize_type Bsize = sizeof(data_type) * AcolsBrows * Bcols;
   datasize_type Csize = sizeof(data_type) * Arows * Bcols;
    
   data_type *d_A = nullptr;
   data_type *d_B = nullptr;
   data_type *d_C = nullptr;
   

   data_type *h_A = nullptr;
   data_type *h_B = nullptr;
   data_type *h_C = nullptr;
   

   cudaError_t status;
   
   // Create CUDA events
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   
   //int BLOCKSIZE;
   //int NUMBLOCKS;
  
  
   
   //const int BLOCKSIZEX = 32;
   //const int BLOCKSIZEY = 32;
   const int NUMBLOCKSX = (Arows + (TX*BLOCKSIZEX)-1)/(TX*BLOCKSIZEX);
   const int NUMBLOCKSY =  (Bcols + (TY*BLOCKSIZEY)-1)/(TY*BLOCKSIZEY);
   const int FAKE_BLOCKSIZEY = BLOCKSIZEY/2;

      
  // allocate number of threads in a block  
  //dim3 threads(BLOCKSIZEX, BLOCKSIZEY, 1 );
  
  dim3 threads(BLOCKSIZEX * BLOCKSIZEY, 1,  1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKSX, NUMBLOCKSY, 1 );


   // pinned data 
   cudaMallocHost((void**)&h_A, Asize);
   cudaMallocHost((void**)&h_B, Bsize);
   cudaMallocHost((void**)&h_C, Csize);
  
  
  for (int i = 0; i < (Arows * AcolsBrows); i ++) {
      h_A[i] = (float) (rand() % 4);
  }
  
  for (int i = 0; i < (AcolsBrows * Bcols); i ++) {
      h_B[i] =  (float) (rand() % 2);
  }

  
  for (int i = 0; i < (Arows * Bcols); i ++) {
      h_C[i] =  1.0;
  }
  const data_type alpha = 1.0;
  const data_type beta = 0.5;
  
  
   
  //allocate memory for A on the GPU device
  status = cudaMalloc( (void **)(&d_A), Asize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&d_A), Asize)");
  fprintf (stderr, "Amount of A transfered to the device is %u Bytes\n", Asize);
     
  //allocate memory for B on the GPU device
  status = cudaMalloc( (void **)(&d_B), Bsize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&d_B), Bsize)");
  fprintf (stderr, "Amount of B transfered to the device is %u Bytes\n", Bsize);
  
  //allocate memory for C on the GPU device
  status = cudaMalloc( (void **)(&d_C), Csize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&d_C), Csize)");
  fprintf (stderr, "Amount of B transfered to the device is %u Bytes\n", Csize);
 

  // copy A from host memory to the device memory:
  status = cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice)");  

  
  
  // copy B from host memory to the device memory:
  status = cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice)");  

  
  // copy C from host memory to the device memory:
  status = cudaMemcpy(d_C, h_C, Csize, cudaMemcpyHostToDevice);
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice)");  

    
  // Get a pointer to the explicit instantiation
  void* kernel_ptr = (void*)WarpSpec_GEMM_Kernel_20<data_type, datasize_type, BLOCKSIZEX, BLOCKSIZEY, FAKE_BLOCKSIZEY>;
  
  int dynSmemSize = (A_SMEM_SIZE + B_SMEM_SIZE) * 4 * 2;
  

  printf("Dynamic Memory Size: %d bytes\n", dynSmemSize);
  
    
  printf("Block size: %d\n",BLOCKSIZEX * BLOCKSIZEY);
  printf("Grid size: %d\n",NUMBLOCKSX * NUMBLOCKSY);
  
  status = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmemSize);

  checkCudaErrors( status," cudaFuncSetAttribute(euclideanMatrixDynamicSharedMemory, cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmemSize); ");  


  // Record the start event
  cudaEventRecord(start, 0); 

  // call the kernel
  WarpSpec_GEMM_Kernel_20<data_type, datasize_type, BLOCKSIZEX, BLOCKSIZEY, FAKE_BLOCKSIZEY><<< grid, threads, dynSmemSize>>>( d_A, d_B, d_C, Arows, Bcols, AcolsBrows, alpha, beta);
  
  status = cudaDeviceSynchronize( );
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
   
  // Calculate elapsed time
  float GpuTime;
  cudaEventElapsedTime(&GpuTime, start, stop); 
  printf("  GPU time: %f milliseconds\n", GpuTime); 
  
  checkCudaErrors( status,"matrixMulAddColBasedARR<<< grid, threads >>>( d_A, d_B, d_C, Arows, Bcols, AcolsBrows, alpha, beta)");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  

  
  // copy C from device memory to host 
  status = cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, "cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost)"); 
  

 
  
/*   
  for (int i = 0; i < Arows; i++) {
       for (int j = 0; j < Bcols; j++) {
	   float sum = 0.00;
	   for (int k = 0; k < AcolsBrows; k++) {
               sum+=(h_A[IDX2C(k,i,Arows)]*h_B[IDX2C(j,k,AcolsBrows)]);	
	   }
	   float sumdiff = sum + 0.5 - h_C[IDX2C(j,i,Arows)];
	   if ((powf(sumdiff, 2.0)) > powf(0.01, 2.0)) {         
	       printf(" it is not equal for this row %d and column %d \n", i, j);
	   }
       }
  }
  
  */

  printf("last value: %f \n", h_C[(Arows * Bcols)-1]);
  printf("second to last value: %f \n", h_C[(Arows * Bcols)-2]);

  // free device memory 
  cudaFree( d_A );
  cudaFree( d_B );
  cudaFree( d_C ); 
  
  cudaFreeHost( h_A );
  cudaFreeHost( h_B );
  cudaFreeHost( h_C );
   
  return 0;
};	
