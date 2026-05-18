#include <math.h>
#include <stdio.h>

using namespace std;


#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define TX 4
#define TY 16
#define BLOCKSIZEX 32
#define BLOCKSIZEY 8
#define NUM_OF_THREADS_IN_BLOCK BLOCKSIZEY * BLOCKSIZEX
#define NUM_OF_WARP_IN_A_BLOCK NUM_OF_THREADS_IN_BLOCK/32
#define NUM_OF_TRIPS_FOR_A  32/NUM_OF_WARP_IN_A_BLOCK
#define NUM_OF_RESULT_PER_THREAD TX*TY
#define B_BATCHSIZE 32
#define A_BATCHSIZE 32
#define B_SMEM_BATCHSIZE BLOCKSIZEY * B_BATCHSIZE 
#define A_SMEM_BATCHSIZE BLOCKSIZEX * A_BATCHSIZE 
#define A_SMEM_SIZE BLOCKSIZEX * A_BATCHSIZE * TX
#define B_SMEM_SIZE BLOCKSIZEY * B_BATCHSIZE * TY



#define c(x) #x
#define stringify(x) c(x)

#define t(s1,s2) s1##s2
#define tg(s1,s2) t(s1,s2)

#define tgg(s1,s2,s3) tg(tg(s1,s2),s3)
#define sum(s2,s3)  tgg(sum,s2,s3)  

#define tggg(s1,s2,s3,s4) tg(tgg(s1,s2,s3),s4)


using data_type = float;
using datasize_type = int;



__global__  void naive2D_GEMM_Kernel_2(data_type* d_A, data_type* d_B,  data_type* d_C, datasize_type Arows, datasize_type Bcols, datasize_type AcolsBrows, data_type alpha, data_type beta) {
    int gid, gidx, gidy;
   gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x;
   gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y;
   gid = (Arows * gidy) + gidx;
   //leading dimension of the A_smem and B_smem array 
   //int LD =  A_BATCHSIZE * A_BATCHSIZE;

  //__shared__ float A_smem[A_SMEM_SIZE];
  //__shared__ float B_smem[B_SMEM_SIZE];
  extern __shared__ float AB_smem[];
  float sum[NUM_OF_RESULT_PER_THREAD];
  int ele;
  for (int i = 0; i < TX; i++) {
        for (int j = 0; j < TY; j++) {
            ele = i * TY + j;
            sum[ele] = 0.0;
	}
  }
  for (int k = 0; k < AcolsBrows; k+=B_BATCHSIZE)  {          
      for (int i = 0; i < TX; i++) {
	 for (int j = 0; j < NUM_OF_TRIPS_FOR_A; j++) {
           gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x + (i*blockDim.x); 
           AB_smem[threadIdx.x + (A_BATCHSIZE * (threadIdx.y + j * blockDim.y)) + (i * A_SMEM_BATCHSIZE)] = d_A[gidx + (k + threadIdx.y + (j * blockDim.y)) * Arows];  	  
         }       	     
      }	            
      __syncthreads();
      for (int j = 0; j < TY; j++) {
	 gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y + (j*blockDim.y);
         AB_smem[A_SMEM_SIZE + threadIdx.x + threadIdx.y * B_BATCHSIZE + (j * B_SMEM_BATCHSIZE)] = d_B[gidy * AcolsBrows + threadIdx.x + k];
      }
      __syncwarp();
      
     /* 
      for (int m = 0; m < B_BATCHSIZE; m++) {
          for (int j = 0; j < TY; j++) {
	      float Reg_B = AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)];  
              for (int i = 0; i < TX; i++) {
                  int ele = i * TY + j;
                  sum[ele]+=(  AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)] * AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]);   
              }
           }
      }
      */


      float Reg_A[(TX/2) * A_BATCHSIZE];
      
      for (int i = 0; i < 2; i++) {
	  for (int m = 0; m < B_BATCHSIZE; m++) {
	      int ind = i * B_BATCHSIZE + m;
	      Reg_A[ind] = AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)];  
          }
      } 
      
      #pragma unroll
      for (int j = 0; j < TY; j++) {
	  #pragma unroll
          for (int i = 0; i < 2; i++) {
	      #pragma unroll
              for (int m = 0; m < B_BATCHSIZE; m++) {
		  int index = i * B_BATCHSIZE + m;
		  int ele = i * TY + j;
                  sum[ele]+=(Reg_A[index] * AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]);  
	      }
	  }
       }
       
       
      for (int i = 2; i < 4; i++) {
	  for (int m = 0; m < B_BATCHSIZE; m++) {
	      int ind = i * B_BATCHSIZE + m;
	      Reg_A[ind] = AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)];  
          }
      } 
      
      #pragma unroll
      for (int j = 0; j < TY; j++) {
	  #pragma unroll 
          for (int i = 2; i < 4; i++) {
	      #pragma unroll
              for (int m = 0; m < B_BATCHSIZE; m++) {
		  int index = i * B_BATCHSIZE + m;
		  int ele = (i + 2) * TY + j;
                  sum[ele]+=(Reg_A[index] * AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]);  
	      }
	  }
       }

      /*       
                 
      for (int m = 0; m < B_BATCHSIZE; m++) {
          for (int i = 0; i < TX; i++) {
	      //float Reg_B = AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)];  
              for (int j = 0; j < TY; j++) {
                  int ele = i * TY + j;
                  //sum[ele]+=( Reg_A * AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]);   
              }
           }
      }

      */
      /*
      float Reg_A[TX * A_BATCHSIZE];
      int ind;
       
      for (int i = 0; i < TX; i++) {
	  for (int m = 0; m < B_BATCHSIZE; m++) {
	      ind = i * B_BATCHSIZE + m;
	      Reg_A[ind] = AB_smem[threadIdx.x + m * A_BATCHSIZE + (i * A_SMEM_BATCHSIZE)];  
          }
      } 
      */
       /* 
      for (int j = 0; j < TY; j++) {
          for (int i = 0; i < TX; i++) {
              for (int m = 0; m < B_BATCHSIZE; m++) {
		  ind = i * B_BATCHSIZE + m;
		  ele = i * TY + j;
 	//	  su += (Reg_A[index] * Reg_B[m]);  
                // sum[ele]+=(Reg_A[index] * Reg_B[m]);  
                   sum[ele]+=(Reg_A[ind] * AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]); 
	      }
	  }
       }
      */ 
      
     /*
      for (int i = 0; i < TX; i++) {
          for (int j = 0; j < TY; j++) {
	      #pragma unroll
              for (int m = 0; m < B_BATCHSIZE; m++) {
	          ind = i * B_BATCHSIZE + m;
	          ele = i * TY + j;
                  sum[ele]+=(Reg_A[ind] * AB_smem[A_SMEM_SIZE + m + threadIdx.y * B_BATCHSIZE + (B_SMEM_BATCHSIZE * j)]);  
	      }
	  }
      }
      
      */
      __syncthreads();                  

  }
    
  for (int i = 0; i < TX; i++) {
      gidx = (blockIdx.x * blockDim.x * TX) + threadIdx.x + (i*blockDim.x); 
      for (int j = 0; j < TY; j++) {
         gidy = (blockIdx.y * blockDim.y * TY) + threadIdx.y + (j*blockDim.y);
         ele = i * TY + j;
         gid = (Arows * gidy) + gidx; 
         d_C[gid] = alpha * sum[ele] + beta * d_C[gid];	
      
      } 
  }
	
}
