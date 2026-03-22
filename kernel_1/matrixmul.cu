#include <math.h>
#include "matrixmul.h"    
#include <stdio.h>

using namespace std;


__global__  void naive_GEMM_Kernel_1(float* d_A, float* d_B,  float* d_C, unsigned int Arows, unsigned int Bcols, unsigned int AcolsBrows, float alpha, float beta) {
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   if (gid < (Arows * Bcols)) { 
        int indexW = gid % Arows;
	int index = gid / Arows;
	int indexRefB = index * AcolsBrows;
    	float sum = 0.0;
	for (int i = 0; i < AcolsBrows; i++)  {
	   sum+=(d_A[indexW + i * Arows] * d_B[i+indexRefB]);
       	}
        d_C[gid] = alpha * sum + beta * d_C[gid];	
	 
   }

}







__global__  void matrixMulColBasedARR(double* weight, double* bias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
   //  int x_index = gid/NUMDATAS;
   //  int y_index = gid % NUMDATAS;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	long long int index = gid / xCols;
        long long int indexW = index * wColsXRows;
        long long int indexStart = gid % wColsXRows;
        long long int IndexMul = indexStart * wColsXRows; 
    	double  sum = 0.0;
	for (long long int i = 0; i < wColsXRows; i++)  {
	   sum+=(weight[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=bias[index];
        activationValues[gid] = sum;	
	 
    }

}


__global__  void matrixMulRowBasedARR(double* weight, double* bias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
   //  int x_index = gid/NUMDATAS;
   //  int y_index = gid % NUMDATAS;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	long long int index = gid / xCols;
	long long int indexR = gid % xCols;
        long long int indexW = index * wColsXRows;
        long long int indexStart = gid % wColsXRows;
        long long int IndexMul = indexStart * wColsXRows; 
    	double  sum = 0.0;
	for (long long int i = 0; i < wColsXRows; i++)  {
	   sum+=(weight[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=bias[index];
        activationValues[index+(indexR*wRows)] = sum;	
	 
    }

}



__global__  void matrixMulAddRowBasedARR(double* weightBias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	long long int index = gid / xCols;
	long long int indexR = gid % xCols;
        long long int indexW = index * (wColsXRows+1);
        long long int indexStart = gid % wColsXRows;
        long long int IndexMul = indexStart * wColsXRows; 
    	double  sum = 0.0;
	for (long long int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=weightBias[indexW+wColsXRows];
        activationValues[index+(indexR*wRows)] = sum;	
	 
    }

}

