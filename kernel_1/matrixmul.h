

#ifndef __MATRIXMUL_H
#define __MATRIXMUL_H


__global__  void naive_GEMM_Kernel_1(float* d_A, float* d_B,  float* d_C, unsigned int Arows, unsigned int Bcols, unsigned int AcolsBrows, float alpha, float beta);

__global__  void matrixMulAddRowBasedARR(double* weightBias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wColsXRows);

__global__  void matrixMulColBasedARR(double* weight, double* bias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wRowsXRows);


__global__  void matrixMulRowBasedARR(double* weight, double* bias, double* xData, double* activationValues, long long int wRows, long long int xCols, long long int wRowsXRows);



#endif
