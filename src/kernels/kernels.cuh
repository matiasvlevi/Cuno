#include <cuda.h>
#include "../cuno.cuh"

#ifndef KERNELS_H
#define KERNELS_H
namespace Cuno {

/**
* CUDA Kernels
*/
namespace Kernels {

/**
* Matrix Dot Product: (M * N) dot (N * P) = (M * P)
*
* @param[in] a The 'A' matrix
* @param[in] b The 'B' matrix 
* @param[in] c The result matrix, The 'C' matrix
* @param[in] M Rows of 'A' 
* @param[in] N Rows of 'B' & Cols of 'A'
* @param[in] P Cols of 'B'
*/  
__global__ void dot(double *a, double *b, double *c, int M, int N, int P);

/**
  THIS USES ONLY THE X THREADS
* Matrix Add op
*
* @param[out] a The 'A' matrix and the result matrix
* @param[in] b The 'B' matrix 
* @param[in] P Unwrapped N by N length of the matrix
*/
__global__ void add(double *a, double *b, int P);

/**
  THIS USES ONLY THE X THREADS
* Matrix initiate at 0:
*
* @param[out] a The Matrix to reset
* @param[in] P Unwrapped N by N length of the matrix
*/
__global__ void reset(double *a, int P);

/**
  THIS USES ONLY THE X THREADS
* Matrix map(sigmoid) operation:
*
* @param[out] a The Matrix to map
* @param[in] P Unwrapped N by N length of the matrix
*/
__global__ void sigmoid(double *a, int P);

/**
* Matrix/Vector Dot Product: (M * N) dot (N * 1) = (M * 1)
* Can ONLY take in a Matrix as 'A' and a vector as 'B'  
* @param[in] a The 'A' matrix
* @param[in] b The 'B' vector
* @param[out] c The result vector, The 'C' vector
* @param[in] M Rows of 'A' 
* @param[in] N Rows of 'B' & Cols of 'A'
*/  
__global__ void matVecDot(double *a, double *b, double *c, int M, int N);

/**
* Matrix/Vector Dot Product + Bias: (M * N) dot (N * 1) = (M * 1)
* Can ONLY take in a Matrix as 'A' and a vector as 'B'  
* @param[in] a The 'A' matrix
* @param[in] b The 'B' vector
* @param[out] c The result vector, The 'C' vector
* @param[in] d The biases 
* @param[in] M Rows of 'A' 
* @param[in] N Rows of 'B' & Cols of 'A'
*/ 
__global__ void layerConv(double *a, double *b, double *c, double *d, int M, int N);

};

};
#endif
