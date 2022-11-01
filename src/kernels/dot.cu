#include "./kernels.cuh"
namespace Cuno {


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
__global__ void Kernels::dot(
  double *a,
  double *b,
  double *c,
  int M,
  int N,
  int P
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= P) return;
  
  for (int k = 0; k < N; k++)   
    c[row * P + col] += a[row * N + k] * b[k * P + col];
  
}

};
