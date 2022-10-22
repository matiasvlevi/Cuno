#include "./kernels.cuh"
namespace Cuno {

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
