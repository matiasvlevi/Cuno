#include "./kernels.cuh"

namespace Cuno {

__global__ 
void Kernels::layerConv(
  double *a,
  double *b,
  double *c,
  double *d,
  int M,
  int N
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  
  if (row >= M || col >= 1) return;
  
  for (int k = 0; k < N; k++)   
    c[row + col] += a[row * N + k] * b[k + col] + d[row];
  
    
  return;
}

};
