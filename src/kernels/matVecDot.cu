#include "./kernels.cuh"
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

namespace Cuno {

__global__ 
void Kernels::matVecDot(
  double *a,
  double *b,
  double *c,
  int M,
  int N
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;
  
  atomicAdd(&c[row], a[row * N + col] * b[col]);
    
  return;
}

};
