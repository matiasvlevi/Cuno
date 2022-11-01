#include "./kernels.cuh"

namespace Cuno {

__global__
void Kernels::reset(double *a, int length) {
  
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= length) return;

  a[row] = 0;
}

}
