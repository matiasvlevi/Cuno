#include "./kernels.cuh"
#include <stdio.h>
namespace Cuno {

__global__ void Kernels::sigmoid(
  double *a,
  int P
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= P) {
    return;
  }  

  a[col] = 1 / (1 + exp(-a[col]));
}

};
