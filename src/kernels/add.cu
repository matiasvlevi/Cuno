#include "./kernels.cuh"
#include <stdio.h>
namespace Cuno {

__global__ void Kernels::add(
  double *a,
  double *b,
  int P
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col >= P) {
    return;
  }  

  a[col] = a[col] + b[col];
}

};
