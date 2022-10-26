#include "./kernels.cuh"
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

    // Abort if out of range
    if (row >= M || col >= N) {
      return;
    }

    // Sum the product of a matrix(a) row, and a vector(b)
    c[row] += a[row * N + col] * b[col];
    // Outputs a vector containing all dot products of vectors
    
    return;
}

}
