#include "./wrappers.cuh"

namespace Cuno {

template <>
void Wrappers::matvec_wrap(
  double* weights,
  double* layer,
  double* nextLayer,
  int N, int M
) {

  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 32;

  int blocks = (N + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = blocks;

  Kernels::matVecDot<<<BLOCKS, THREADS>>>(
      weights, layer, nextLayer,
      M, N
  );

  return;
}

};
