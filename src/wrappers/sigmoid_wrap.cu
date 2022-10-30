#include "./wrappers.cuh"

namespace Cuno {

template <> 
void Wrappers::sigmoid_wrap(double *a, int P) {
  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 1;

  int blocks = (P + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = 1;

  Kernels::sigmoid<<<BLOCKS, THREADS>>>(
      a, P
  );

}

};
