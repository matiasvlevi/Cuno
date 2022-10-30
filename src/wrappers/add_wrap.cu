#include "./wrappers.cuh"

namespace Cuno {

template <> 
void Wrappers::add_wrap(double *a, double *b, int P) {
  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 1;

  int blocks = (P + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = 1;

  Kernels::add<<<BLOCKS, THREADS>>>(
      a, b, P
  );

}

};
