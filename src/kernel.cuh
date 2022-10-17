#ifndef KERNEL_H
#define KERNEL_H
#include <cuda.h>

#include <iostream>

namespace Kernel {


__global__
void add(float *input, float *c, int N); 


}
#endif
