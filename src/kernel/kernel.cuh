#include <cuda.h>

#include <iostream> // TODO: REMOVE 

#include "../types/DeviceModelData.cuh"

#ifndef KERNEL_H
#define KERNEL_H


namespace Kernel {
    void trainWrapper();

    __global__ 
    void dot(float *a, float *b, float *c, int M, int N, int P); 

    void DotWrapper(
        float *a,
        float *b,
        float *c,
        int M,
        int N,
        int P
    );

};
#endif
