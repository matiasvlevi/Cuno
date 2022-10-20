

#include <cuda.h>
#include "../utils/utils.cuh"

#ifndef KERNEL_H
#define KERNEL_H

namespace Kernel {
    void TrainWrapper(ModelData *host, DeviceModelData *device);

    __global__ 
    void dot(double *a, double *b, double *c, int M, int N, int P); 

    void DotWrapper(
        double *a,
        double *b,
        double *c,
        int M,
        int N,
        int P
    );

};
#endif
