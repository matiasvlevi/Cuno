

#include <cuda.h>
#include "../components/DeviceModelData.cuh"

#ifndef KERNEL_H
#define KERNEL_H

namespace Kernel {
    void TrainWrapper(ModelData *host, DeviceModelData *device);

    void DotWrapper(
        double *a,
        double *b,
        double *c,
        int M,
        int N,
        int P
    );

    __global__ 
    void dot(double *a, double *b, double *c, int M, int N, int P); 

    __global__ 
    void dotOpt(double *a, double *b, double *c, int M, int N, int P); 



};
#endif
