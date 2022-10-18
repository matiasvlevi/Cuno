#ifndef KERNEL_H
#define KERNEL_H
#include <cuda.h>
#include <iostream> // TODO: REMOVE 
namespace Kernel {

    // __global__ 
    // void add(float *input, float *c, int N); 

    // void AddWrapper(
    //     float *input,
    //     float *c,
    //     int N
    // );

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
