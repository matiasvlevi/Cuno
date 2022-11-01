#include <cuda.h>
#include "../cuno.cuh"

#ifndef KERNELS_H
#define KERNELS_H
namespace Cuno {

namespace Kernels {

__global__ void dot(double *a, double *b, double *c, int M, int N, int P);

__global__ void add(double *a, double *b, int P);

__global__ void reset(double *a, int P);

__global__ void sigmoid(double *a, int P);

__global__ void matVecDot(double *a, double *b, double *c, int M, int N);

__global__ void layerConv(double *a, double *b, double *c, double *d, int M, int N);

};

};
#endif
