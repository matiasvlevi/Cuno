#include "./kernel.cuh"

__global__ void Kernel::add(
    float *input,
    float *c,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;	

    // Abort if out of range
    if (col >= N) return;

    c[col] = input[N + col] + input[col];

    return;
}
