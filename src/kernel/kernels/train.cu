#include "../kernel.cuh"

__global__ 
void Kernel::train(
    float *a,
    float *b,
    float *c,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;	

    // Abort if out of range
    if (row >= M || col >= P) return;

    // Compute row
    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[row * N + k] * b[k * P + col];
    }
    c[row * P + col] = sum;

    return;
}

void Kernel::trainWrapper(
    float *a,
    float *b,
    float *c,
    int M,
    int N,
    int P
) {
    float *dev_a = 0; 
    float *dev_b = 0;
    float *dev_c = 0;

    size_t sizeA = sizeof(float) * M * N;
    size_t sizeB = sizeof(float) * N * P;
    size_t sizeC = sizeof(float) * M * P;

    cudaMalloc(&dev_a, sizeA);
    cudaMalloc(&dev_b, sizeB); 
    cudaMalloc(&dev_c, sizeC); 

    cudaMemcpy(dev_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, sizeC, cudaMemcpyHostToDevice);

    dim3 THREADS;
    THREADS.x = 32;
    THREADS.y = 32;

    int blocks = (N + THREADS.x - 1) / THREADS.x;

    dim3 BLOCKS;
    BLOCKS.x = blocks;
    BLOCKS.y = blocks;

    Kernel::dot<<<BLOCKS, THREADS>>>(dev_a, dev_b, dev_c, M, N, P);

    cudaMemcpy(c, dev_c, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}