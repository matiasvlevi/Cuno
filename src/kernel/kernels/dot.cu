#include "../kernel.cuh"

__global__ 
void Kernel::dot(
    double *a,
    double *b,
    double *c,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;	

    // Abort if out of range
    if (row >= M || col >= P) {
      printf("Kernel out of range! %d, %d\n", row, col);
      return;
    }

    // Compute row
    double sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[row * N + k] * b[k * P + col];
    }
    c[row * P + col] = sum;

    return;
}

__global__ 
void Kernel::dotOpt(
    double *a,
    double *b,
    double *c,
    int M,
    int N,
    int P
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;	

    // Abort if out of range
    if (row >= M || col >= N) {
      return;
    }


    //printf("Pos: %d, %d Vals: %d %d %d\n", row, col, a[row * N + col], b[col * P]);

    c[row * P] += a[row * N + col] * b[col * P];
    
    return;
}

void Kernel::DotWrapper(
    double *a,
    double *b,
    double *c,
    int M,
    int N,
    int P
) {
    double *dev_a = 0; 
    double *dev_b = 0;
    double *dev_c = 0;

    size_t sizeA = sizeof(double) * M * N;
    size_t sizeB = sizeof(double) * N * P;
    size_t sizeC = sizeof(double) * M * P;

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
