#include "../kernel.cuh"


// __global__ 
// void Kernel::train(
//     float *a,
//     float *b,
//     float *c,
//     int M,
//     int N,
//     int P
// ) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;	

//     // Abort if out of range
//     if (row >= M || col >= P) return;

//     // Compute row
//     float sum = 0;
//     for (int k = 0; k < N; k++) {
//         sum += a[row * N + k] * b[k * P + col];
//     }
//     c[row * P + col] = sum;

//     return;
// }

void Kernel::TrainWrapper(
    ModelData *host,
    DeviceModelData *device
) {

  //cudaMalloc(&(device->layers[i]), device->arch[i] * sizeof(double));


}
