// #include "../kernel.cuh"

// __global__ 
// void Kernel::add(
//     float *input,
//     float *c,
//     int N
// ) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;	

//     // Abort if out of range
//     if (col * N + row >= N) return;

//     c[row] = input[row + N] + input[row];

//     return;
// }

// void kernel::AddWrapper(
//     float *input,
//     float *c,
//     int N
// ) {

//   float *dev_input = 0;
//   float *dev_c = 0;
//   size_t bufSize = sizeof(float) * N;
 
//   cudaMalloc(&dev_input, bufSize*2); 
//   cudaMalloc(&dev_c, bufSize); 
   
//   cudaMemcpy(dev_input, input, bufSize * 2, cudaMemcpyHostToDevice);
//   cudaMemcpy(dev_c, c, bufSize, cudaMemcpyHostToDevice);
 
//  	dim3 THREADS;
//  	THREADS.x = 32;
//  	THREADS.y = 32;
 
//  	int blocks = (N + THREADS.x - 1) / THREADS.x;
 
//  	dim3 BLOCKS;
//  	BLOCKS.x = blocks;
//  	BLOCKS.y = blocks;

//   Kernel::add<<<BLOCKS, THREADS>>>(dev_input, dev_c, N);

//   cudaMemcpy(c, dev_c, bufSize, cudaMemcpyDeviceToHost);
  

//   cudaFree(dev_input);
//   cudaFree(dev_c);

// }