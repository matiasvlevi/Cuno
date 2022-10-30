#include "./MethodInput.cuh"
#include <cuda_runtime.h>

namespace Cuno {

template<>
void MethodInput<double>::allocate() {
  // TODO: CHECK FOR AVAILABLE SPACE IN DEVICE MEM

  cudaMalloc(&(this->a), sizeof(double) * M * N); 
  cudaMalloc(&(this->b), sizeof(double) * N * P);
  cudaMalloc(&(this->c), sizeof(double) * M * P);
}

template<>
void MethodInput<double>::toDevice(double *values_a, double *values_b) {
  cudaMemcpy(
      this->a, values_a,
      sizeof(double) * this->M * this->N,
      cudaMemcpyHostToDevice
  );
  cudaMemcpy(
      this->b, values_b,
      sizeof(double) * this->N * this->P,
      cudaMemcpyHostToDevice
  );
}

template<>
void MethodInput<double>::getOutput(double *buffer) {
  cudaMemcpy(
    buffer, this->c,
    sizeof(double) * this->M * this->P,
    cudaMemcpyDeviceToHost
  );
} 

};