#include "./MethodInput.cuh"
#include "../../error/error.hpp"
#include <cuda_runtime.h>

namespace Cuno {

template<>
bool MethodInput<double>::allocate() {
  // TODO: CHECK FOR AVAILABLE SPACE IN DEVICE MEM
  cudaError_t error;
  
  error = cudaMalloc(&(this->a), sizeof(double) * M * N); 
  if (error != cudaSuccess) {
    Error::throw_("Matrix 'a' failed to properly allocate ");
    return false;
  }
  error = cudaMalloc(&(this->b), sizeof(double) * N * P);
  if (error != cudaSuccess) {
    Error::throw_("Matrix 'b' failed to properly allocate ");
    return false;
  }
  error = cudaMalloc(&(this->c), sizeof(double) * M * P);
  if (error != cudaSuccess) {
    Error::throw_("Matrix 'c' (result) failed to properly allocate ");
    return false;
  }

  return true;
}

template<>
void MethodInput<double>::toDevice(double *values_a, double *values_b) {
  if (!(this->valid)) return;
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
  if (!(this->valid)) return;
   cudaMemcpy(
    buffer, this->c,
    sizeof(double) * this->M * this->P,
    cudaMemcpyDeviceToHost
  );
} 


};