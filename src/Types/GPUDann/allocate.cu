#include "./GPUDann.cuh"

#include "../../error/error.hpp"

template <>
bool Cuno::GPUDann<double>::allocate() 
{
 cudaError_t error;
 for (int i = 0; i < this->length; i++) {

    this->layers[i]    = 0; 
    error = cudaMalloc(&(this->layers[i])   , sizeof(double) * this->arch[i]);
    if (error != cudaSuccess) {
      Error::throw_("'layers' were not successfully allocated");
      return false;
    };

    if (i >= this->length-1) continue;

    this->biases[i]    = 0;
    this->weights[i]   = 0;
    this->gradients[i] = 0;
    this->errors[i]    = 0;
    
    error = cudaMalloc(&(this->biases[i])   , sizeof(double) * this->arch[i+1]);
    if (error != cudaSuccess) {
      Error::throw_("'biases' were not successfully allocated");
      return false;
    };
    error = cudaMalloc(&(this->weights[i])  , sizeof(double) * this->arch[i] * this->arch[i+1]);
    if (error != cudaSuccess) {
      Error::throw_("'weights' were not successfully allocated");
      return false;
    };
    error = cudaMalloc(&(this->gradients[i]), sizeof(double) * this->arch[i+1]); 
    if (error != cudaSuccess) {
      Error::throw_("'gradients' were not successfully allocated");
      return false;
    };
    error = cudaMalloc(&(this->errors[i])   , sizeof(double) * this->arch[i+1]);
    if (error != cudaSuccess) {
      Error::throw_("'errors' were not successfully allocated");
      return false;
    };
  }

  return true;
}
