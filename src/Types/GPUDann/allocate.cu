#include "./GPUDann.cuh"


template <>
void Cuno::GPUDann<double>::allocate() 
{
 for (int i = 0; i < this->length; i++) {

    this->layers[i]    = 0; 
    cudaMalloc(&(this->layers[i])   , sizeof(double) * this->arch[i]);

    if (i >= this->length-1) continue;

    this->biases[i]    = 0;
    this->weights[i]   = 0;
    this->gradients[i] = 0;
    this->errors[i]    = 0;
    
    cudaMalloc(&(this->biases[i])   , sizeof(double) * this->arch[i+1]);
    cudaMalloc(&(this->weights[i])  , sizeof(double) * this->arch[i] * this->arch[i+1]);
    cudaMalloc(&(this->gradients[i]), sizeof(double) * this->arch[i+1]); 
    cudaMalloc(&(this->errors[i])   , sizeof(double) * this->arch[i+1]);
  }
}
