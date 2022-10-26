#include "../DeviceDann.cuh"

template <>
void Cuno::DeviceDann<double>::toDevice(
  double **layers,
  double **biases,
  double **weights,
  double **gradients,
  double **errors
) {
  for (int i = 0; i < this->length; i++) {
    cudaMemcpy(
      this->layers[i], (double*)layers[i],
      sizeof(double) * this->arch[i],
      cudaMemcpyHostToDevice
    );
    if (i >= this->length-1) continue;

    cudaMemcpy(
      this->biases[i], (double*)biases[i],
      sizeof(double) * this->arch[i+1],
      cudaMemcpyHostToDevice
    );

    cudaMemcpy(
      this->weights[i], weights[i],
      sizeof(double) * this->arch[i] * this->arch[i+1],
      cudaMemcpyHostToDevice
    );

    cudaMemcpy(
      this->gradients[i], (double*)gradients[i],
      sizeof(double) * this->arch[i+1],
      cudaMemcpyHostToDevice
    );

    cudaMemcpy(
      this->errors[i], (double*)errors[i],
      sizeof(double) * this->arch[i+1],
      cudaMemcpyHostToDevice
    );
  }
} 
