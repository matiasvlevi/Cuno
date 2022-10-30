#include "../../cuno.cuh"

#ifndef DEVICE_DANN_H
#define DEVICE_DANN_H

namespace Cuno { 

template <class T>
class DeviceDann {
public:
  // Model
  int length;
  int *arch;
  T **layers;
  T **biases;
  T **weights;

  // Train related
  T **gradients;
  T **errors;

  DeviceDann(int *arch, int length) 
  {
    this->length = length;
    this->arch = (int*)malloc(sizeof(int) * length);
    
    this->layers    = (T**)malloc(sizeof(T*) * length);
    this->biases    = (T**)malloc(sizeof(T*) * length-1);
    this->weights   = (T**)malloc(sizeof(T*) * length-1);
    this->gradients = (T**)malloc(sizeof(T*) * length-1);
    this->errors    = (T**)malloc(sizeof(T*) * length-1);
    for (int i = 0; i < length; i++) this->arch[i] = arch[i];
    this->allocate();
  }

  void allocate(); 

  void toDevice(
    T **layers,
    T **biases,
    T **weights,
    T **gradients,
    T **errors
  ); 

  ~DeviceDann() {
    for (int i = 0; i < this->length; i++) {
      cudaFree(this->layers[i]);
      cudaFree(this->biases[i]);
      cudaFree(this->weights[i]);
      cudaFree(this->gradients[i]);
      cudaFree(this->errors[i]);
    }

    // Free device pointer arrays in heap
    free(this->layers);
    free(this->biases);
    free(this->weights);
    free(this->gradients);
    free(this->errors);
    free(this->arch);
    this->layers    = 0;
    this->biases    = 0;
    this->weights   = 0;
    this->gradients = 0;
    this->errors    = 0;
    this->arch      = 0;
  }

};

};
#endif
