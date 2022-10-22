#ifndef DEVICE_DANN_H
#define DEVICE_DANN_H
namespace Cuno {

template <class T>
class DeviceDann {
public:
  // Model
  uint8_t length;
  int *arch;
  T **layers;
  T **biases;
  T **weights;

  // Train related
  T **gradients;
  T **errors;

  DeviceDann(int *arch, uint8_t length) {
    this->length = length;
    this->arch = (int*)malloc(sizeof(int) * length);
    
    this->layers   = (T**)malloc(sizeof(T*) * length);
    this->biases   = (T**)malloc(sizeof(T*) * length-1);
    this->weights  = (T**)malloc(sizeof(T*) * length-1);
    this->gradients = (T**)malloc(sizeof(T*) * length-1);
    this->errors    = (T**)malloc(sizeof(T*) * length-1);
    for (uint8_t i = 0; i < length; i++) this->arch[i] = arch[i];

    //this->allocate();
  }

  void allocate() {
   for (uint8_t i = 0; i < this->length; i++) {
      this->layers[i]    = 0;
 
      cudaMalloc(&(this->layers[i])   , sizeof(T) * this->arch[i]);

      if (i == this->length-1) continue;

      this->biases[i]    = 0;
      this->weights[i]   = 0;
      this->gradients[i] = 0;
      this->errors[i]    = 0;

      cudaMalloc(&(this->biases[i])   , sizeof(T) * this->arch[i+1]);
      cudaMalloc(&(this->weights[i])  , sizeof(T) * this->arch[i] * this->arch[i+1]);
      cudaMalloc(&(this->gradients[i]), sizeof(T) * this->arch[i+1]);
      cudaMalloc(&(this->errors[i])   , sizeof(T) * this->arch[i+1]);
    }
  }


  void toDevice(
    T **layers,
    T **biases,
    T **weights,
    T **gradients,
    T **errors
  ) {
    for (uint8_t i = 0; i < this->length; i++) {
      cudaMemcpy(
        this->layers[i], layers[i],
        sizeof(T) * this->arch[i],
        cudaMemcpyHostToDevice
      );
      if (i == this->length-1) continue;

      cudaMemcpy(
        this->biases[i], biases[i],
        sizeof(T) * this->arch[i+1],
        cudaMemcpyHostToDevice
      );

      cudaMemcpy(
        this->weights[i], weights[i],
        sizeof(T) * this->arch[i] * this->arch[i+1],
        cudaMemcpyHostToDevice
      );

      cudaMemcpy(
        this->gradients[i], gradients[i],
        sizeof(T) * this->arch[i+1],
        cudaMemcpyHostToDevice
      );

      cudaMemcpy(
        this->errors[i], errors[i],
        sizeof(T) * this->arch[i+1],
        cudaMemcpyHostToDevice
      );
    }
  } 

  ~DeviceDann() {
    for (uint8_t i = 0; i < this->length; i++) {
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
