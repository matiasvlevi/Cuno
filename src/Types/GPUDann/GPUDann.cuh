#include "../../cuno.cuh"
#include <cuda_runtime.h>

#ifndef DEVICE_DANN_H
#define DEVICE_DANN_H

namespace Cuno { 

template <class T>
class GPUDann {
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
  
  // /**
  //  * @brief Thread Kernel Launch parameters for each layers
  //  */
  // dim3 *THREADS;

  // /**
  // * @brief Thread Kernel Launch parameters for each layers
  // */
  // dim3 *BLOCKS;

  /**
   * @brief Whether or not the GPUDann is valid
   */
  bool valid;

  /**
  * @brief Construct a new GPUDann object
  * 
  * @param arch An array of the architecture of the model
  * @param length The length of the model's architecture
  */
  GPUDann(int *arch, int length) 
  {
    this->length = length;
    this->arch = (int*)malloc(sizeof(int) * length);
    
    // Allocate matrix device pointers for each layer
    this->layers    = (T**)malloc(sizeof(T*) * length);
    this->biases    = (T**)malloc(sizeof(T*) * length-1);
    this->weights   = (T**)malloc(sizeof(T*) * length-1);
    this->gradients = (T**)malloc(sizeof(T*) * length-1);
    this->errors    = (T**)malloc(sizeof(T*) * length-1);

    // Copy the architecture
    for (int i = 0; i < length; i++) this->arch[i] = arch[i];
    
    // Allocate device pointers
    this->valid = this->allocate();
    // this->THREADS = (dim3*)malloc(sizeof(dim3) * length-1);
    // this->BLOCKS = (dim3*)malloc(sizeof(dim3) * length-1);

    // for (int i = 0; i < length-1; i++) {
    //   // TODO: OPTIMIZE THREADS FOR DIFFERENT KERNELS/DIMENTIONS

    //   this->THREADS[i] = dim3(32, 32);
    //   int blocks = 
    //   (this->arch[i] + this->THREADS[i].y - 1) 
    //   / this->THREADS[i].y;

    //   this->BLOCKS[i] = dim3(blocks, blocks);
    // }
  }

  bool allocate(); 

  void toDevice(
    T **layers,
    T **biases,
    T **weights,
    T **gradients,
    T **errors
  ); 

  ~GPUDann() {
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
    // free(this->THREADS);
    // free(this->BLOCKS);
    this->layers    = 0;
    this->biases    = 0;
    this->weights   = 0;
    this->gradients = 0;
    this->errors    = 0;
    this->arch      = 0;
    // this->THREADS    = 0;
    // this->BLOCKS     = 0;
  }
};

};
#endif
