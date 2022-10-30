#include "../../cuno.cuh"

#ifndef METHOD_INPUT_H
#define METHOD_INPUT_H
namespace Cuno {

template <class T>
class MethodInput {
public:
  T *a, *b, *c;
  int M, N, P;
  
  MethodInput(int M, int N, int P) {
    this->a = 0;
    this->b = 0;
    this->c = 0;

    this->M = M;
    this->N = N;
    this->P = P;

    // Allocate device pointers
    this->allocate();
  }

  ~MethodInput() {
    // Free device pointers
    cudaFree(this->a); 
    cudaFree(this->b);
    cudaFree(this->c);
  }

  void allocate();

  void toDevice(T *values_a, T *values_b);

  void getOutput(T *buffer);
};

};
#endif
