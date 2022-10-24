
#ifndef METHOD_INPUT_H
#define METHOD_INPUT_H

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

  void allocate() {
    // TODO: CHECK FOR AVAILABLE SPACE IN DEVICE MEM

    cudaMalloc(&(this->a), sizeof(T) * M * N); 
    cudaMalloc(&(this->b), sizeof(T) * N * P);
    cudaMalloc(&(this->c), sizeof(T) * M * P);
  }

  void toDevice(T *values_a, T *values_b) {
    cudaMemcpy(
        this->a, values_a,
        sizeof(T) * this->M * this->N,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        this->b, values_b,
        sizeof(T) * this->N * this->P,
        cudaMemcpyHostToDevice
    );
  }

  void getOutput(T *buffer) {
    cudaMemcpy(
      buffer, this->c,
      sizeof(T) * this->M * this->P,
      cudaMemcpyDeviceToHost
    );
  } 

  ~MethodInput() {
    // Free device pointers
    cudaFree(this->a); 
    cudaFree(this->b);
    cudaFree(this->c);
  }
};
#endif
