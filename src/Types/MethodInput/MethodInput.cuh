#include "../../cuno.cuh"

#ifndef METHOD_INPUT_H
#define METHOD_INPUT_H
namespace Cuno {

/**
 * Contains device pointers to the matrix values for a, b & c and their size
 * 
 * @tparam T Value type, recommended to use double
 */
template <class T>
class MethodInput {
public:

  /**
  * @brief Matrix device pointers
  */
  T *a, *b, *c;
  
  /**
   * @brief Matrix dimensions
   */
  int M, N, P;

  /**
   * @brief wheter or not the memory allocation is valid
   * 
   */
  bool valid;

  /**
   * @brief Construct a new Method Input object
   * 
   * @param M Rows of the A matrix
   * @param N Rows of the B matrix & Cols of the A matrix (COMMON DIMENSION) 
   * @param P Cols of the B matrix
   */
  MethodInput(int M, int N, int P) {
    this->a = 0;
    this->b = 0;
    this->c = 0;

    this->M = M;
    this->N = N;
    this->P = P;

    // Allocate device pointers & store validity of memory allocation
    this->valid = this->allocate();
  }

  /**
   * @brief Destroy the Method Input object
   * Free device pointers
   */
  ~MethodInput() {
    // Free device pointers
    cudaFree(this->a); 
    cudaFree(this->b);
    cudaFree(this->c);
  }

  /**
  * @brief Allocate device pointers, return false if failed.
  */
  bool allocate();

  /**
   * @brief Move host values to device values
   * 
   * @param values_a A matrix as a T*
   * @param values_b B matrix as a T*
   */
  void toDevice(T *values_a, T *values_b);

  /**
   * @brief Get the ouput values from the device to the host
   * 
   * @param buffer A host buffer to save the device values on the host.
   */
  void getOutput(T *buffer);

  static void error();
};

};
#endif
