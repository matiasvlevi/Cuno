#include "../v8/v8utils.hpp"
#include "../kernels/kernels.cuh"

#ifndef WRAPPERS_H
#define WRAPPERS_H
namespace Cuno {

namespace Wrappers {

  template<class T>
  void matvec_wrap(
    T* weights,
    T* layer,
    T* nextLayer,
    int N, int M
  );

  template<class T>
  void layer_wrap(
    T* weights,
    T* layer,
    T* nextLayer,
    T* biases,
    int N, int M
  );


  void ffw(GPUDann<double>* nn, double *input);

  template <class T>
  void add_wrap(T *a, T *b, int P);

  template <class T>
  void sigmoid_wrap(T *a, int P);

  void dot_wrap(const v8::FunctionCallbackInfo<v8::Value>& args);
   
};

};
#endif
