#include "../v8/v8utils.cuh"
#include "../kernels/kernels.cuh"

#ifndef WRAPPERS_H
#define WRAPPERS_H
namespace Cuno {

namespace Wrappers {
  void dot_wrap(const v8::FunctionCallbackInfo<v8::Value>& args);
  void train_wrap(const v8::FunctionCallbackInfo<v8::Value>& args);
};

};
#endif
