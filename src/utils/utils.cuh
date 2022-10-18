
#include <node/node.h>
#include "../kernel/kernel.cuh"
#include "../types/methodInput.cuh"

#ifndef UTILS_H
#define UTILS_H
namespace Utils {

MethodInput *convertArgs(
    const v8::Local<v8::Context> context,
    const v8::FunctionCallbackInfo<v8::Value>& args
);

};
#endif