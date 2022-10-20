#include "../kernel/kernel.cuh"

#ifndef BINDINGS_H
#define BINDINGS_H
namespace Bindings {
    namespace Utils {
        MethodInput *convertArgs(
            const v8::Local<v8::Context> context,
            const v8::FunctionCallbackInfo<v8::Value>& args
        );
    }
    void DotProd(const v8::FunctionCallbackInfo<v8::Value>& args);
    void train(const v8::FunctionCallbackInfo<v8::Value>& args);
    void Init(v8::Local<v8::Object> exports);
};

#endif
