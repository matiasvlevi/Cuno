#include "../wrappers/wrappers.cuh"

#ifndef BINDINGS_H
#define BINDINGS_H
namespace Cuno {

namespace Bindings {
     void FeedForward(const v8::FunctionCallbackInfo<v8::Value>& args);
     
     void Dot(const v8::FunctionCallbackInfo<v8::Value>& args);
     
     void MatVecDot(const v8::FunctionCallbackInfo<v8::Value>& args);
     
     void Map(const v8::FunctionCallbackInfo<v8::Value>& args);

     void Init(v8::Local<v8::Object> exports); 
};

};
#endif
