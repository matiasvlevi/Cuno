#include "./bindings/bindings.cuh"

void Bindings::Init(v8::Local<v8::Object> exports)  {
  NODE_SET_METHOD(exports, "dot", Bindings::DotProd);
  NODE_SET_METHOD(exports, "train", Bindings::train);
};

NODE_MODULE(NODE_GYP_MODULE_NAME, Bindings::Init)

