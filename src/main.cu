#include "./bindings/bindings.cuh"

void Cuno::Bindings::Init(v8::Local<v8::Object> exports)  {
  NODE_SET_METHOD(exports, "train", Cuno::Wrappers::train_wrap);
  NODE_SET_METHOD(exports, "dot"  , Cuno::Wrappers::dot_wrap);
};

NODE_MODULE(NODE_GYP_MODULE_NAME, Cuno::Bindings::Init)

