#include "./bindings/bindings.cuh"

void Cuno::Bindings::Init(v8::Local<v8::Object> exports)  {
  NODE_SET_METHOD(exports, "train", Cuno::Bindings::FeedForward);
  NODE_SET_METHOD(exports, "dot"  , Cuno::Bindings::Dot);
  NODE_SET_METHOD(exports, "matVecDot"  , Cuno::Bindings::MatVecDot);
  NODE_SET_METHOD(exports, "map"  , Cuno::Bindings::Map);
};

NODE_MODULE(NODE_GYP_MODULE_NAME, Cuno::Bindings::Init)

