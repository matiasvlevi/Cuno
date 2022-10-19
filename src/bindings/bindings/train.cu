#include "../bindings.cuh"

void Bindings::train(const v8::FunctionCallbackInfo<v8::Value>& args) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();
 
  ModelData *input = Utils::getModelData(env, context, args); 

  Kernel::trainWrapper(/* input*/);
// 
//   // Convert to JS Array
//   v8::Local<v8::Array> buffer = v8::Array::New(env, input->outputLength);
// 
//   for (int i = 0; i < input->outputLength; i++) {
//     buffer->Set(context, i, v8::Number::New(env, *(c+i)));
//   }
//   
//   free(c);
// 
  args.GetReturnValue().Set(v8::Number::New(env, 4));
}


