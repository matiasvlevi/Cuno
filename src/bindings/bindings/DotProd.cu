#include "../bindings.cuh"

void Bindings::DotProd(const v8::FunctionCallbackInfo<v8::Value>& args) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  MethodInput *input = Utils::convertArgs(context, args); 

  double *c = (double*)malloc(sizeof(double) * input->outputLength);

  Kernel::DotWrapper(input->a, input->b, c, input->M, input->N, input->P);

  // Convert to JS Array
  v8::Local<v8::Array> buffer = v8::Array::New(env, input->outputLength);

  for (int i = 0; i < input->outputLength; i++) {
    buffer->Set(context, i, v8::Number::New(env, *(c+i)));
  }
  
  free(c);

  args.GetReturnValue().Set(buffer);
}


