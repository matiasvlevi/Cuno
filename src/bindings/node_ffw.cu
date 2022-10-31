#include "./bindings.cuh"

/**
  Wrapper for the batch train node binding, 
  creates a device dann instance, which allocates memory on the cuda device.
  Launches a cuda kernel after allocation
*/  
void Cuno::Bindings::FeedForward(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  Cuno::GPUDann<double> *nn = 
  Cuno::v8Utils::FromNativeModel<double>(context, env, args); 

  double inputs[nn->arch[0]];

  for (int i = 0; i < nn->arch[0]; i++)
    inputs[i] = 0;
  

  Wrappers::ffw(nn, inputs);

  // from Device
  double buffer[1 * nn->arch[nn->length-1]];
  cudaMemcpy(
      buffer, nn->layers[nn->length-1],
      nn->arch[nn->length-1] * sizeof(double),
      cudaMemcpyDeviceToHost
  );

  // to v8
  v8::Local<v8::Array> output = 
    v8Utils::toJaggedArray<double>(context, env, buffer, 1, nn->arch[nn->length-1]); 

  // Wait for user input
  int x;
  std::cin >> x;

  // return v8 value
  args.GetReturnValue().Set(output);
}

