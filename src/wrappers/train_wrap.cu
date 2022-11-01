#include "./wrappers.cuh"


/**
  Wrapper for the batch train node binding, 
  creates a device dann instance, which allocates memory on the cuda device.
  Launches a cuda kernel after allocation
*/  
void Cuno::Wrappers::train_wrap(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  Cuno::DeviceDann<double> *nn = 
  Cuno::v8Utils::FromNativeModel<double>(context, env, args); 



  double inputs[nn->arch[0]];

  for (int i = 0; i < nn->arch[0]; i++) {
    inputs[i] = 1;
  }

  Wrappers::ffw(nn, inputs);

  
  double buffer[1 * nn->arch[nn->length-1]];
  cudaMemcpy(buffer, nn->layers[nn->length-1], nn->arch[nn->length-1] * sizeof(double), cudaMemcpyDeviceToHost);
  v8::Local<v8::Array> output = 
    v8Utils::toJaggedArray<double>(context, env, buffer, 1, nn->arch[nn->length-1]); 

  int x;
  std::cin >> x;
  args.GetReturnValue().Set(output);
}

