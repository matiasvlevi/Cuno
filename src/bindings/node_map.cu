#include "./bindings.cuh"

namespace Cuno {

void Bindings::Map(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  MethodInput<double> *device =
    v8Utils::getSingleCallArgs<double>(context, args);

  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 32;

  int blocks = (device->N + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = blocks;

  Kernels::sigmoid<<<BLOCKS, THREADS>>>(
      device->a, device->M * device->N
  );

  double buffer[device->M * device->N];
  //device->getOutput(buffer);

  cudaMemcpy(buffer, device->a, sizeof(double) * device->N * device->M, cudaMemcpyDeviceToHost);

  v8::Local<v8::Array> output = 
    v8Utils::toJaggedArray<double>(context, env, buffer, device->M, device->N);

  Log::deviceMatrix<double>(device->a, device->M, device->N);
  
  args.GetReturnValue().Set(output);
  return;
}

};
