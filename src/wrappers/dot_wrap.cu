#include "./wrappers.cuh"
namespace Cuno {

void Wrappers::dot_wrap(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  MethodInput<double> *device =
    v8Utils::getSingleCallArgs<double>(context, args);

  if (device == 0) {
    args.GetReturnValue().Set(v8::Number::New(env, -1));
    return;
  }

  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 32;

  int blocks = (device->N + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = blocks;

  Kernels::dot<<<BLOCKS, THREADS>>>(
      device->a, device->b, device->c,
      device->M, device->N, device->P
  );

  double buffer[device->M * device->P];
  device->getOutput(buffer);
  delete device;

  v8::Local<v8::Array> output = 
    v8Utils::toJaggedArray<double>(context, env, buffer, device->M, device->P);

  // Log::deviceMatrix<double>(device->a, device->M, device->N);
  // Log::deviceMatrix<double>(device->b, device->N, device->P);
  // Log::deviceMatrix<double>(device->c, device->M, device->P);

  args.GetReturnValue().Set(output);
  return;
}

};
