#include "./bindings.cuh"


namespace Cuno {

void Bindings::MatVecDot(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  MethodInput<double> *device =
    v8Utils::getSingleCallArgs<double>(context, args, true);

  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 32;

  int blocks = (device->N + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = blocks;

  //double *d = 0;
  //double bias_values[device->N] = {};

  //cudaMalloc(&d, sizeof(double) * device->N);
  //cudaMemcpy(d, bias_values, sizeof(double) * device->N, cudaMemcpyHostToDevice);

  Kernels::matVecDot<<<BLOCKS, THREADS>>>(
      device->a, device->b, device->c,
      device->M, device->N
  );

  double buffer[device->M * device->P];
  device->getOutput(buffer);

  v8::Local<v8::Array> output = 
    v8Utils::toJaggedArray<double>(context, env, buffer, device->M, device->P);

  Log::deviceMatrix<double>(device->a, device->M, device->N);
  Log::deviceMatrix<double>(device->b, device->N, device->P);
  Log::deviceMatrix<double>(device->c, device->M, device->P);
  
  args.GetReturnValue().Set(output);
  return;
}

};
