#include "./wrappers.cuh"

void Cuno::Wrappers::ffw(DeviceDann<double> *nn, double *input) {
  Log::hostArray<double>(input, nn->arch[0]);
  cudaMemcpy(nn->layers[0], input,nn->arch[0] * sizeof(double), cudaMemcpyHostToDevice);

  for (int i = 0; i < nn->length-1; i++) {
    Wrappers::matvec_wrap<double>(
       nn->weights[i], nn->layers[i],
       nn->layers[i+1],
       nn->arch[i], nn->arch[i+1]
    );

    Wrappers::add_wrap<double>(nn->layers[i+1], nn->biases[i], nn->arch[i+1]);
    Wrappers::sigmoid_wrap<double>(nn->layers[i+1], nn->arch[i+1]);

    Log::deviceArray<double>(nn->layers[i+1], nn->arch[i+1]);
  }


  std::cout << "output" << std::endl;
  Log::deviceArray<double>(nn->layers[nn->length-1], nn->arch[nn->length-1]);
}




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



  // Log all
//   for (int i = 0; i < nn->length; i++) {
//     std::cout << "Layer " << i << std::endl;
//     Log::deviceArray<double>(nn->layers[i], nn->arch[i]);
//   
//     if (i >= nn->length-1) continue; 
// 
//     std::cout << "Weight " << i << std::endl;
//     Log::deviceArray<double>(nn->weights[i], nn->arch[i] * nn->arch[i+1]);
// 
//     std::cout << "Biases " << i << std::endl;
//     Log::deviceArray<double>(nn->biases[i], nn->arch[i+1]);
// 
//     std::cout << "errors " << i << std::endl;
//     Log::deviceArray<double>(nn->errors[i], nn->arch[i+1]);
// 
//     std::cout << "gradients " << i << std::endl;
//     Log::deviceArray<double>(nn->gradients[i], nn->arch[i+1]);
//   }

  double buffer[1 * nn->arch[nn->length-1]];
  cudaMemcpy(buffer, nn->layers[nn->length-1], nn->arch[nn->length-1] * sizeof(double), cudaMemcpyDeviceToHost);
  v8::Local<v8::Array> output = 
    v8Utils::toJaggedArray<double>(context, env, buffer, 1, nn->arch[nn->length-1]); 

  int x;
  std::cin >> x;
  args.GetReturnValue().Set(output);
}

