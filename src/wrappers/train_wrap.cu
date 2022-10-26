#include "./wrappers.cuh"

void Cuno::Wrappers::train_wrap(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  Cuno::DeviceDann<double> *nn = 
  Cuno::v8Utils::FromNativeModel<double>(context, env, args); 

  for (int i = 0; i < nn->length; i++) {
    std::cout << "Layer " << i << std::endl;
    Log::deviceArray<double>(nn->layers[i], nn->arch[i]);
  
    if (i >= nn->length-1) continue; 

    std::cout << "Weight " << i << std::endl;
    Log::deviceArray<double>(nn->weights[i], nn->arch[i] * nn->arch[i+1]);

    std::cout << "Biases " << i << std::endl;
    Log::deviceArray<double>(nn->biases[i], nn->arch[i+1]);

    std::cout << "errors " << i << std::endl;
    Log::deviceArray<double>(nn->errors[i], nn->arch[i+1]);

    std::cout << "gradients " << i << std::endl;
    Log::deviceArray<double>(nn->gradients[i], nn->arch[i+1]);

  }
  int x;
  std::cin >> x;
  args.GetReturnValue().Set(v8::Number::New(env, x));
}

