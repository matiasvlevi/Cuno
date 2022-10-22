#include "./wrappers.cuh"
namespace Cuno {

void Wrappers::train_wrap(
  const v8::FunctionCallbackInfo<v8::Value>& args
) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  


  DeviceDann<double> *nn = 
    v8Utils::FromNativeModel<double>(context, env, args); 
  
  args.GetReturnValue().Set(v8::Number::New(env, 1));
}

};
