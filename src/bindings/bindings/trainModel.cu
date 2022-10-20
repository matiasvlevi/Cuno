#include "../bindings.cuh"

void Bindings::train(const v8::FunctionCallbackInfo<v8::Value>& args) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();
 
  ModelData *input = new ModelData(env, context, args); 
  DeviceModelData *output = new DeviceModelData(input);

  Kernel::TrainWrapper(input, output);


  //v8::Local<v8::Object> v8_output = output->toV8();
  //args.GetReturnValue().Set(v8_output);
  
  args.GetReturnValue().Set(v8::Number::New(env, 100));
}


