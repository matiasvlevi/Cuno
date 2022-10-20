#include "./utils.cuh"

ModelData *Utils::getModelData(
    v8::Isolate *env,
    const v8::Local<v8::Context> context,
    const v8::FunctionCallbackInfo<v8::Value>& args
) {
  
  // Abort if no array specified
  for (int i = 0; i < 3; i++) 
    if (!args[i]->IsObject()) return NULL; 

  // Store data
  ModelData *output = new ModelData(
    env, context,
    args[0].As<v8::Array>(),
    args[1].As<v8::Array>(),
    args[2].As<v8::Array>(),
    args[3].As<v8::Array>(),
    args[4].As<v8::Number>()
  );

  return output;

}
