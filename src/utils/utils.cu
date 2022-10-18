#include "./utils.cuh"

MethodInput *Utils::convertArgs(
    const v8::Local<v8::Context> context,
    const v8::FunctionCallbackInfo<v8::Value>& args
) {
  
  // Abort if no array specified
  for (int i = 0; i < 2; i++) 
    if (!args[i]->IsArray()) return NULL; 

  // Abort if no dimensions specified 
  for (int i = 2; i < 5; i++) 
    if (!args[i]->IsNumber()) return NULL; 

  // Create output object
  MethodInput *output = new MethodInput(
    args[2].As<v8::Number>()->Value(),
    args[3].As<v8::Number>()->Value(),
    args[4].As<v8::Number>()->Value()
  );
  
  int i, j;
  for (i = 0; i < 2; i++) {
    v8::Local<v8::Array> array = (args[i]).As<v8::Array>();

    for (j = 0; j < output->outputLength; j++) {       
    
        v8::MaybeLocal<v8::Value> maybeValue = (array->Get(context, j)); 
        v8::Local<v8::Number> value = maybeValue.FromMaybe(
            v8::Local<v8::Value>()
        ).As<v8::Number>();
    
        // Fill in the data
        if (i % 2) output->b[j] = value->Value();
        else output->a[j] = value->Value();
    }
  }

  return output;

}
