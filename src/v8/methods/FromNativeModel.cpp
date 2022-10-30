#include "../v8utils.hpp"

template <>
Cuno::DeviceDann<double> *Cuno::v8Utils::FromNativeModel(
  Local<Context> context,
  Isolate *env,
  const FunctionCallbackInfo<Value>& args
) {
  Local<Object> native_model = args[0].As<Object>(); 

  // Get model's arch
  Local<Value> key = String::NewFromUtf8Literal(env, "arch").As<Value>();
  MaybeLocal<Value> maybe_arch = native_model->Get(context, key);
  Local<Array> native_arch = v8Utils::FromMaybe<Value>(maybe_arch).As<Array>();

  // Get model's weights
  Local<Array> native_weights = v8Utils::getFrom<Array>(context, env, native_model, "weights"); 

  // Get model's weights
  Local<Array> native_biases = v8Utils::getFrom<Array>(context, env, native_model, "biases"); 

  int arch[native_arch->Length()];
  for (int i = 0; i < native_arch->Length(); i++) {
    arch[i] = v8Utils::getFromArray<Number>(context, native_arch, i)->Value();
  }

  DeviceDann<double> *model = new DeviceDann<double>(arch, (uint8_t)native_arch->Length());

  double *layers[model->length];
  for (uint8_t i = 0 ; i < model->length; i++) {
    layers[i] = (double*)malloc(sizeof(double) * model->arch[i]);
    for (int j = 0; j < model->arch[i]; j++) {
      layers[i][j] = 0; 
    }
  } 

  double *biases[model->length-1];
  for (uint8_t i = 0 ; i < model->length-1; i++) {
    Local<Object> matrix = v8Utils::getFromArray<Object>(context, native_weights, i);
    Local<Array> bmatrix = v8Utils::getFrom<Array>(context, env,
      matrix,
      "matrix"
    );
    biases[i] = (double*)malloc(sizeof(double) * model->arch[i+1]);
    for (int j = 0; j < model->arch[i+1]; j++) {
      biases[i][j] = v8Utils::getFromArray<Number>(context,
          v8Utils::getFromArray<Array>(context, bmatrix, j).As<Array>(), 0
      )->Value(); 
    }
  }

  double *weights[model->length-1];
  for (uint8_t i = 0 ; i < model->length-1; i++) {
    
    weights[i] = (double*)malloc(sizeof(double) * model->arch[i] * model->arch[i+1]);
    Local<Object> matrix = v8Utils::getFromArray<Object>(context, native_weights, i);
    Local<Array> wmatrix = v8Utils::getFrom<Array>(context, env,
      matrix,
      "matrix"
    );

    int k = 0;
    int r = 0; 
    Local<Array> row;
    for (int j = 0; j < model->arch[i] * model->arch[i+1]; j++) {
      if (j % model->arch[i] == 0) {
        row = v8Utils::getFromArray<Array>(context, wmatrix, r).As<Array>();
        k = 0;
        r++;
      } 
      weights[i][j] = v8Utils::getFromArray<Number>(context, row, k)->Value();
      k++;
    }
  }

  double *gradients[model->length-1];
  for (uint8_t i = 0 ; i < model->length-1; i++) {
    gradients[i] = (double*)malloc(sizeof(double) * model->arch[i+1]);
    for (int j = 0; j < model->arch[i+1]; j++) {
      gradients[i][j] = 0; 
    }      
  }

  double *errors[model->length-1];
  for (uint8_t i = 0 ; i < model->length-1; i++) {
    errors[i] = (double*)malloc(sizeof(double) * model->arch[i+1]);
    for (int j = 0; j < model->arch[i+1]; j++) {
      errors[i][j] = 0; 
    }
  }

  model->toDevice(
    layers, biases, weights, gradients, errors 
  );

  for(int i = 0; i < model->length; i++) {
    free(layers[i]);
    if (i >= model->length-1) continue;
    free(biases[i]);
    free(gradients[i]);
    free(errors[i]);
    free(weights[i]);
  }

  return model; 
};
