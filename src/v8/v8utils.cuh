#include <node.h>

#include "../logger/logger.cuh"

#include "../Types/MethodInput.cuh"
#include "../Types/DeviceDann.cuh"

#ifndef V8_UTILS_H
#define V8_UTILS_H
namespace Cuno {

namespace v8Utils {

  using v8::Array;
  using v8::Value;
  using v8::Context;
  using v8::Local;
  using v8::MaybeLocal;
  using v8::FunctionCallbackInfo;
  using v8::Number;
  using v8::Object;
  using v8::String;
  using v8::Isolate;

  template <class T>
  Local<Value> FromMaybe(MaybeLocal<T> maybeValue) {
    Local<Value> ans = maybeValue.FromMaybe(
      Local<Value>()
    );
    return ans;
  }

  template <class T>
  Local<T> getFromArray(
    Local<Context> context,
    Local<Array> array,
    int index
  ) {
    MaybeLocal<Value> maybeValue = array->Get(context, index); 
    Local<T> value = maybeValue.FromMaybe(
      Local<Value>()
    ).As<T>();
    return value;
  }

  template <class T>
  MethodInput<T> *getSingleCallArgs(
    const Local<Context> context,
    const FunctionCallbackInfo<Value>& args
  ) {
    // Abort if arguments are not arrays
    for (int i = 0; i < 2; i++) if (!args[i]->IsArray()) {
      // SHOULD ABORT IF ARRAYS DO NOT CONTAIN ARRAYS WITH THE SAME SIZE
      return 0;
    }

    // Get computation dimensions from given Matrix Jagged Arrays 
    int M = args[0].As<Array>()->Length();
    int N = v8Utils::getFromArray<Array>(context, args[0].As<Array>(), 0)->Length();
    int P = v8Utils::getFromArray<Array>(context, args[1].As<Array>(), 0)->Length();

    // Allocate device memory
    MethodInput<T> *input = new MethodInput<T>(M, N, P);

    // Allocate temporary stack pointers
    T a[M * N];
    T b[N * P];

    for (int k = 0; k < 2; k++) {
      // Get the Matrix Jagged Array from v8 values
      Local<Array> matrix = args[k].As<Array>();

      for (int i = 0; i < matrix->Length(); i++) {
        // Get a row from the Matrix Jagged Array
        Local<Array> currentRow = 
          v8Utils::getFromArray<Array>(context, matrix, i);
    
        for (int j = 0; j < currentRow->Length(); j++) {
          // Get the value from the Matrix Jagged Array
          T value = 
            v8Utils::getFromArray<Number>(context, currentRow, j)->Value();

          // Allocate stack matrix pointers
          if (k % 2 == 0) a[i * currentRow->Length() + j] = value; 
          else b[i * currentRow->Length() + j] = value; 

        }
      }
    }

    input->toDevice(a, b);
    return input;
  }

  template <class T>
  Local<Array> toArray(
      Local<Context> context,
      Isolate *env,
      T *buffer,
      int length
  ) {
    Local<Array> array = Array::New(env, length); 
    for (int i = 0; i < length; i++) {
      array->Set(context, i, v8::Number::New(env, (T)(*(buffer + i))));
    }
    return array;
  }

  template <class T>
  Local<Array> toJaggedArray(
      Local<Context> context,
      Isolate *env,
      T *buffer,
      int R,
      int C
  ) {
    Local<Array> array = Array::New(env, R); 
    for (int i = 0; i < array->Length(); i++) {
      array->Set(context, i, v8Utils::toArray<T>(context, env, buffer + C*i, C));
    }
    return array;
  }

//   Local<Array> getRawMatrix(
//       Local<Context> context,
//       Isolate *env,
//       Local<Object> matrix
//   ) {
//     return v8Utils::FromMaybe<Value>(matrix->Get(
//         context,
//         String::NewFromUtf8Literal(env, "matrix").As<Value>()
//     )).As<Array>();
//   } 
// 

  template <class T>
  Local<T> getFrom(
      Local<Context> context,
      Isolate *env,
      Local<Object> matrix,
      const char *key
  ) {
    return v8Utils::FromMaybe<Value>(matrix->Get(
        context,
        v8Utils::FromMaybe<String>(String::NewFromUtf8(env, key)).As<Value>()
    )).As<T>();
  } 

  template <class T>
  DeviceDann<T> *FromNativeModel(
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

    DeviceDann<T> *model = new DeviceDann<T>(arch, (uint8_t)native_arch->Length());

    T *layers[model->length];
    for (uint8_t i = 0 ; i < model->length; i++) {
      layers[i] = (T*)malloc(sizeof(T) * model->arch[i]);
      for (int j = 0; j < model->arch[i]; j++) {
        layers[i][j] = 0; 
      }
    } 

    T *biases[model->length-1];
    for (uint8_t i = 0 ; i < model->length-1; i++) {
      Local<Object> matrix = v8Utils::getFromArray<Object>(context, native_weights, i);
      Local<Array> bmatrix = v8Utils::getFrom<Array>(context, env,
        matrix,
        "matrix"
      );
      biases[i] = (T*)malloc(sizeof(T) * model->arch[i+1]);
      for (int j = 0; j < model->arch[i+1]; j++) {
        biases[i][j] = v8Utils::getFromArray<Number>(context,
            v8Utils::getFromArray<Array>(context, bmatrix, j).As<Array>(), 0
        )->Value(); 
      }
    }

    T *weights[model->length-1];
    for (uint8_t i = 0 ; i < model->length-1; i++) {
      
      weights[i] = (T*)malloc(sizeof(T) * model->arch[i] * model->arch[i+1]);
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

    T *gradients[model->length-1];
    for (uint8_t i = 0 ; i < model->length-1; i++) {
      gradients[i] = (T*)malloc(sizeof(T) * model->arch[i+1]);
      for (int j = 0; j < model->arch[i+1]; j++) {
        gradients[i][j] = 0; 
      }      
    }

    T *errors[model->length-1];
    for (uint8_t i = 0 ; i < model->length-1; i++) {
      errors[i] = (T*)malloc(sizeof(T) * model->arch[i+1]);
      for (int j = 0; j < model->arch[i+1]; j++) {
        errors[i][j] = 0; 
      }
    }

    Log::hostArray<double>(layers[0], model->arch[0]);

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
  }

};

};
#endif
