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
  Local<T> FromMaybe(MaybeLocal<T> maybeValue) {
    Local<T> ans = maybeValue.FromMaybe(
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

  template <class T>
  DeviceDann<T> *FromNativeModel(
    Local<Context> context,
    Isolate *env,
    const FunctionCallbackInfo<Value>& args
  ) {
    Local<Object> native_model = args[0].As<Object>(); 

    // Get model's arch
    MaybeLocal<Value> maybe_arch = native_model->Get(
        context,
        String::NewFromUtf8Literal(env, "arch")
    );
    Local<Array> native_arch = v8Utils::FromMaybe<Value>(maybe_arch).As<Array>();


    int arch[native_arch->Length()];
    for (int i = 0; i < native_arch->Length(); i++) {
      arch[i] = v8Utils::getFromArray<Number>(context, native_arch, i)->Value();
    }

    DeviceDann<T> *model = new DeviceDann<T>(arch, (uint8_t)native_arch->Length());

    Log::hostArray<int>(model->arch, model->length);

    T *layers[model->length];
    for (uint8_t i = 0 ; i < model->length; i++) {
      T list[model->arch[i]] = {};
      
      layers[i] = list;
    }

//     T *biases[model->length-1];
//     for (uint8_t i = 0 ; i < model->length-1; i++) {
//       T list[model->arch[i]];
//       for (int j = 0; j < model->arch[i+1]; j++) {
//         biases[i][j] = 0;
//       }
//     }
// 
//     T *weights[model->length-1];
//     for (uint8_t i = 0 ; i < model->length-1; i++) {
//       for (int j = 0; j < model->arch[i] * model->arch[i+1]; j++) {
//         weights[i][j] = 0;
//       }
//     }
// 
//     T *gradients[model->length-1];
//     for (uint8_t i = 0 ; i < model->length-1; i++) {
//       for (int j = 0; j < model->arch[i+1]; j++) {
//         gradients[i][j] = 0;
//       }
//     }
// 
//     T *errors[model->length-1];
//     for (uint8_t i = 0 ; i < model->length-1; i++) {
//       for (int j = 0; j < model->arch[i+1]; j++) {
//         errors[i][j] = 0;
//       }
//     }

    Log::hostArray<double>(layers[0], model->arch[0]);
// 
//     model->toDevice(
//      layers, biases, weights, gradients, errors 
//     );


    //Log::deviceMatrix(model->weights[0], model->arch[0], model->arch[1]);

    return model; 
  }

};

};
#endif
