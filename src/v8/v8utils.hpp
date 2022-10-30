#include <node.h>

#include "../logger/logger.hpp"

#include "../Types/MethodInput/MethodInput.cuh"
#include "../Types/DeviceDann/DeviceDann.cuh"

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
  Cuno::DeviceDann<T> *FromNativeModel(
    Local<Context> context,
    Isolate *env,
    const FunctionCallbackInfo<Value>& args
  );

  template <class T>
  Cuno::MethodInput<T> *getSingleCallArgs(
    const Local<Context> context,
    const FunctionCallbackInfo<Value>& args
  );

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

};

};
#endif
