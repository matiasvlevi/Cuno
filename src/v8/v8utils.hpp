#include <node.h>

#include "../error/error.hpp"

#include "../Types/MethodInput/MethodInput.cuh"
#include "../Types/GPUDann/GPUDann.cuh"

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

  /**
  * Convert argument field to a device allocated Dannjs model
  *
  * @param[in] context The v8 context
  * @param[in] env     The v8 environement
  * @param[in] args    The API side call arguments
  * @param[in] index   The argument field number
  */ 
  template <class T>
  Cuno::GPUDann<T> *FromNativeModel(
    Local<Context> context,
    Isolate *env,
    const FunctionCallbackInfo<Value>& args,
    int index = 0
  );

  /**
  * Get argument fields as device pointers for a single call calculation 
  * (mostly called from the nodejs API)
  *
  * @param[in] context The v8 context
  * @param[in] args    The v8 arguments
  * @param[in] matVec  Whether or not 'b' matrix is actually a vector (1 column)
  */ 
  template <class T>
  Cuno::MethodInput<T> *getSingleCallArgs(
    const Local<Context> context,
    const FunctionCallbackInfo<Value>& args,
    bool matVec = false
  );

  /**
  * Assume the value is defined, cast away MaybeLocal to local
  *
  * @param[in] maybeValue Maybe Value to cast await
  */ 
  template <class T>
  Local<Value> FromMaybe(MaybeLocal<T> maybeValue) {
    Local<Value> ans = maybeValue.FromMaybe(
      Local<Value>()
    );
    return ans;
  }

  /**
  * Get value from v8 local Array as v8 T
  *
  * @param[in] context The v8 context
  * @param[in] array   The v8 local array
  * @param[in] index   THe v8 value's index
  */ 
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

  /**
  * Convert T buffer to Array
  *
  * @param[in] context    The v8 context
  * @param[in] env        The v8 environement
  * @param[in] buffer     T type buffer
  * @param[in] length     The length of the buffer 
  */ 
  template <class T>
  Local<Array> toArray(
      Local<Context> context,
      Isolate *env,
      T *buffer,
      int length
  ) {
    Local<Array> array = Array::New(env, length); 
    for (int i = 0; i < length; i++) {
      (void)array->Set(context, i, v8::Number::New(env, (T)(*(buffer + i))));
    }
    return array;
  }

  /**
  * Convert T buffer to jagged array based on R,C dimensions
  *
  * @param[in] context   The v8 context
  * @param[in] env       The v8 environement
  * @param[out] buffer   T type buffer 
  * @param[in] R         Row dimension
  * @param[in] C         Column dimension
  */ 
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

  /**
  * Get value from v8 Object
  *
  * @param[in] context    The v8 context
  * @param[in] env        The v8 environement
  * @param[in] matrix     The Dannjs native matrix
  * @param[in] key        The object string key
  */ 
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

  /**
  * @brief Convert a v8 local array to a T buffer 
  */ 
  template <class T>
  void fromArrayToBuf(
    Local<Context> context,
    T *buf,
    Local<Array> array
  ) {
    for (int i = 0; i < array->Length(); i++) {
      buf[i] = v8Utils::getFromArray<Number>(context, array, i)->Value();
    }
  }

};

};
#endif
