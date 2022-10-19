
#include <node/node.h>

#ifndef MODELDATA_H
#define MODELDATA_H

class ModelData {
  public:

    // Training config 
    int epoch;
    std::vector<float*> inputs;
    std::vector<float*> outputs;

    // Model
    std::vector<int> arch;
    std::vector<float*> layers;
    std::vector<float*> weights;
    std::vector<float*> biases;
    ModelData();
    ModelData(
      v8::Isolate *env,
      v8::Local<v8::Context> context,
      v8::Local<v8::Array> arch,
      v8::Local<v8::Array> weights,
      v8::Local<v8::Array> biases,
      v8::Local<v8::Array> data,
      v8::Local<v8::Number> epoch
    );

    void setWeights(int index, float *weight_ptr);

    template <class T> 
    void logPtr(T values, int length);

    template <class T> 
    void logArrayData(std::vector<T> array, int length = 0);

    template <class T> 
    void logArrayAsModelComponent(std::vector<T> array, int dec = 0);

    void logWeights(std::vector<float *> weights);

    void logData();

    void logModel(); 

    float *fromArrayToFloatAlloc(
      v8::Local<v8::Context> context,
      v8::Local<v8::Array> array
    );

    template <class T> 
    v8::Local<T> getFromArray(
      v8::Local<v8::Context> context,
      v8::Local<v8::Array> array,
      int index
    );

    template <class T> 
    std::vector<T> jaggedToAlloc(
        v8::Local<v8::Context> context,
        v8::Local<v8::Array> array
    );

    template <class T> 
    std::vector<T> toArray(
        v8::Local<v8::Context> context,
        v8::Local<v8::Array> array
    );

    ~ModelData();

};

#endif
