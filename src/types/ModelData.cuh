#include <node/node.h>
#include <iostream>

#ifndef MODELDATA_H
#define MODELDATA_H

class ModelData {
  public:

    // Training config 
    int epoch;
    std::vector<double*> inputs;
    std::vector<double*> outputs;

    // Model
    std::vector<int> arch;
    std::vector<double*> layers;
    std::vector<double*> weights;
    std::vector<double*> biases;

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

    static size_t MAX_HOST_ALLOC;
    static size_t MAX_DEVICE_ALLOC;

    static size_t getMemoryUsage(std::vector<int> arch);
    static void debugMemory(const size_t model_size, const size_t data_size);

    void setWeights(int index, double *weight_ptr);

    template <class T> 
    void logPtr(T values, int length);

    template <class T> 
    void logArrayData(std::vector<T> array, int length = 0);

    template <class T> 
    void logArrayAsModelComponent(std::vector<T> array, int dec = 0);

    void logWeights(std::vector<double *> weights);

    void logData();

    void logModel(); 

    double *fromArrayToFloatAlloc(
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
