#include "../utils/logger.cuh"

#ifndef MODELDATA_H
#define MODELDATA_H

class ModelData {
  public:

    // Training config 
    int epoch;
    std::vector<double*> inputs;
    std::vector<double*> outputs;

    // Model
    std::vector<unsigned int> arch;
    std::vector<double*> layers;
    std::vector<double*> weights;
    std::vector<double*> biases;

    // Default
    ModelData();

    // From v8 Args
    ModelData(
      v8::Isolate *env,
      const v8::Local<v8::Context> context,
      const v8::FunctionCallbackInfo<v8::Value>& args
    );

    // From filtered v8 values
    void Allocate(
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

    static size_t getMemoryUsage(std::vector<unsigned int> arch);
    static void debugMemory(const size_t model_size, const size_t data_size);

    void setWeights(int index, double *weight_ptr);

    void logLayers();

    void logData();

    void logModel(); 

    template <class T>
    void ptr_arr(T values, unsigned int length);

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
