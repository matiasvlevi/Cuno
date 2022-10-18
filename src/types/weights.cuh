
#ifndef MODELDATA_H
#define MODELDATA_H

template <typename T>
struct Blob {
  T input;
  T output;
};

class ModelData {
public:

  std::vector<float*> inputs;
  std::vector<float*> outputs;
  std::vector<float*> weights;
  std::vector<float*> biases;
  std::vector<int> arch;

  ModelData(
    v8::Isolate *env,
    v8::Local<v8::Context> context,
    v8::Local<v8::Array> arch,
    v8::Local<v8::Array> weights,
    v8::Local<v8::Array> biases,
    v8::Local<v8::Array> data,
    v8::Local<v8::Number> epoch
  ) {
   
    // Get architecture
    this->arch = this->toArray<int>(context, arch);

    // Get weights
    this->weights = this->jaggedToAlloc<float*>(context, weights);

    // Get biases
    this->biases = this->jaggedToAlloc<float*>(context, biases); 

    for (int i = 0; i < data->Length(); i++) {

      v8::MaybeLocal<v8::Value> maybeValue = (data->Get(context, i)); 
      v8::Local<v8::Array> array_set = maybeValue.FromMaybe(
        v8::Local<v8::Value>()
      ).As<v8::Array>();


      for (int j = 0; j < 2; j++) {
        v8::MaybeLocal<v8::Value> maybeValue = (array_set->Get(context, j)); 
        v8::Local<v8::Array> datapoint = maybeValue.FromMaybe(
          v8::Local<v8::Value>()
        ).As<v8::Array>();

        if (i % 2) this->outputs.push_back(this->fromArrayToFloatAlloc(context, datapoint));
        else this->inputs.push_back(this->fromArrayToFloatAlloc(context, datapoint));
      }


      std::cout << "INPUTS" << std::endl;
      for (int k = 0; k < 8; k++) std::cout << this->inputs[k] << std::endl;
      std::cout << "end" << std::endl;
    }
  }


  float *fromArrayToFloatAlloc(
    v8::Local<v8::Context> context,
    v8::Local<v8::Array> array
  ) {
    float *alloc = (float*)malloc(sizeof(float) * array->Length());

    for (int i = 0; i < array->Length(); i++) {
      v8::MaybeLocal<v8::Value> maybeValue = (array->Get(context, i)); 
      v8::Local<v8::Number> value = maybeValue.FromMaybe(
        v8::Local<v8::Value>()
      ).As<v8::Number>();
      std::cout << "Actual values " << i << std::endl;
      std::cout << value->Value() << std::endl;
     (alloc)[i] = value->Value();
    }
    return alloc;

  }

  template <class T>
  std::vector<T> jaggedToAlloc(
      v8::Local<v8::Context> context,
      v8::Local<v8::Array> array
  ) {
    std::vector<T> ans;
    int i, j;
    for (i = 0; i < array->Length(); i++) {
         v8::MaybeLocal<v8::Value> maybeValue = (array->Get(context, i)); 
         v8::Local<v8::Array> array_set = maybeValue.FromMaybe(
             v8::Local<v8::Value>()
         ).As<v8::Array>();
 

        float *row_data = (float*)malloc(sizeof(float) * array_set->Length());
        // Get weight raw matrix
        for (j = 0; j < array_set->Length(); j++) {
          v8::Local<v8::Number> value = (array_set->Get(context, j)).FromMaybe(
              v8::Local<v8::Value>()
          ).As<v8::Number>();
 
          row_data[j] = value->Value(); 
        }

        ans.push_back(row_data);
    }
    return ans;
  }

  template <class T>
  std::vector<T> toArray(
      v8::Local<v8::Context> context,
      v8::Local<v8::Array> array
  ) {
    std::vector<T> ans;
    for (int i = 0; i < array->Length(); i++) {       
        v8::MaybeLocal<v8::Value> maybeValue = (array->Get(context, i)); 
        v8::Local<v8::Number> value = maybeValue.FromMaybe(
            v8::Local<v8::Value>()
        ).As<v8::Number>();
        ans.push_back(value->Value());
    }
    return ans;
  }

  void setWeights(int index, float *weight_ptr) {
    this->weights[index] = weight_ptr;
  }

  ~ModelData() {
    for (int i = 0; i < this->weights.size(); i++) {
        free(this->weights[i]);
    }

    for (int i = 0; i < this->biases.size(); i++) {
        free(this->biases[i]);
    }

    for (int i = 0; i < this->inputs.size(); i++) {
        free(this->inputs[i]);
    }

    for (int i = 0; i < this->outputs.size(); i++) {
        free(this->outputs[i]);
    }
  }
};

#endif
