#include "./ModelData.cuh"
#include <iostream>

ModelData::ModelData() {
  this->epoch = 1;
  this->arch = std::vector<int>();
  this->weights = std::vector<float*>();
  this->biases = std::vector<float*>();
  this->layers = std::vector<float*>();
}

ModelData::ModelData(
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

  // Get layers
  for (int i = 0; i < this->arch.size(); i++) {
    float *layer_alloc = (float*)malloc(sizeof(float) * this->arch[i]);

    // Init data
    for (int j = 0; j < this->arch[i]; j++) {
      layer_alloc[j] = 0.0f;
    }

    this->layers.push_back(layer_alloc);
  }

  // Get weights
  this->weights = this->jaggedToAlloc<float*>(context, weights);

  // Get biases
  this->biases = this->jaggedToAlloc<float*>(context, biases); 

  for (int i = 0; i < data->Length(); i++) {
    v8::Local<v8::Array> array_set =this->getFromArray<v8::Array>(
        context, data, i
    );

    v8::Local<v8::Array> inputs = this->getFromArray<v8::Array>(
      context, array_set, 0
    );
    this->inputs.push_back(this->fromArrayToFloatAlloc(context, inputs));

    v8::Local<v8::Array> outputs = this->getFromArray<v8::Array>(
      context, array_set, 1
    );
    this->outputs.push_back(this->fromArrayToFloatAlloc(context, outputs));
  }

  this->epoch = epoch->Value();


}

template <class T>
void ModelData::logPtr(T values, int length) {

  for (int i = 0; i < length-1; i++) 
    std::cout << "\x1b[92m" << values[i] << "\x1b[0m, ";

  std::cout << "\x1b[92m" << values[length-1] << "\x1b[0m";
  std::cout << "\n" << std::endl;
}

template <class T>
void ModelData::logArrayData(std::vector<T> array, int length) {
  for (int i = 0; i < array.size(); i++) {
    this->logPtr<T>(array[i], (length > 0) ? length : this->arch[i+1]);
  } 
  std::cout << "\n";
} 

template <class T>
void ModelData::logArrayAsModelComponent(std::vector<T> array, int dec) {

  std::cout << "fptr VECTOR" << std::endl;
  for (int i = 0; i < array.size(); i++) {
    this->logPtr<T>(array[i], this->arch[i+dec]);
  } 
} 

void ModelData::logWeights(std::vector<float *> weights) {
  for (int i = 0; i < weights.size(); i++) {
    
    std::cout << "Weights " << i <<
      "  (" << this->arch[i] <<
      "," << this->arch[i+1] <<  ") {";

    for (int j = 0; j < this->arch[i] * this->arch[i+1]; j++) {
    
      if (j % this->arch[i] == 0) 
        std::cout << "\n\t";
      else 
        std::cout << ", ";
      
      std::cout << "\x1b[92m" << weights[i][j] << "\x1b[0m";
    }

    std::cout << "\n}\n" << std::endl;
  }
}

void ModelData::logData() {
  std::cout << "DATASET --- Inputs ---\n" << std:: endl;
  this->logArrayData<float *>(this->inputs, this->arch[0]);

  std::cout << "DATASET --- Output ---\n" << std:: endl;
  this->logArrayData<float *>(this->outputs, this->arch[this->arch.size()-1]);
}

void ModelData::logModel() {
  
  std::cout << "Epoch target specified: \n" << this->epoch << std::endl;

  std::cout << "--- Weights --- \n" << std:: endl;
  this->logWeights(this->weights);

  std::cout << "--- Biases --- \n" << std:: endl;
  this->logArrayAsModelComponent<float *>(this->biases, 1);

  std::cout << "--- Layers ---\n" << std:: endl;
  this->logArrayAsModelComponent<float *>(this->layers);
}

template <class T>
v8::Local<T> ModelData::getFromArray(
  v8::Local<v8::Context> context,
  v8::Local<v8::Array> array,
  int index
) {
    v8::MaybeLocal<v8::Value> maybeValue = array->Get(context, index); 
    v8::Local<T> value = maybeValue.FromMaybe(
      v8::Local<v8::Value>()
    ).As<T>();
    return value;
}


float *ModelData::fromArrayToFloatAlloc(
  v8::Local<v8::Context> context,
  v8::Local<v8::Array> array
) {
  float *alloc = (float*)malloc(sizeof(float) * array->Length());

  for (int i = 0; i < array->Length(); i++) { 
    alloc[i] = this->getFromArray<v8::Number>(
        context, array, i
    )->Value();
  }

  return alloc;
}

template <class T>
std::vector<T> ModelData::jaggedToAlloc(
    v8::Local<v8::Context> context,
    v8::Local<v8::Array> array
) {
  std::vector<T> ans;

  int i, j;
  for (i = 0; i < array->Length(); i++) {
    v8::Local<v8::Array> array_set = 
      this->getFromArray<v8::Array>(context, array, i); 

    float *row_data = (float*)malloc(sizeof(float) * array_set->Length());
    // Get weight raw matrix
    for (j = 0; j < array_set->Length(); j++) {
      row_data[j] = this->getFromArray<v8::Number>(
        context, array_set, j
      )->Value(); 
    }

    ans.push_back(row_data);
  }

  return ans;
}

template <class T>
std::vector<T> ModelData::toArray(
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

void ModelData::setWeights(int index, float *weight_ptr) {
  this->weights[index] = weight_ptr;
}

ModelData::~ModelData() {
  for (int i = 0; i < this->weights.size(); i++) {
      free(this->weights[i]);
  }

  for (int i = 0; i < this->biases.size(); i++) {
      free(this->biases[i]);
  }

  for (int i = 0; i < this->layers.size(); i++) {
      free(this->layers[i]);
  }

  for (int i = 0; i < this->inputs.size(); i++) {
      free(this->inputs[i]);
  } 

  for (int i = 0; i < this->outputs.size(); i++) {
      free(this->outputs[i]);
  }
}