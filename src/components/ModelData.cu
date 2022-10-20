#include "./ModelData.cuh"

size_t ModelData::MAX_HOST_ALLOC = 105022000;
size_t ModelData::MAX_DEVICE_ALLOC = 512000000;

ModelData::ModelData() {
  this->epoch = 1;
  this->arch = std::vector<unsigned int>();
  this->weights = std::vector<double*>();
  this->biases = std::vector<double*>();
  this->layers = std::vector<double*>();
}

ModelData::ModelData(
    v8::Isolate *env,
    const v8::Local<v8::Context> context,
    const v8::FunctionCallbackInfo<v8::Value>& args
) {
  
  // Abort if no array specified
  for (int i = 0; i < 3; i++) 
    if (!args[i]->IsObject()) return; 

  Allocate(
    env, context,
    args[0].As<v8::Array>(),
    args[1].As<v8::Array>(),
    args[2].As<v8::Array>(),
    args[3].As<v8::Array>(),
    args[4].As<v8::Number>()
  );;

}


void ModelData::Allocate(
  v8::Isolate *env,
  v8::Local<v8::Context> context,
  v8::Local<v8::Array> arch,
  v8::Local<v8::Array> weights,
  v8::Local<v8::Array> biases,
  v8::Local<v8::Array> data,
  v8::Local<v8::Number> epoch
) {
 
  // Get architecture
  this->arch = this->toArray<unsigned int>(context, arch);


  size_t data_size = data->Length() * (this->arch[0] + this->arch[this->arch.size()-1]);
  size_t model_size = ModelData::getMemoryUsage(this->arch);
 
  ModelData::debugMemory(model_size, data_size);
  
  if (
      model_size + data_size >= ModelData::MAX_HOST_ALLOC ||
      model_size + data_size >= ModelData::MAX_DEVICE_ALLOC
  ) {
    ModelData::debugMemory(model_size, data_size);
    return;
  }; 
  // ABORT IF NOT ENOUGH DEDICATED MEMORY
  


  this->epoch = epoch->Value();

  // Allocate Data
  for (int i = 0; i < data->Length(); i++) {
    // Get both properties of data sample
    v8::Local<v8::Array> array_set =this->getFromArray<v8::Array>(
        context, data, i
    );

    // Allocate property input from data sample
    v8::Local<v8::Array> inputs = this->getFromArray<v8::Array>(
      context, array_set, 0
    );
    this->inputs.push_back(this->fromArrayToFloatAlloc(context, inputs));

    // Allocate property output from data sample
    v8::Local<v8::Array> outputs = this->getFromArray<v8::Array>(
      context, array_set, 1
    );
    this->outputs.push_back(this->fromArrayToFloatAlloc(context, outputs));
  }

  // Allocate layers & Init layers
  for (int i = 0; i < this->arch.size(); i++) {
    double *layer_alloc = (double*)malloc(sizeof(double) * this->arch[i]);

    // Init layers
    for (int j = 0; j < this->arch[i]; j++) {
      layer_alloc[j] = 0;
    }

    this->layers.push_back(layer_alloc);
  }

  // Allocate biases
  this->biases = this->jaggedToAlloc<double*>(context, biases); 

  // Allocate weights
  this->weights = this->jaggedToAlloc<double*>(context, weights);
}

size_t ModelData::getMemoryUsage(std::vector<unsigned int> arch) {
  size_t weights_size = 0;
  size_t layers_size = 0;
  size_t arch_size = arch.size() * sizeof(unsigned int);

  for (int i = 0; i < arch.size(); i++) {
    layers_size += arch[i] * sizeof(double);

    if (i == 0) continue;
    weights_size += arch[i-1] * arch[i] * sizeof(double);
  }

  return weights_size + (4 * layers_size) + arch_size;
} 

void ModelData::debugMemory(const size_t model_size, const size_t data_size) {
  std::cout << "Model Size: " << (model_size) << " B\n";  
  std::cout << "Data  Size: " << (data_size) << " B\n";
  
  std::cout << "Total     : " 
    << (model_size + data_size) << " B\n"
    << "MAX(HOST)   : " << ModelData::MAX_HOST_ALLOC << " B\n" 
    << "MAX(DEVICE) : " << ModelData::MAX_DEVICE_ALLOC << " B\n" 
    << std::endl;  

} 

void ModelData::logData() {
  std::cout << "DATASET --- Inputs ---\n" << std:: endl;
  for (int i = 0; i < this->inputs.size(); i++)
    this->ptr_arr<double*>(this->inputs[i], this->arch[0]);

  std::cout << "DATASET --- Output ---\n" << std::endl;
  for (int i = 0; i < this->outputs.size(); i++)
    this->ptr_arr<double*>(this->outputs[i], this->arch[this->arch.size()-1]);
}

void ModelData::logModel() {
  
  std::cout << "Epoch target specified: \n" << this->epoch << std::endl;

  std::cout << "\nWeights\n" << std::endl;
  for (int i = 0; i < this->arch.size(); i++)
    Log::weights<double*>(this->weights, this->arch, i);

  std::cout << "\nBiases\n" << std::endl;
  for (int i = 0; i < this->arch.size(); i++)
    this->ptr_arr<double*>(this->biases[i], this->arch[i]);

  std::cout << "\nLayers\n" << std::endl;
  for (int i = 0; i < this->arch.size(); i++)
    this->ptr_arr<double*>(this->layers[i], this->arch[i]);
}

void ModelData::logLayers() {
  std::cout << "\nLayers\n" << std:: endl;

  for (int i = 0; i < this->arch.size(); i++)
    this->ptr_arr<double*>(this->layers[i], this->arch[i]);
}

template <class T>
void ModelData::ptr_arr(T values, unsigned int length) {
  std::cout << "ptr: \x1b[93m" << values << "\x1b[0m (" << length << ")" << " { ";
  for (unsigned int i = 0; i < length; i++) {
    std::cout << "\x1b[92m" << *(values+i);
    if (i < length-1) std::cout << "\x1b[0m, ";
  }
  std::cout << " \x1b[0m}\n" << std::endl;
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


double *ModelData::fromArrayToFloatAlloc(
  v8::Local<v8::Context> context,
  v8::Local<v8::Array> array
) {
  double *alloc = (double*)malloc(sizeof(double) * array->Length());

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

    double *row_data = (double*)malloc(sizeof(double) * array_set->Length());
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

void ModelData::setWeights(int index, double *weight_ptr) {
  this->weights[index] = weight_ptr;
}

ModelData::~ModelData() {
  for (int i = 0; i < this->weights.size(); i++) {
      free(this->weights[i]);
      this->weights[i] = NULL;
  }

  for (int i = 0; i < this->biases.size(); i++) {
      free(this->biases[i]);
      this->biases[i] = NULL;
  }

  for (int i = 0; i < this->layers.size(); i++) {
      free(this->layers[i]);
      this->layers[i] = NULL;
  }

  for (int i = 0; i < this->inputs.size(); i++) {
      free(this->inputs[i]);
      this->inputs[i] = NULL;
  } 

  for (int i = 0; i < this->outputs.size(); i++) {
      free(this->outputs[i]);
      this->outputs[i] = NULL;
  }
}
