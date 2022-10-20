#include "./ModelData.cuh"

size_t ModelData::MAX_HOST_ALLOC = 105022000;
size_t ModelData::MAX_DEVICE_ALLOC = 512000000;

ModelData::ModelData() {
  this->epoch = 1;
  this->arch = std::vector<int>();
  this->weights = std::vector<double*>();
  this->biases = std::vector<double*>();
  this->layers = std::vector<double*>();
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

size_t ModelData::getMemoryUsage(std::vector<int> arch) {
  size_t weights_size = 0;
  size_t layers_size = 0;
  size_t arch_size = arch.size() * sizeof(int);

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

void ModelData::logWeights(std::vector<double *> weights) {
  for (int i = 0; i < weights.size(); i++) {
    
    std::cout << "Weights " << i <<
      "  (" << this->arch[i] <<
      "," << this->arch[i+1] <<  ") {";

    for (int j = 0; j < this->arch[i] * this->arch[i+1]; j++) {
    
      if (j % this->arch[i+1] == 0) 
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
  this->logArrayData<double *>(this->inputs, this->arch[0]);

  std::cout << "DATASET --- Output ---\n" << std:: endl;
  this->logArrayData<double *>(this->outputs, this->arch[this->arch.size()-1]);
}

void ModelData::logModel() {
  
  std::cout << "Epoch target specified: \n" << this->epoch << std::endl;

  std::cout << "--- Weights --- \n" << std:: endl;
  this->logWeights(this->weights);

  std::cout << "--- Biases --- \n" << std:: endl;
  this->logArrayAsModelComponent<double *>(this->biases, 1);

  std::cout << "--- Layers ---\n" << std:: endl;
  this->logArrayAsModelComponent<double *>(this->layers);
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
