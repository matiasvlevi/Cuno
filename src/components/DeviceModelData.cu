#include "./DeviceModelData.cuh"


DeviceModelData::DeviceModelData(ModelData* parent) {
  this->epoch = parent->epoch;
  
  this->inputs = parent->inputs; 
  this->outputs = parent->outputs;

  for (int i = 0; i < parent->arch.size(); i++) {
    this->arch.push_back(parent->arch[i]);

    double *l_ptr = 0;
    this->layers.push_back(l_ptr);

    if (i == 0) continue;

    double *w_ptr = 0;
    this->weights.push_back(w_ptr);

    double *b_ptr = 0;
    this->biases.push_back(b_ptr);

    double *e_ptr = 0;
    this->errors.push_back(e_ptr);

    double *g_ptr = 0;
    this->gradients.push_back(g_ptr);
  }
}



DeviceModelData::~DeviceModelData() {
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
  for (int i = 0; i < this->gradients.size(); i++) {
      free(this->gradients[i]);
      this->gradients[i] = NULL;
  } 

  for (int i = 0; i < this->errors.size(); i++) {
      free(this->errors[i]);
      this->errors[i] = NULL;
  }
}
