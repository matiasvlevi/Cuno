#include "./DeviceModelData.cuh"


DeviceModelData::DeviceModelData(ModelData* parent) {
  this->epoch = parent->epoch;
  
  this->inputs = parent->inputs; 
  this->outputs = parent->outputs;

  for (int i = 0; i < parent->arch.size(); i++) {
    this->arch.push_back(parent->arch[i]);

    float *l_ptr = 0;
    this->layers.push_back(l_ptr);

    if (i == 0) continue;

    float *w_ptr = 0;
    this->weights.push_back(w_ptr);

    float *b_ptr = 0;
    this->biases.push_back(b_ptr);

    float *e_ptr = 0;
    this->errors.push_back(e_ptr);

    float *g_ptr = 0;
    this->gradients.push_back(g_ptr);
  }
}



