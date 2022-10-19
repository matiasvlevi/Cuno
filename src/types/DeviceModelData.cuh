#include "./ModelData.cuh"

#ifndef DEVICEMODELDATA_H
#define DEVICEMODELDATA_H

class DeviceModelData: public ModelData
{
public:
    std::vector<float*> errors;
    std::vector<float*> gradients;

    DeviceModelData(ModelData* parent);
};

#endif
