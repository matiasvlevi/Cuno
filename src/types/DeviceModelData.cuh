#include "./ModelData.cuh"

#ifndef DEVICEMODELDATA_H
#define DEVICEMODELDATA_H

class DeviceModelData: public ModelData
{
public:
    std::vector<double*> errors;
    std::vector<double*> gradients;

    DeviceModelData(ModelData* parent);
    ~DeviceModelData();
};

#endif
