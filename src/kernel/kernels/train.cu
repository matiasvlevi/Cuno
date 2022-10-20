#include "../kernel.cuh"

void Kernel::TrainWrapper(
    ModelData *host,
    DeviceModelData *device
) {

  // Allocate memory 
  for (int i = 0; i < host->arch.size(); i++) { 
    cudaMalloc(&(device->layers[i]), host->arch[i] * sizeof(double));
  }

  for (int i = 1; i < host->arch.size(); i++) {
    cudaMalloc(&(device->biases[i-1]), host->arch[i] * sizeof(double));
    cudaMalloc(&(device->errors[i-1]), host->arch[i] * sizeof(double));
    cudaMalloc(&(device->gradients[i-1]), host->arch[i] * sizeof(double));
    cudaMalloc(&(device->weights[i-1]), host->arch[i-1] * host->arch[i] * sizeof(double));
  }

  // Copy data in devices's input layer
  cudaMemcpy(
    device->layers[0],
    host->inputs[0],
    host->arch[0] * sizeof(double),
    cudaMemcpyHostToDevice
  );
 
  // Copy biases & weights to the device
  for (int i = 0; i < device->arch.size()-1; i++) {
    cudaMemcpy(
        device->biases[i], host->biases[i],
        host->arch[i] * sizeof(double),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        device->weights[i], host->weights[i],
        host->arch[i] * host->arch[i+1] * sizeof(double),
        cudaMemcpyHostToDevice
    );
  }

  // Kernel --- 

  dim3 THREADS;
  THREADS.x = 32;
  THREADS.y = 32;

  int blocks = (host->arch[1] + THREADS.y - 1) / THREADS.y;

  dim3 BLOCKS;
  BLOCKS.x = blocks;
  BLOCKS.y = blocks;

  Kernel::dotOpt<<<BLOCKS, THREADS>>>(
      device->weights[0], device->layers[0],
      device->layers[1],
      host->arch[0],
      host->arch[1], 1
  );

  // ---

  // Copy result back to host
  cudaMemcpy(
    host->layers[1], device->layers[1],
    host->arch[1] * sizeof(double),
    cudaMemcpyDeviceToHost
  );

  // Log debug
  host->logLayers();

  // deallocate layers
  for (int i = 0; i < device->arch.size(); i++) { 
    cudaFree(device->layers[i]);
  }

  // deallocate other values
  for (int i = 0; i < device->arch.size(); i++) {
    cudaFree(device->biases[i]);
    cudaFree(device->errors[i]);
    cudaFree(device->gradients[i]);
    cudaFree(device->weights[i]);
  }

}
