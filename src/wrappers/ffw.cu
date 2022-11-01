#include "./wrappers.cuh"

void Cuno::Wrappers::ffw(GPUDann<double> *nn, double *input) {
  cudaMemcpy(
    nn->layers[0],
    input,
    nn->arch[0] * sizeof(double),
    cudaMemcpyHostToDevice
  );

  for (int i = 0; i < nn->length-1; i++) {

    Wrappers::matvec_wrap(
       nn->weights[i], nn->layers[i],
       nn->layers[i+1],
       nn->arch[i], nn->arch[i+1]
    );
    Wrappers::add_wrap(nn->layers[i+1], nn->biases[i], nn->arch[i+1]);
    Wrappers::sigmoid_wrap(nn->layers[i+1], nn->arch[i+1]);

  }
}
