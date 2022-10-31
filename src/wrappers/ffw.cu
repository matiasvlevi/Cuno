#include "./wrappers.cuh"

void Cuno::Wrappers::ffw(GPUDann<double> *nn, double *input) {
  Log::hostArray<double>(input, nn->arch[0]);
  cudaMemcpy(nn->layers[0], input, nn->arch[0] * sizeof(double), cudaMemcpyHostToDevice);

  Log::deviceArray<double>(nn->biases[0], nn->arch[1]);
  Log::deviceArray<double>(nn->weights[0], nn->arch[0] * nn->arch[1]);

  std::cout << "--- FFW ---\n" ;
  for (int i = 0; i < nn->length-1; i++) {

    Wrappers::matvec_wrap(
       nn->weights[i], nn->layers[i],
       nn->layers[i+1],
       nn->arch[i], nn->arch[i+1]
    );

    std::cout << "Conv: " << i << " Arch:" << nn->arch[i] << "," << nn->arch[i+1] << std::endl; 
    Log::deviceArray<double>(nn->layers[i+1], nn->arch[i+1]);

    Wrappers::add_wrap<double>(nn->layers[i+1], nn->biases[i], nn->arch[i+1]);
    std::cout << "after biases: " << std::endl;
    Log::deviceArray<double>(nn->layers[i+1], nn->arch[i+1]);

    // Wrappers::sigmoid_wrap<double>(nn->layers[i+1], nn->arch[i+1]);
    //   Log::deviceArray<double>(nn->layers[i+1], nn->arch[i+1]);

  }
  std::cout << "--- FFW END --- " << std::endl;
  std::cout << "biases" << std::endl;
  Log::deviceArray<double>(nn->biases[nn->length-2], nn->arch[nn->length-1]);

  std::cout << "output" << std::endl;
  Log::deviceArray<double>(nn->layers[nn->length-1], nn->arch[nn->length-1]);
}
