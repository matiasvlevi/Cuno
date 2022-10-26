#include "../logger.cuh"


template <> 
void Log::deviceArray<double>(double *values, int N) {
    // Allocate temporary stack pointer for log
    double *host_values = (double*)malloc(sizeof(double) * N);

    // Copy to stack
    cudaMemcpy(
        host_values, values,
        sizeof(double) * N,
        cudaMemcpyDeviceToHost
    );

    std::cout << 
        Log::Header<double>("Device Array", values, N) 
    << " { ";
    if (
      N > 32
    ) {
      for (int i = 0; i < 8; i++) {
          if (i != 0) std::cout << ", ";
          std::cout << "\x1b[92m" << host_values[i] << "\x1b[0m";
      }
      std::cout << ", ... " << N - 16 << " more values " << " ... "; 
      for (int i = N - 8; i < N; i++) {
          if (i < N) std::cout << ", ";
          std::cout << "\x1b[92m" << host_values[i] << "\x1b[0m";
      }
    
    } else {
      
      for (int i = 0; i < N; i++) {
          if (i != 0 && i < N) std::cout << ", ";
          std::cout << "\x1b[92m" << host_values[i] << "\x1b[0m";
      }
    
    }
    std::cout << " }\n" << std::endl;
    free(host_values);
} 

