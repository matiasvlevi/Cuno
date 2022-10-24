#include "../logger.cuh"


template <> 
void Log::deviceArray<double>(double *values, int N) {
    // Allocate temporary stack pointer for log
    double host_values[N];

    // Copy to stack
    cudaMemcpy(
        host_values, values,
        sizeof(double) * N,
        cudaMemcpyDeviceToHost
    );

    std::cout << 
        Log::Header<double>("Device Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
        if (i != 0 && i < N) std::cout << ", ";
        
        std::cout << "\x1b[92m" << host_values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
} 

