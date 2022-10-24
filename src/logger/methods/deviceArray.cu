#include "../logger.cuh"


template <class T> 
void Log::deviceArray(T *values, int N) {
    // Allocate temporary stack pointer for log
    T host_values[N];

    // Copy to stack
    cudaMemcpy(
        host_values, values,
        sizeof(T) * N,
        cudaMemcpyDeviceToHost
    );

    std::cout << 
        Log::Header<T>("Device Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
        if (i != 0 && i < N) std::cout << ", ";
        
        std::cout << "\x1b[92m" << host_values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
} 

