#include "../logger.cuh"

template <>
void Log::deviceMatrix<double>(double *values, int R, int C) {
    // Allocate temporary stack pointer for log
    double host_values[R * C];

    // Copy to stack
    cudaMemcpy(
        host_values, values,
        sizeof(double) * R * C,
        cudaMemcpyDeviceToHost
    );

    std::cout << 
        Log::Header<double>("Device Matrix", values, R, C) 
    << " {";

    for (int j = 0; j < R * C; j++) {
        if (j % C == 0) std::cout << "\n\t";
        else            std::cout << ", ";

        std::cout << "\x1b[92m" << host_values[j] << "\x1b[0m";
    }

    std::cout << "\n}\n" << std::endl;
}
