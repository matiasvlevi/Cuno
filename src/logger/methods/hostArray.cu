#include "../logger.cuh"


template <>
void Log::hostArray<double>(double *values, int N) {
    std::cout << 
        Log::Header<double>("Host Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
        if (i != 0 && i < N) std::cout << ", ";
        
        std::cout << "\x1b[96m" << values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
}

template <>
void Log::hostArray<int>(int *values, int N) {
    std::cout << 
        Log::Header<int>("Host Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
        if (i != 0 && i < N) std::cout << ", ";
        
        std::cout << "\x1b[96m" << values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
}

