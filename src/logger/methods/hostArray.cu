#include "../logger.cuh"


template <class T>
void Log::hostArray(T *values, int N) {
    std::cout << 
        Log::Header<T>("Host Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
        if (i != 0 && i < N) std::cout << ", ";
        
        std::cout << "\x1b[96m" << values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
}

