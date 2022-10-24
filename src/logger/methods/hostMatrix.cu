#include "../logger.cuh"


template <>
void Log::hostMatrix<double>(double *values, int R, int C) {
    std::cout << 
        Log::Header<double>("Host Matrix", values, R, C) 
    << " {";

    for (int j = 0; j < R * C; j++) {
        if (j % C == 0) std::cout << "\n\t";
        else            std::cout << ", ";
        
        std::cout << "\x1b[96m" << values[j] << "\x1b[0m";
    }

    std::cout << "\n}\n" << std::endl;
}

