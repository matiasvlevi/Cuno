#include "../logger.hpp"

template <>
void Log::hostArray<double>(double *values, int N) {
    std::cout << 
        Log::Header<double>("Host Array", values, N) 
    << " { ";

    if (
      N > 64
    ) {
      for (int i = 0; i < 8; i++) {
          if (i != 0) std::cout << ", ";
          std::cout << "\x1b[92m" << values[i] << "\x1b[0m";
      }
      std::cout << ", ... " << N - 16 << " more values " << " ... "; 
      for (int i = N - 8; i < N; i++) {
          if (i < N) std::cout << ", ";
          std::cout << "\x1b[92m" << values[i] << "\x1b[0m";
      }
    
    } else {
      
      for (int i = 0; i < N; i++) {
          if (i != 0 && i < N) std::cout << ", ";
          std::cout << "\x1b[92m" << values[i] << "\x1b[0m";
      }
    
    }

    std::cout << " }\n" << std::endl;
}

template <>
void Log::hostArray<int>(int *values, int N) {
    std::cout << 
        Log::Header<int>("Host Array", values, N) 
    << " { ";

    if (
      N > 64
    ) {
      for (int i = 0; i < 8; i++) {
          if (i != 0) std::cout << ", ";
          std::cout << "\x1b[92m" << values[i] << "\x1b[0m";
      }
      std::cout << ", ... " << N - 16 << " more values " << " ... "; 
      for (int i = N - 8; i < N; i++) {
          if (i < N) std::cout << ", ";
          std::cout << "\x1b[92m" << values[i] << "\x1b[0m";
      }
    
    } else {
      
      for (int i = 0; i < N; i++) {
          if (i != 0 && i < N) std::cout << ", ";
          std::cout << "\x1b[92m" << values[i] << "\x1b[0m";
      }
    
    }

    std::cout << " }\n" << std::endl;
}

