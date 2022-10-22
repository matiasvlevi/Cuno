#include <iostream>
#include <sstream>
#include "../cuno.cuh"

#ifndef LOGGER_H
#define LOGGER_H
namespace Cuno {

namespace Log {

  /**
    Output a Header in the stream
  */  
  template <class T>
  class Header {
    public:
      Header(
          std::string title,
          T *ptr, int r, int c = 0
      ) : title_(title), ptr_(ptr), r_(r), c_(c) {}
      friend std::ostream& operator<<(std::ostream& os, const Header& mp) 
      {
          std::stringstream dim;
          if (mp.c_ == 0) {
            dim << " (\x1b[94m" << mp.r_ << "\x1b[0m)";
          } else {
            dim << " (\x1b[94m" << mp.r_ << "\x1b[0m, \x1b[94m" << mp.c_ <<  "\x1b[0m)";
          }

          os << "\x1b[92m" << "[ "<< mp.title_ <<" ] " << "\x1b[0m\n" <<
          "ptr: " << "\x1b[93m" << mp.ptr_ << "\x1b[0m" << dim.str();
          return os;
      }
    private:
      std::string title_;
      T *ptr_;
      int r_, c_;
  };

  template <class T>
  void deviceMatrix(T *values, int R, int C) {
    // Allocate temporary stack pointer for log
    T host_values[R * C];

    // Copy to stack
    cudaMemcpy(
        host_values, values,
        sizeof(T) * R * C,
        cudaMemcpyDeviceToHost
    );

    std::cout << 
      Header<T>("Device Matrix", values, R, C) 
    << " {";

    for (int j = 0; j < R * C; j++) {
      if (j % C == 0) std::cout << "\n\t";
      else            std::cout << ", ";
        
      std::cout << "\x1b[92m" << host_values[j] << "\x1b[0m";
    }

    std::cout << "\n}\n" << std::endl;
  }

  template <class T>
  void deviceArray(T *values, int N) {
    // Allocate temporary stack pointer for log
    T host_values[N];

    // Copy to stack
    cudaMemcpy(
        host_values, values,
        sizeof(T) * N,
        cudaMemcpyDeviceToHost
    );

    std::cout << 
      Header<T>("Device Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
      if (i != 0 && i < N) std::cout << ", ";
        
      std::cout << "\x1b[92m" << host_values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
  } 

  template <class T>
  void hostMatrix(T *values, int R, int C) {
    std::cout << 
      Header<T>("Host Matrix", values, R, C) 
    << " {";

    for (int j = 0; j < R * C; j++) {
      if (j % C == 0) std::cout << "\n\t";
      else            std::cout << ", ";
        
      std::cout << "\x1b[96m" << values[j] << "\x1b[0m";
    }

    std::cout << "\n}\n" << std::endl;
  }

  template <class T>
  void hostArray(T *values, int N) {
    std::cout << 
      Header<T>("Host Array", values, N) 
    << " { ";

    for (int i = 0; i < N; i++) {
      if (i != 0 && i < N) std::cout << ", ";
        
      std::cout << "\x1b[96m" << values[i] << "\x1b[0m";
    }

    std::cout << " }\n" << std::endl;
  } 

};

};
#endif
