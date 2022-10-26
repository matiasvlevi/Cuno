#include <iostream>
#include <sstream>
#include "../cuno.cuh"



#ifndef LOG_H
#define LOG_H

namespace Log {

  template <class T>
  class Header {
    public:
      Header(
          std::string title,
          T *ptr, int r, int c = 0
      ) : title_(title), ptr_(ptr), r_(r), c_(c) {}
   
     friend std::ostream& operator<<(std::ostream& os, const Header& mp) {
          std::stringstream dim;
          if (mp.c_ == 0) dim << " (\x1b[94m" << mp.r_ << "\x1b[0m)";
          else dim << " (\x1b[94m" << mp.r_ << "\x1b[0m, \x1b[94m" << mp.c_ <<  "\x1b[0m)";
        
          os << "\x1b[92m" << "[ "<< mp.title_ <<" ] " << "\x1b[0m\n" <<
          "ptr: " << "\x1b[93m" << mp.ptr_ << "\x1b[0m" << dim.str();
          return os;
      }
    private:
      std::string title_;
      T *ptr_;
      int r_, c_;
  };

  template <typename T>
  void deviceMatrix(T *values, int R, int C);

  template <typename T>
  void deviceArray(T *values, int N);

  template <typename T>
  void hostMatrix(T *values, int R, int C);

  template <typename T>
  void hostArray(T *values, int N); 
};

#endif
