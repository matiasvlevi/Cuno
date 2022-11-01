#include <iostream>
#include <sstream>
#include "../cuno.cuh"

#ifndef LOG_H
#define LOG_H
namespace Log {

  /**
  * log a header containing meta data
  *
  * @param[in] Title The header title
  * @param[in] ptr   The device or host pointer
  * @param[in] r     The row dimension
  * @param[in] c     The col dimension
  */ 
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

  /**
  * log a T device buffer as a Matrix
  *
  * @param[in] values The T device buffer
  * @param[in] R      The Row dimension
  * @param[in] C      The Column dimension
  */ 
  template <typename T>
  void deviceMatrix(T *values, int R, int C);

  /**
  * log a T device buffer as an Array
  *
  * @param[in] values The T device buffer
  * @param[in] N      The length
  */ 
  template <typename T>
  void deviceArray(T *values, int N);

  /**
  * log a T host buffer as a Matrix
  *
  * @param[in] values The T host buffer
  * @param[in] R      The Row dimension
  * @param[in] C      The Column dimension
  */
  template <typename T>
  void hostMatrix(T *values, int R, int C);

  /**
  * log a T host buffer as an Array
  *
  * @param[in] values The T host buffer
  * @param[in] N      The length
  */ 
  template <typename T>
  void hostArray(T *values, int N); 

};
#endif
