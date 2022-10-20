#include <iostream>
#include <node/node.h>

#include "./types/methodInput.cuh"


#ifndef LOG_H
#define LOG_H

namespace Log {

  void out(const char *content);

  template <class T>
  void ptr_arr(
    T values,
    unsigned int length
  );

  template <class T>
  void weights(
    std::vector<T> weights,
    std::vector<unsigned int> arch,
    unsigned int index
  );



};
#endif
