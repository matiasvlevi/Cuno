#include "./error.hpp"

void Error::throw_(const char* message) {
  std::cout << message << std::endl;
  std::cout << "------- ERROR ------" << std::endl;
}