#include "./logger.cuh"

void Log::out(const char *content) {
  std::cout << content << std::endl;
}

template <class T>
void Log::ptr_arr(T values, unsigned int length) {
  std::cout << "ptr: " << values << " (" << length << ")" << " {\n";
  for (unsigned int i = 0; i < length; i++) {
    std::cout << *(values+i);
    if (i < length-1) std::cout << ",";
  }
  std::cout << "\n}\n" << std::endl;
}

template <class T>
void Log::weights(
  std::vector<T> values,
  std::vector<unsigned int> arch,
  unsigned int index
) {
  std::cout << "ptr: " << values << 
            " (" <<
            arch[index] << "," << arch[index+1] <<  
            ")" << " {\n";


  for (int j = 0; j < arch[index] * arch[index+1]; j++) {
    if (j % arch[index+1] == 0) 
      std::cout << "\n\t";
    else 
      std::cout << ", ";
      
    std::cout << "\x1b[92m" << values[index][j] << "\x1b[0m";
  }

  std::cout << "\n}\n" << std::endl;
}