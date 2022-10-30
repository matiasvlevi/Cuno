#include "../v8utils.cuh"

template <>
Cuno::MethodInput<double> *Cuno::v8Utils::getSingleCallArgs(
  const Local<Context> context,
  const FunctionCallbackInfo<Value>& args
) {
  // Abort if arguments are not arrays
  for (int i = 0; i < 2; i++) if (!args[i]->IsArray()) {
    // SHOULD ABORT IF ARRAYS DO NOT CONTAIN ARRAYS WITH THE SAME SIZE
    return 0;
  }

  // Get computation dimensions from given Matrix Jagged Arrays 
  int M = args[0].As<Array>()->Length();
  int N = v8Utils::getFromArray<Array>(context, args[0].As<Array>(), 0)->Length();
  int P = v8Utils::getFromArray<Array>(context, args[1].As<Array>(), 0)->Length();

  // Allocate device memory
  MethodInput<double> *input = new MethodInput<double>(M, N, P);

  // Allocate temporary stack pointers
  double a[M * N];
  double b[N * P];

  for (int k = 0; k < 2; k++) {
    // Get the Matrix Jagged Array from v8 values
    Local<Array> matrix = args[k].As<Array>();

    for (int i = 0; i < matrix->Length(); i++) {
      // Get a row from the Matrix Jagged Array
      Local<Array> currentRow = 
        v8Utils::getFromArray<Array>(context, matrix, i);
  
      for (int j = 0; j < currentRow->Length(); j++) {
        // Get the value from the Matrix Jagged Array
        double value = 
          v8Utils::getFromArray<Number>(context, currentRow, j)->Value();

        // Allocate stack matrix pointers
        if (k % 2 == 0) a[i * currentRow->Length() + j] = value; 
        else b[i * currentRow->Length() + j] = value; 

      }
    }
  }

  input->toDevice(a, b);
  return input;
};