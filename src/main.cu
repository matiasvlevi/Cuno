#include "./kernel.cuh"
#include <node/node.h>

void kernelAddWrapper(
    float *input,
    float *c,
    int N
) {

  float *dev_input = 0;
  float *dev_c = 0;
  size_t bufSize = sizeof(float) * N;

  for (int i = 0; i < 6; i++) std::cout << input[i] << std::endl; 
  
  cudaMalloc(&dev_input, bufSize); 
  cudaMalloc(&dev_c, bufSize); 
   
  cudaMemcpy(dev_input, input, bufSize * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, bufSize, cudaMemcpyHostToDevice);
 
 	dim3 THREADS;
 	THREADS.x = 32;
 	THREADS.y = 32;
 
 	int blocks = (N + THREADS.x - 1) / THREADS.x;
 
 	dim3 BLOCKS;
 	BLOCKS.x = blocks;
 	BLOCKS.y = blocks;

  Kernel::add<<<BLOCKS, THREADS>>>(dev_input, dev_c, N);

  cudaMemcpy(c, dev_c, bufSize, cudaMemcpyDeviceToHost);
  

  cudaFree(dev_input);
  cudaFree(dev_c);

}

class Args {
public:
  float *data;
  int length;
  Args(int len) {
    this->data = (float*)malloc(sizeof(float) * 2 * len);
    this->length = len;
  }
  Args() {
    free(this->data);
  }
};

Args *convertArgs(
    const v8::Local<v8::Context> context,
    const v8::FunctionCallbackInfo<v8::Value>& args
) {
  
  if (!args[0]->IsArray()) return NULL;
  v8::Local<v8::Array> test = (args[0]).As<v8::Array>();
  int length = test->Length();
 
  Args *output = new Args(length);
  
  int i, j;
  for (i = 0; i < 2; i++) {
    v8::Local<v8::Array> test = (args[i]).As<v8::Array>();

    for (j = 0; j < output->length; j++) {       
    
        v8::MaybeLocal<v8::Value> maybeValue = (test->Get(context, j)); 
        

        v8::Local<v8::Number> value = maybeValue.FromMaybe(v8::Local<v8::Value>()).As<v8::Number>();
        output->data[i * output->length + j] = value->Value();
    }
  }

  return output;

}

void Method(const v8::FunctionCallbackInfo<v8::Value>& args) {
  v8::Isolate* env = args.GetIsolate();
  v8::Local<v8::Context> context = env->GetCurrentContext();

  Args *input = convertArgs(context, args); 

  float c[input->length] = {};


  kernelAddWrapper(input->data, c, input->length);

  // Convert to JS Array
  v8::Local<v8::Array> buffer = v8::Array::New(env, input->length);

  for (int i = 0; i < input->length; i++) {
    buffer->Set(context, i, v8::Number::New(env, *(c+i)));
  }
  
  args.GetReturnValue().Set(buffer);
}



void Initialize(v8::Local<v8::Object> exports) {
  NODE_SET_METHOD(exports, "hello", Method);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, Initialize)

