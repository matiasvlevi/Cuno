#include <node/node.h>
#include <iostream>

#include "v8/conversions.hpp"

template<typename T>
void printBuffer(T *buffer, uint32_t length) {
	for (int i = 0; i < length; i++) {
		std::cout << buffer[i];
		if (i < length-1) 
			std::cout << ',';
	}
	std::cout << std::endl;
}

void printArray(const v8::FunctionCallbackInfo<v8::Value> &args) {
	v8::Isolate* env = args.GetIsolate();
	v8::Local<v8::Context> context = env->GetCurrentContext();

	// Get first argument as an v8::Array
	v8::Local<v8::Array> array = args[0].As<v8::Array>();

	// Allocate buffer with the same length as the array
	double *buffer = (double*)malloc(sizeof(double) * array->Length());

	// Fill buffer with the array values 
	Cuno::Error_t err = 
		Cuno::ArrayToBuffer(context, array, buffer);

	if (err) {
		std::cout << "Error: " << err << std::endl;

		return;
	}

	// Print the buffer
	printBuffer(buffer, array->Length());	

	// Free the buffer
	free(buffer);
}

void printMatrix(const v8::FunctionCallbackInfo<v8::Value> &args) {
	v8::Isolate* env = args.GetIsolate();
	v8::Local<v8::Context> context = env->GetCurrentContext();

	// Get first argument as an v8::Array
	v8::Local<v8::Array> matrix = args[0].As<v8::Array>();

	// Get matrix dimensions
	uint32_t rowCount = matrix->Length();
	uint32_t columnCount = matrix->Get(context, 0)
							.ToLocalChecked()
							.As<v8::Array>()->Length();

	// Allocate buffer with the same length as the array
	double *buffer = (double*)malloc(sizeof(double) * rowCount * columnCount);

	// Fill buffer with the array values 
	Cuno::Error_t err = 
		Cuno::MatrixToBuffer(context, matrix, buffer);

	if (err) {
		std::cout << "Error: " << err << std::endl;

		return;
	}

	// Print the buffer
	printBuffer(buffer, rowCount * columnCount);	

	// Free the buffer
	free(buffer);
}

void Init(v8::Local<v8::Object> exports) {
	NODE_SET_METHOD(exports, "printArray", printArray);
	NODE_SET_METHOD(exports, "printMatrix", printMatrix);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, Init)
