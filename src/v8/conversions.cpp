#include <v8.h>

#include "./conversions.hpp"

Cuno::Error_t Cuno::ArrayToBuffer(
	v8::Local<v8::Context> context,
	v8::Local<v8::Array> array,
	double *buffer) 
{
	for (uint32_t i = 0; i < array->Length(); i++) {

		// Get value object
 		v8::Local<v8::Value> value = 
 			array->Get(context, i).ToLocalChecked();
 
		if (!value->IsNumber()) return NaN; 

		// Store value in buffer
 		buffer[i] = value->NumberValue(context).FromJust();
 
	}

	return Success;
}

Cuno::Error_t Cuno::MatrixToBuffer(
	v8::Local<v8::Context> context,
	v8::Local<v8::Array> matrix,
	double *buffer) 
{
	for (uint32_t i = 0; i < matrix->Length(); i++) {

		// Get array object (row)
 		v8::Local<v8::Array> row = 
 			matrix->Get(context, i).ToLocalChecked().As<v8::Array>();

		if (!row->IsArray()) return NaArray;

		for (uint32_t j = 0; j < row->Length(); j++) {

			// Get value object
	 		v8::Local<v8::Value> value = 
				row->Get(context, j).ToLocalChecked();

			if (!value->IsNumber()) return NaN; 
			

			// Store value in buffer
			buffer[i * matrix->Length() + j] = value->NumberValue(context).FromJust();
		}
 
	}

	return Success;
}


