/**
 * @file conversions.hpp
 * @brief 
 * Contains utilites to convert from v8 Objects to various general types used internally
 *
 * @date 2023-05-12
 */
#ifndef V8_CONVERSIONS_HPP
#define V8_CONVERSIONS_HPP

/**
 * v8 Forward Declarations
 */
namespace v8 {
	template <typename T>
	class Local;
	class Context;	
	class Array;
};

namespace Cuno {
	enum Error_t {
		Success,
		NaN,
		NaArray
	};

	/**
	 * Fill a buffer with 1 dimensional array objects
	 * 
	 * @param context - The v8 context
	 * @param matrix  - An array containing numbers
	 * @param buffer  - The pre-allocated buffer to fill
	 */ 
	Error_t ArrayToBuffer(
		v8::Local<v8::Context> context,
		v8::Local<v8::Array> array,
		double *buffer);

	/**
	 * Fill a buffer with 2 dimensional array objects
	 * 
	 * @param context - The v8 context
	 * @param matrix  - An array containing arrays 
	 * @param buffer  - The pre-allocated buffer to fill
	 */ 
	Error_t MatrixToBuffer(
		v8::Local<v8::Context> context,
		v8::Local<v8::Array> matrix,
		double *buffer);

};

#endif

