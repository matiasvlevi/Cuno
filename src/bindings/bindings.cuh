#include "../wrappers/wrappers.cuh"

#ifndef BINDINGS_H
#define BINDINGS_H
namespace Cuno {
using namespace Cuno;

namespace Bindings {
     void Init(v8::Local<v8::Object> exports); 
};

};
#endif
