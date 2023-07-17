#include <Amino/Core/Ptr.h>
#include <Bifrost/Object/Object.h>
#include <Bifrost/Math/Types.h>

#include <Amino/Core/Any.h>
#include <Amino/Core/Array.h>
#include <Amino/Core/String.h>
#include <Amino/Cpp/Annotate.h>
#include <numpy/arrayobject.h>


PyObject* anyToPy(Amino::Any data);
PyObject* bobToDict(Bifrost::Object& bob, Amino::Array<Amino::String> keys);
PyObject* bobToDict(Bifrost::Object& bob, bool properties);


int initToPython();