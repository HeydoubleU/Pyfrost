#include <Amino/Core/Ptr.h>
#include <Bifrost/Object/Object.h>
#include <Bifrost/Math/Types.h>

#include <Amino/Core/Any.h>
#include <Amino/Core/Array.h>
#include <Amino/Core/String.h>
#include <Amino/Cpp/Annotate.h>
#include <numpy/arrayobject.h>


int initToPython();
namespace ToPython {
	PyObject* fromSimple(Amino::Ptr<Bifrost::Object> bob);
}
PyObject* anyToPy(Amino::Any data);