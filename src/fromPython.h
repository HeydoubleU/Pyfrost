#include <Amino/Core/Ptr.h>
#include <Bifrost/Object/Object.h>
#include <Bifrost/Math/Types.h>

#include <Amino/Core/Any.h>
#include <Amino/Core/Array.h>
#include <Amino/Core/String.h>
#include <Amino/Cpp/Annotate.h>
#include <numpy/arrayobject.h>

Amino::Any anyFromPy(PyObject* py_obj);

namespace FromPython {
    void mergeBobWithDict(Amino::MutablePtr<Bifrost::Object>& bob, PyObject* py_obj);
    Amino::Ptr<Bifrost::Object> toSimple(PyObject* py_obj);
}

int initFromPython();