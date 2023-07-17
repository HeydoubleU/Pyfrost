#include "python.h"
#include "toPython.h"


int initToPython() {
    import_array();
    return 0;
}

// Vector Templates
template <typename T>
PyObject* genericV2ToPy(T vector) {
    auto x = PyFloat_FromDouble(vector.x);
    auto y = PyFloat_FromDouble(vector.y);
    auto py_vector = PyTuple_Pack(2, x, y);
    Py_DECREF(x); Py_DECREF(y);

    return py_vector;
}

template <typename T>
PyObject* genericV3ToPy(T vector) {
    auto x = PyFloat_FromDouble(vector.x);
    auto y = PyFloat_FromDouble(vector.y);
    auto z = PyFloat_FromDouble(vector.x);
    auto py_vector = PyTuple_Pack(3, x, y, z);
    Py_DECREF(x); Py_DECREF(y); Py_DECREF(z);

    return py_vector;
}

template <typename T>
PyObject* genericV4ToPy(T vector) {
    auto x = PyFloat_FromDouble(vector.x);
    auto y = PyFloat_FromDouble(vector.y);
    auto z = PyFloat_FromDouble(vector.x);
    auto w = PyFloat_FromDouble(vector.w);
    auto py_vector = PyTuple_Pack(4, x, y, z, w);
    Py_DECREF(x); Py_DECREF(y); Py_DECREF(z); Py_DECREF(w);

    return py_vector;
}


// Convert Single of type
PyObject* singleToPy(float value) {
    return PyFloat_FromDouble(value);
}

PyObject* singleToPy(Bifrost::Math::float2 value) {
    return genericV2ToPy(value);
}

PyObject* singleToPy(Bifrost::Math::float3 value) {
    return genericV3ToPy(value);
}

PyObject* singleToPy(Bifrost::Math::float4 value) {
    return genericV4ToPy(value);
}

PyObject* singleToPy(long long value) {
    return PyLong_FromLong(value);
}

PyObject* singleToPy(Bifrost::Math::long2 value) {
    return genericV2ToPy(value);
}

PyObject* singleToPy(Bifrost::Math::long3 value) {
    return genericV3ToPy(value);
}

PyObject* singleToPy(Bifrost::Math::long4 value) {
    return genericV4ToPy(value);
}

PyObject* singleToPy(unsigned int value) {
    return PyLong_FromLong(value);
}

PyObject* singleToPy(bool value) {
    if (value)
        return Py_True;
    return Py_False;
}

PyObject* singleToPy(Amino::String value) {
    return PyUnicode_FromString(value.c_str());
}

PyObject* singleToPy(Amino::Ptr<Bifrost::Object> bob) {
    PyObject* dict = PyDict_New();
    auto keys = bob->keys();
    for (int i = 0; i < keys->size(); i++) {
        const char* key = keys->at(i).c_str();
        PyObject* converted = anyToPy(bob->getProperty(key));
        if (converted) {
            PyDict_SetItemString(dict, key, converted);
            Py_DECREF(converted);
        }
    }
    return dict;
}


// Convert Arrays
template <typename T>  // To Python list
PyObject* arrayToPy(T amino_array) {
    PyObject* pyList = PyList_New((int)amino_array.size());
    for (int i = 0; i < amino_array.size(); i++) {
        auto val = singleToPy(amino_array[i]);
        PyList_SetItem(pyList, i, val);
    }
    return pyList;
}

template <typename T>  // To Numpy Array
PyObject* arrayToNp(T amino_array, int vector_dim, int np_type) {
    int array_dims = 1;
    npy_intp dims[2];
    dims[0] = amino_array->size();

    if (vector_dim) {
        array_dims = 2;
        dims[1] = vector_dim;
    }
    
    void* data = const_cast<void*>(static_cast<const void*>(amino_array->data()));
    PyObject* npArray = PyArray_SimpleNewFromData(array_dims, dims, np_type, data);
    return npArray;
}



// This function takes any data, and tries casting first to single then to an array of type T
// Specififying the extra args indicates we want to convert to numpy instead of list (for arrays)
template <typename T>
PyObject* tryCastToPy(Amino::Any data, int vector_dim = -1, int np_type = NPY_FLOAT) {
    auto payload = Amino::any_cast<T>(&data);
    if (payload != nullptr) {
        return singleToPy(*payload);
    }

    auto a_payload = Amino::any_cast<Amino::Ptr<Amino::Array<T>>>(data);
    if (a_payload != nullptr) {
        if (vector_dim < 0)
            return arrayToPy(*a_payload);
        else
            return arrayToNp(a_payload, vector_dim, np_type);
    }

    return NULL;
}



// High level conversion function, this will take any bifrost data and convert to a comparable
// Unsupported types return Python None object.
PyObject* anyToPy(Amino::Any data) {
    PyObject* py_obj;

    py_obj = tryCastToPy<float>(data, 0, NPY_FLOAT);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::float2>(data, 2, NPY_FLOAT);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::float3>(data, 3, NPY_FLOAT);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::float4>(data, 4, NPY_FLOAT);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<long long>(data, 0, NPY_LONGLONG);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::long2>(data, 2, NPY_LONGLONG);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::long3>(data, 3, NPY_LONGLONG);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::long4>(data, 4, NPY_LONGLONG);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<unsigned int>(data, 0, NPY_UINT);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<bool>(data, 0, NPY_BOOL);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Amino::String>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Amino::Ptr<Bifrost::Object>>(data);
    if (py_obj) return py_obj;

    return Py_None;
}


// Used for input object conversion.
// As opposed to 'singleToPy', this function allows us to specify which properties we want to convert (also verifying they exist)
PyObject* bobToDict(Bifrost::Object& bob, Amino::Array<Amino::String> keys) {
    PyObject* dict = PyDict_New();
    for (int i = 0; i < keys.size(); i++) {
        const char* key = keys.at(i).c_str();
        if (bob.hasProperty(key)) {
            PyObject* converted = anyToPy(bob.getProperty(key));
            if (converted) {
                PyDict_SetItemString(dict, key, converted);
                Py_CLEAR(converted);
            }
        }
    }
    return dict;
}

// This overload is for when the properties args is a bool, in which case we can just use 'singleToPy'.
//PyObject* bobToDict(Bifrost::Object& bob, bool properties) {
//    if (properties)
//        return singleToPy(&bob);
//    else
//        return PyDict_New();
//}

PyObject* bobToDict(Bifrost::Object& bob, bool properties) {
    PyObject* dict = PyDict_New();
    if (!properties) return dict;

    auto keys = *bob.keys();
    for (int i = 0; i < keys.size(); i++) {
        const char* key = keys.at(i).c_str();
        PyObject* converted = anyToPy(bob.getProperty(key));
        if (converted) {
            PyDict_SetItemString(dict, key, converted);
            Py_CLEAR(converted);
        }
    }
    return dict;
}