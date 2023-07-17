#include "python.h"
#include "fromPython.h"


int initFromPython() {
    import_array();
    return 0;
}

// Prototypes
template <typename T>
T singleFromPy(PyObject& py_value);

void setPropertyFromDictItem(Bifrost::Object& bob, PyObject& py_key, PyObject& py_value);



void addDictToBob(Bifrost::Object& bob, PyObject& dict) {
    // Used for converting dict to Object. its used in on overload of 'singleFromPy', where an empty object is created for the input.
    // This is also used in the operator to add properties the user's input Object, hence why its separate

    PyObject* keys = PyDict_Keys(&dict);
    PyObject* values = PyDict_Values(&dict);
    for (int i = 0; i < PyList_GET_SIZE(keys); i++) {
        setPropertyFromDictItem(bob, *PyList_GetItem(keys, i), *PyList_GetItem(values, i));
    }

    Py_DECREF(keys); Py_DECREF(values);
}


int npArrayVectorDims(PyArrayObject* np_array) {
    // Utility for determining which bifrost vector type to convert a numpy array to. This should always return 2, 3, or 4. I can also return 1 indicating a scalar array.

    npy_intp* shape = PyArray_SHAPE(np_array);
    int num_dims = PyArray_NDIM(np_array);

    // If the array is not an array of arrays (second dim being the vector) return indicating scalar array
    if (num_dims != 2)
        return 1;

    // The the second dim is greater than 4, it will not fit in any Bifrost vector type so return indicating scalar array
    if (shape[1] > 4)
        return 1;

    // At this point its safe to use whatever this value is.
    return shape[1];

}


// =================================================================================================
// Basic Conversions

// tuple -> vector
template <typename VT, typename T>
VT vector2FromPy(PyObject& py_value) {
    VT vector;
    vector.x = singleFromPy<T>(*PyTuple_GetItem(&py_value, 0));
    vector.y = singleFromPy<T>(*PyTuple_GetItem(&py_value, 1));
    return vector;
}

template <typename VT, typename T>
VT vector3FromPy(PyObject& py_value) {
    VT vector;
    vector.x = singleFromPy<T>(*PyTuple_GetItem(&py_value, 0));
    vector.y = singleFromPy<T>(*PyTuple_GetItem(&py_value, 1));
    vector.z = singleFromPy<T>(*PyTuple_GetItem(&py_value, 2));
    return vector;
}

template <typename VT, typename T>
VT vector4FromPy(PyObject& py_value) {
    VT vector;
    vector.x = singleFromPy<T>(*PyTuple_GetItem(&py_value, 0));
    vector.y = singleFromPy<T>(*PyTuple_GetItem(&py_value, 1));
    vector.z = singleFromPy<T>(*PyTuple_GetItem(&py_value, 2));
    vector.w = singleFromPy<T>(*PyTuple_GetItem(&py_value, 3));
    return vector;
}



template <>  // float -> float
float singleFromPy(PyObject& py_value) {
    return PyFloat_AS_DOUBLE(&py_value);
}

template <>  // tuple -> float2
Bifrost::Math::float2 singleFromPy(PyObject& py_value) {
    return vector2FromPy<Bifrost::Math::float2, float>(py_value);
}

template <>  // tuple -> float3
Bifrost::Math::float3 singleFromPy(PyObject& py_value) {
    return vector3FromPy<Bifrost::Math::float3, float>(py_value);
}

template <>  // tuple -> float4
Bifrost::Math::float4 singleFromPy(PyObject& py_value) {
    return vector4FromPy<Bifrost::Math::float4, float>(py_value);
}

template <>  // int -> long long
long long singleFromPy(PyObject& py_value) {
    return PyLong_AsLongLong(&py_value);
}

template <>  // tuple -> long2
Bifrost::Math::long2 singleFromPy(PyObject& py_value) {
    return vector2FromPy<Bifrost::Math::long2, long long>(py_value);
}

template <>  // tuple -> long3
Bifrost::Math::long3 singleFromPy(PyObject& py_value) {
    return vector3FromPy<Bifrost::Math::long3, long long>(py_value);
}

template <>  // tuple -> long4
Bifrost::Math::long4 singleFromPy(PyObject& py_value) {
    return vector4FromPy<Bifrost::Math::long4, long long>(py_value);
}

template <>  // str -> string
Amino::String singleFromPy(PyObject& py_value) {
    return PyUnicode_AsUTF8(&py_value);
}

template <>  // True/False -> bool
bool singleFromPy(PyObject& py_value) {
    return PyObject_IsTrue(&py_value);
}

template <>  // dict -> Object
Amino::Ptr<Bifrost::Object> singleFromPy(PyObject& py_value) {
    auto bob = Bifrost::createObject();
    addDictToBob(*bob, py_value);
    return bob.toImmutable();

}

// Np -> array
template <typename T>
Amino::Ptr<Amino::Array<T>> arrayFromNp(PyArrayObject* np_array, int vector_dims) {

    T* data = (T*)PyArray_DATA(np_array);
    int numElements = PyArray_SIZE(np_array)/vector_dims;

    std::initializer_list<T> init(data, data + numElements);
    Amino::Array<T> amino_array(init);
    auto array_ptr = Amino::newClassPtr<Amino::Array<T>>(std::move(amino_array));
    return array_ptr;

}

// list -> array
template <typename T>
Amino::Ptr<Amino::Array<T>> arrayFromPy(PyObject& py_value, int list_len) {

    Amino::Array<T> amino_array(list_len);
    for (int i = 0; i < list_len; i++) {
        auto py_item = PyList_GetItem(&py_value, i);
        amino_array[i] = singleFromPy<T>(*py_item);
    }
    auto out_array = Amino::newClassPtr<Amino::Array<T>>(std::move(amino_array));
    return out_array;

}



// =================================================================================================
// This set of functions try to convert the given PyObject and add it the Object.


// Single Vector
void setPropertyTuple(Bifrost::Object& bob, const char* key, PyObject& py_value) {
    
    int tuple_len = PyTuple_GET_SIZE(&py_value);
    if (tuple_len < 2 || tuple_len > 4)
        return;  // tuple length does not match any vector

    PyObject* x_comp = PyTuple_GetItem(&py_value, 0);

    // float vector
    if (Py_IS_TYPE(x_comp, &PyFloat_Type)) {
        switch (tuple_len) {
        case 2: bob.setProperty(key, singleFromPy<Bifrost::Math::float2>(py_value)); return;
        case 3: bob.setProperty(key, singleFromPy<Bifrost::Math::float3>(py_value)); return;
        case 4: bob.setProperty(key, singleFromPy<Bifrost::Math::float4>(py_value)); return;
        }
    }

    // long vector
    else if (Py_IS_TYPE(x_comp, &PyLong_Type)) {
        switch (tuple_len) {
        case 2: bob.setProperty(key, singleFromPy<Bifrost::Math::long2>(py_value)); return;
        case 3: bob.setProperty(key, singleFromPy<Bifrost::Math::long3>(py_value)); return;
        case 4: bob.setProperty(key, singleFromPy<Bifrost::Math::long4>(py_value)); return;
        }
    }
}


// Array (Np)
void setPropertyNumpy(Bifrost::Object& bob, const char* key, PyObject& py_value) {

    PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(&py_value);
    PyArray_Descr* descr = PyArray_DESCR(np_array);

    switch (descr->type_num) {
    case NPY_FLOAT:
        switch (npArrayVectorDims(np_array)) {
        case 1: bob.setProperty(key, arrayFromNp<float>(np_array, 1)); return;
        case 2: bob.setProperty(key, arrayFromNp<Bifrost::Math::float2>(np_array, 2)); return;
        case 3: bob.setProperty(key, arrayFromNp<Bifrost::Math::float3>(np_array, 3)); return;
        case 4: bob.setProperty(key, arrayFromNp<Bifrost::Math::float4>(np_array, 4)); return;
        default: return;
        }
    case NPY_LONGLONG:
        switch (npArrayVectorDims(np_array)) {
        case 1: bob.setProperty(key, arrayFromNp<long long>(np_array, 1)); return;
        case 2: bob.setProperty(key, arrayFromNp<Bifrost::Math::long2>(np_array, 2)); return;
        case 3: bob.setProperty(key, arrayFromNp<Bifrost::Math::long3>(np_array, 3)); return;
        case 4: bob.setProperty(key, arrayFromNp<Bifrost::Math::long4>(np_array, 4)); return;
        default: return;
        }

    case NPY_BOOL:
        bob.setProperty(key, arrayFromNp<bool>(np_array, 1));
        return;

    case NPY_UINT:
		bob.setProperty(key, arrayFromNp<unsigned int>(np_array, 1));
        return;
    }
}


// Array (List)
void setPropertyList(Bifrost::Object& bob, const char* key, PyObject& py_value) {

    // length check
    const int list_len = PyList_GET_SIZE(&py_value);
    if (list_len == 0)
        return;

    PyObject* first = PyList_GetItem(&py_value, 0);

    // Type checks
    if (Py_IS_TYPE(first, &PyDict_Type)) {
        bob.setProperty(key, arrayFromPy<Amino::Ptr<Bifrost::Object>>(py_value, list_len));
    }
    else if (Py_IS_TYPE(first, &PyUnicode_Type)) {
        bob.setProperty(key, arrayFromPy<Amino::String>(py_value, list_len));
    }

    // Numeric arrays are now handled, but user could still create a list within their script so conditions for this remain.
    else if (Py_IS_TYPE(first, &PyFloat_Type)) {
        bob.setProperty(key, arrayFromPy<float>(py_value, list_len));
    }
    else if (Py_IS_TYPE(first, &PyLong_Type)) {
        bob.setProperty(key, arrayFromPy<long long>(py_value, list_len));
    }
    else if (Py_IS_TYPE(first, &PyTuple_Type)) {
        int tuple_len = PyTuple_GET_SIZE(first);
        if (tuple_len) {
            PyObject* x_comp = PyTuple_GetItem(first, 0);
            if (Py_IS_TYPE(x_comp, &PyFloat_Type)) {
                switch (tuple_len) {
                case 2: bob.setProperty(key, arrayFromPy<Bifrost::Math::float2>(py_value, list_len)); return;
                case 3: bob.setProperty(key, arrayFromPy<Bifrost::Math::float3>(py_value, list_len)); return;
                case 4: bob.setProperty(key, arrayFromPy<Bifrost::Math::float4>(py_value, list_len)); return;
                default: return;
                }
            }
            else if (Py_IS_TYPE(x_comp, &PyLong_Type)) {
                switch (tuple_len) {
                case 2: bob.setProperty(key, arrayFromPy<Bifrost::Math::long2>(py_value, list_len)); return;
                case 3: bob.setProperty(key, arrayFromPy<Bifrost::Math::long3>(py_value, list_len)); return;
                case 4: bob.setProperty(key, arrayFromPy<Bifrost::Math::long4>(py_value, list_len)); return;
                default: return;
                }
            }
        }

    }
    else if (Py_IS_TYPE(first, &PyBool_Type)) {
        bob.setProperty(key, arrayFromPy<bool>(py_value, list_len));
    }
}


// =================================================================================================
// Calls the appropriate set property function based on the PyObject type
void setPropertyFromDictItem(Bifrost::Object& bob, PyObject& py_key, PyObject& py_value) {

    auto key = PyUnicode_AsUTF8(&py_key);

    if (Py_IS_TYPE(&py_value, &PyFloat_Type)) // -> float
        bob.setProperty(key, singleFromPy<float>(py_value));

    else if (Py_IS_TYPE(&py_value, &PyLong_Type)) // -> long long
        bob.setProperty(key, singleFromPy<long long>(py_value));

    else if (Py_IS_TYPE(&py_value, &PyTuple_Type)) // -> vector
        setPropertyTuple(bob, key, py_value);

    else if (PyArray_Check(&py_value)) // -> array
        setPropertyNumpy(bob, key, py_value);

    else if (Py_IS_TYPE(&py_value, &PyList_Type)) // -> array
        setPropertyList(bob, key, py_value);

    else if (Py_IS_TYPE(&py_value, &PyDict_Type)) // -> Object
        bob.setProperty(key, singleFromPy<Amino::Ptr<Bifrost::Object>>(py_value));

    else if (Py_IS_TYPE(&py_value, &PyBool_Type)) // -> bool
        bob.setProperty(key, singleFromPy<bool>(py_value));

    else if (Py_IS_TYPE(&py_value, &PyUnicode_Type)) // -> string
        bob.setProperty(key, singleFromPy<Amino::String>(py_value));

}