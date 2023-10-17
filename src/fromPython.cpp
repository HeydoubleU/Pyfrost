#include "python.h"
#include "fromPython.h"

using ulong = unsigned long long;
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;
using PtrBob = Amino::Ptr<Bifrost::Object>;

int initFromPython() {
    import_array();
    return 0;
}

namespace FromPython
{
    void mergeBobWithDict(Amino::MutablePtr<Bifrost::Object>& bob, PyObject* py_obj)
    {
        PyObject* keys = PyDict_Keys(py_obj);
        PyObject* values = PyDict_Values(py_obj);
        for (int i = 0; i < PyList_GET_SIZE(keys); i++) {
            auto py_key = PyList_GetItem(keys, i);
            auto py_item = PyList_GetItem(values, i);
            bob->setProperty(PyUnicode_AsUTF8(py_key), anyFromPy(py_item));
        }
        Py_DECREF(keys); Py_DECREF(values);
    }

    template <typename T>
    T toSimple(PyObject* py_obj);

    template <>
    PtrBob toSimple(PyObject* py_obj) {
        auto bob = Bifrost::createObject();
        mergeBobWithDict(bob, py_obj);
        return bob.toImmutable();
	}

    template <>
    Amino::String toSimple(PyObject* py_obj) {
        return PyUnicode_AsUTF8(py_obj);
    }

    template <>
    float toSimple(PyObject* py_obj) {
        return PyFloat_AS_DOUBLE(py_obj);
    }

    template <>
    long long toSimple(PyObject* py_obj) {
		return PyLong_AsLongLong(py_obj);
	}

    template <>
    bool toSimple(PyObject* py_obj) {
		return PyObject_IsTrue(py_obj);
	}

    template <typename T>
    Amino::Ptr<Amino::Array<T>> arrayFromSequence(PyObject* py_obj) {
        auto list_len = PySequence_Size(py_obj);
        Amino::Array<T> amino_array(list_len);
        for (int i = 0; i < list_len; i++) {
            auto py_item = PySequence_GetItem(py_obj, i);
            amino_array[i] = toSimple<T>(py_item);
        }
        auto out_array = Amino::newClassPtr<Amino::Array<T>>(std::move(amino_array));
        return out_array;
    }

    Amino::Any anyFromSequence(PyObject* py_obj) {
		auto list_len = PySequence_Size(py_obj);
		if (list_len == 0) // if the list/tuple is empty the type can not be determined, assume a list of objects.
			return Amino::newClassPtr<Amino::Array<PtrBob>>();

        Amino::Any result;
        PyObject* first = PySequence_GetItem(py_obj, 0);

        if (Py_IS_TYPE(first, &PyDict_Type)) // -> Object
            result = arrayFromSequence<PtrBob>(py_obj);
        else if (Py_IS_TYPE(first, &PyUnicode_Type))
            result = arrayFromSequence<Amino::String>(py_obj);
        else if (Py_IS_TYPE(first, &PyFloat_Type))
            result = arrayFromSequence<float>(py_obj);
        else if (Py_IS_TYPE(first, &PyLong_Type))
            result = arrayFromSequence<long long>(py_obj);
        else if (Py_IS_TYPE(first, &PyBool_Type))
            result = arrayFromSequence<bool>(py_obj);
        else
            result = Amino::newClassPtr<Amino::Array<PtrBob>>();

        Py_CLEAR(first);
        return result;
	}
}


namespace FromNumpy
{
    Amino::Any toScalar(PyObject* py_obj)
    {
        auto desc = PyArray_DescrFromScalar(py_obj)->type_num;
        void* data;
        PyArray_ScalarAsCtype(py_obj, data);
        switch (PyArray_DescrFromScalar(py_obj)->type_num) {
        case NPY_FLOAT: return *static_cast<float*>(data);
        case NPY_DOUBLE: return *static_cast<double*>(data);
        case NPY_LONGLONG: return *static_cast<long long*>(data);
        case NPY_ULONGLONG: return *static_cast<ulong*>(data);
        case NPY_INT: case NPY_LONG: return *static_cast<int*>(data);
        case NPY_UINT: case NPY_ULONG: return *static_cast<uint*>(data);
        case NPY_SHORT: return *static_cast<short*>(data);
        case NPY_USHORT: return *static_cast<ushort*>(data);
        case NPY_BYTE: return *static_cast<signed char*>(data);
        case NPY_UBYTE: return *static_cast<uchar*>(data);
        case NPY_BOOL: return *static_cast<bool*>(data);
        default: return FromPython::toSimple<Amino::String>(PyObject_Str(py_obj));
        }

        // otherwise return py_obj as string

    }

    inline bool validMembers(int members) {
        // Verify the number of memebers represents a valid vector or matrix type
        return members <= 4 && members >= 2;
    }

    int solveArrayShapeType(PyArrayObject* np_array, int& size) {
        npy_intp* shape = PyArray_SHAPE(np_array);
        int ndim = PyArray_NDIM(np_array);

        switch (ndim) {
        case 0 || 1: // scalar
            size = PyArray_SIZE(np_array);
            return 0;
        case 2: // vector
            if (validMembers(shape[1])) {
                size = np_array->dimensions[0];
                return shape[1];
            }
            break;
        case 3: // matrix
            if (validMembers(shape[1]) && validMembers(shape[2])) {
                size = np_array->dimensions[0];
                return shape[2] * 10 + shape[1];
            }
            break;
        }

        size = PyArray_SIZE(np_array);
        return 0;
    }

    int solveComplexShapeType(PyArrayObject* np_array, int& size) {
        npy_intp* shape = PyArray_SHAPE(np_array);
        int ndim = PyArray_NDIM(np_array);

        switch (ndim) {
        case 0: // array with no dimension
            size = PyArray_SIZE(np_array);
            return 0;
        case 1: // vector
            if (validMembers(shape[0]))
                return shape[0];
            break;
        case 2: // matrix
            if (validMembers(shape[0]) && validMembers(shape[1]))
                return shape[1] * 10 + shape[0];
            break;
        }

        size = PyArray_SIZE(np_array);
        return 0;
    }

    template <typename T>
    Amino::Any castArrayToType(PyArrayObject* np_array, int size)
    {
        T* data = (T*)PyArray_DATA(np_array);

        if (size < 0)
            return *data;

        std::initializer_list<T> init(data, data + size);
        Amino::Array<T> amino_array(init);
        auto array_ptr = Amino::newClassPtr<Amino::Array<T>>(std::move(amino_array));
        return array_ptr;
    }

    Amino::Any castToObjectArray(PyArrayObject* np_array, int size)
    {
        Amino::Array<PtrBob> amino_array(size);

        for (int i = 0; i < size; i++) {
            auto py_item = PyArray_GETITEM(np_array, PyArray_GETPTR1(np_array, i));
            amino_array[i] = FromPython::toSimple<PtrBob>(py_item);
            Py_CLEAR(py_item);
        }

        auto array_ptr = Amino::newClassPtr<Amino::Array<PtrBob>>(std::move(amino_array));
        return array_ptr;
    }

    Amino::Any castToStringArray(PyArrayObject* np_array, int size)
    {
        Amino::Array<Amino::String> amino_array(size);

        for (int i = 0; i < size; i++) {
            auto py_item = PyArray_GETITEM(np_array, PyArray_GETPTR1(np_array, i));
            amino_array[i] = FromPython::toSimple<Amino::String>(py_item);
            Py_CLEAR(py_item);
        }

        auto array_ptr = Amino::newClassPtr<Amino::Array<Amino::String>>(std::move(amino_array));
        return array_ptr;
    }

    Amino::Any arrayToAny(PyObject* py_obj)
    {
        PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_obj);
        int size = -1; int shape_type;
        bool has_md = (bool)np_array->descr->metadata;

        if (has_md)
            shape_type = solveComplexShapeType(np_array, size);
        else
			shape_type = solveArrayShapeType(np_array, size);

        switch (np_array->descr->type_num) {
        case NPY_OBJECT:
            size = PyArray_SIZE(np_array);
            if (size == 0) {  // If empty we refer to metadata to determine type
                if (has_md)
                    return castToStringArray(np_array, size);
				else
                    return castToObjectArray(np_array, size);
            }
            else {  // Else we check the first element's type
                auto first = PyArray_GETITEM(np_array, PyArray_GETPTR1(np_array, 0));
                bool is_dict = PyDict_Check(first);
                Py_CLEAR(first);
                if (is_dict)
                    return castToObjectArray(np_array, size);
                else
                    return castToStringArray(np_array, size);
            }
            
        case NPY_UNICODE:
			return castToStringArray(np_array, PyArray_SIZE(np_array));
        case NPY_STRING:
            return castToStringArray(np_array, PyArray_SIZE(np_array));
        case NPY_FLOAT:
            switch (shape_type) {
            case 0: return castArrayToType<float>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::float2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::float3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::float4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::float2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::float2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::float2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::float3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::float3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::float3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::float4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::float4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::float4x4>(np_array, size);
            }
        case NPY_DOUBLE:
            switch (shape_type) {
            case 0: return castArrayToType<double>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::double2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::double3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::double4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::double2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::double2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::double2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::double3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::double3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::double3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::double4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::double4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::double4x4>(np_array, size);
            }
        case NPY_LONGLONG:
            switch (shape_type) {
            case 0: return castArrayToType<long long>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::long2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::long3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::long4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::long2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::long2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::long2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::long3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::long3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::long3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::long4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::long4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::long4x4>(np_array, size);
            }
        case NPY_ULONGLONG:
            switch (shape_type) {
            case 0: return castArrayToType<ulong>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::ulong2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::ulong3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::ulong4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::ulong2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::ulong2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::ulong2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::ulong3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::ulong3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::ulong3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::ulong4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::ulong4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::ulong4x4>(np_array, size);
            }
        case NPY_INT:
        case NPY_LONG:
            switch (shape_type) {
            case 0: return castArrayToType<int>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::int2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::int3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::int4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::int2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::int2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::int2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::int3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::int3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::int3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::int4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::int4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::int4x4>(np_array, size);
            }
        case NPY_UINT:
        case NPY_ULONG:
            switch (shape_type) {
            case 0: return castArrayToType<uint>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::uint2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::uint3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::uint4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::uint2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::uint2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::uint2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::uint3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::uint3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::uint3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::uint4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::uint4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::uint4x4>(np_array, size);
            }
        case NPY_SHORT:
            switch (shape_type) {
            case 0: return castArrayToType<short>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::short2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::short3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::short4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::short2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::short2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::short2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::short3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::short3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::short3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::short4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::short4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::short4x4>(np_array, size);
            }
        case NPY_USHORT:
            switch (shape_type) {
            case 0: return castArrayToType<ushort>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::ushort2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::ushort3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::ushort4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::ushort2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::ushort2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::ushort2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::ushort3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::ushort3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::ushort3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::ushort4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::ushort4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::ushort4x4>(np_array, size);
            }
        case NPY_BYTE:
            switch (shape_type) {
            case 0: return castArrayToType<signed char>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::char2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::char3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::char4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::char2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::char2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::char2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::char3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::char3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::char3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::char4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::char4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::char4x4>(np_array, size);
            }
        case NPY_UBYTE:
            switch (shape_type) {
            case 0: return castArrayToType<uchar>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::uchar2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::uchar3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::uchar4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::uchar2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::uchar2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::uchar2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::uchar3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::uchar3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::uchar3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::uchar4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::uchar4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::uchar4x4>(np_array, size);
            }
        case NPY_BOOL:
            switch (shape_type) {
            case 0: return castArrayToType<bool>(np_array, size);
            case 2: return castArrayToType<Bifrost::Math::bool2>(np_array, size);
            case 3: return castArrayToType<Bifrost::Math::bool3>(np_array, size);
            case 4: return castArrayToType<Bifrost::Math::bool4>(np_array, size);
            case 22: return castArrayToType<Bifrost::Math::bool2x2>(np_array, size);
            case 23: return castArrayToType<Bifrost::Math::bool2x3>(np_array, size);
            case 24: return castArrayToType<Bifrost::Math::bool2x4>(np_array, size);
            case 32: return castArrayToType<Bifrost::Math::bool3x2>(np_array, size);
            case 33: return castArrayToType<Bifrost::Math::bool3x3>(np_array, size);
            case 34: return castArrayToType<Bifrost::Math::bool3x4>(np_array, size);
            case 42: return castArrayToType<Bifrost::Math::bool4x2>(np_array, size);
            case 43: return castArrayToType<Bifrost::Math::bool4x3>(np_array, size);
            case 44: return castArrayToType<Bifrost::Math::bool4x4>(np_array, size);
            }
        }

        return FromPython::toSimple<Amino::String>(PyObject_Str(py_obj));
    }
}


Amino::Any anyFromPy(PyObject* py_obj)
{
    if (Py_IS_TYPE(py_obj, &PyDict_Type))
        return FromPython::toSimple<PtrBob>(py_obj);
    else if (Py_IS_TYPE(py_obj, &PyUnicode_Type))
		return FromPython::toSimple<Amino::String>(py_obj);
    else if (PyArray_CheckScalar(py_obj))
        return FromNumpy::toScalar(py_obj);
    else if (PyArray_Check(py_obj))
        return FromNumpy::arrayToAny(py_obj);
    else if (Py_IS_TYPE(py_obj, &PyFloat_Type))
		return FromPython::toSimple<float>(py_obj);
	else if (Py_IS_TYPE(py_obj, &PyLong_Type))
		return FromPython::toSimple<long long>(py_obj);
    else if (Py_IS_TYPE(py_obj, &PyBool_Type))
        return FromPython::toSimple<bool>(py_obj);
	else if (Py_IS_TYPE(py_obj, &PyList_Type) || Py_IS_TYPE(py_obj, &PyTuple_Type))
		return FromPython::anyFromSequence(py_obj);
    return FromPython::toSimple<Amino::String>(PyObject_Str(py_obj));
}