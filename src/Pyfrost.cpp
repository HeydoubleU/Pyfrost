#include "python.h"
#include "Pyfrost.h"
#include <Windows.h>
#include <string>

PyObject* PY_EXEC_FUNC;
PyObject* PY_GLOBAL_EXEC_FUNC;
int PY_INITED = 0;

void initPython() {
    PY_INITED = 1;
    Py_Initialize();
    std::string sys_cmd = "import sys; sys.path.append('')";
    sys_cmd.replace(29, 0, std::getenv("PYFROST_MODULE_PATH"));
    PyRun_SimpleString(sys_cmd.c_str());
    PyObject* py_module = PyImport_Import(PyUnicode_FromString("PyfrostIO"));
    PY_EXEC_FUNC = PyObject_GetAttrString(py_module, (char*)"_execNode");
    PY_GLOBAL_EXEC_FUNC = PyObject_GetAttrString(py_module, (char*)"_execNodeGlobal");
}

// Prototypes
void addDictToBob(Bifrost::Object& bob, PyObject& dict);
PyObject* anyToPy(Amino::Any data);


// BIFROST TO PYTHON ============================================================================================================================
//===============================================================================================================================================

// Base type conversions ------------------------------------------------------------------------------------------------------------------------
PyObject* valueToPy(float value) {
    return PyFloat_FromDouble(value);
}

PyObject* valueToPy(Bifrost::Math::float2 value) {
    return PyTuple_Pack(2, PyFloat_FromDouble(value.x), PyFloat_FromDouble(value.y));
}

PyObject* valueToPy(Bifrost::Math::float3 value) {
    return PyTuple_Pack(3, PyFloat_FromDouble(value.x), PyFloat_FromDouble(value.y), PyFloat_FromDouble(value.z));
}

PyObject* valueToPy(Bifrost::Math::float4 value) {
    return PyTuple_Pack(4, PyFloat_FromDouble(value.x), PyFloat_FromDouble(value.y), PyFloat_FromDouble(value.z), PyFloat_FromDouble(value.w));
}

PyObject* valueToPy(Bifrost::Math::long2 value) {
    return PyTuple_Pack(2, PyLong_FromLong(value.x), PyLong_FromLong(value.y));
}

PyObject* valueToPy(Bifrost::Math::long3 value) {
    return PyTuple_Pack(3, PyLong_FromLong(value.x), PyLong_FromLong(value.y), PyLong_FromLong(value.z));
}

PyObject* valueToPy(Bifrost::Math::long4 value) {
    return PyTuple_Pack(4, PyLong_FromLong(value.x), PyLong_FromLong(value.y), PyLong_FromLong(value.z), PyLong_FromLong(value.w));
}

PyObject* valueToPy(long long value) {
    return PyLong_FromLong(value);
}

PyObject* valueToPy(unsigned int value) {
    return PyLong_FromLong(value);
}

PyObject* valueToPy(bool value) {
    if (value)
        return Py_True;
    return Py_False;
}

PyObject* valueToPy(Amino::String value) {
    return PyUnicode_FromString(value.c_str());
}

PyObject* valueToPy(Amino::Ptr<Bifrost::Object> bob) {
    PyObject* dict = PyDict_New();
    auto keys = bob->keys();
    for (int i = 0; i < keys->size(); i++) {
        const char* key = keys->at(i).c_str();
        PyObject* converted = anyToPy(bob->getProperty(key));
        if (converted) PyDict_SetItemString(dict, key, converted);
    }
    return dict;
}

// Template for array conversion ----------------------------------------------------------------------------------------------------------------
template <typename T>
PyObject* arrayToPy(T amino_array) {
    PyObject* pyList = PyList_New((int)amino_array.size());
    for (int i = 0; i < amino_array.size(); i++) {
        PyList_SetItem(pyList, i, valueToPy(amino_array[i]));
    }
    return pyList;
}

// Try converting type cast to single  ----------------------------------------------------------------------------------------------------------
template <typename T>
PyObject* tryCastToPy(Amino::Any data) {
    auto payload = Amino::any_cast<T>(&data);
    if (payload != nullptr) {
        return valueToPy(*payload);
    }

    auto a_payload = Amino::any_cast<Amino::Ptr<Amino::Array<T>>>(data);
    if (a_payload != nullptr) {
        return arrayToPy(*a_payload);
    }

    // TODO: add conditions for 2/3D array

    return NULL;
}

// Any conversion ----------------------------------------------------------------------------------------------------------------------------
PyObject* anyToPy(Amino::Any data) {
    PyObject* py_obj;

    py_obj = tryCastToPy<float>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::float2>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::float3>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::float4>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<long long>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::long2>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::long3>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Bifrost::Math::long4>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<unsigned int>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<bool>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Amino::String>(data);
    if (py_obj) return py_obj;
    py_obj = tryCastToPy<Amino::Ptr<Bifrost::Object>>(data);
    if (py_obj) return py_obj;

    return Py_None;
}

// Used for input object conversion, as opposed to 'valueToPy' this function has arg for converting only certain keys
PyObject* bobToDict(Bifrost::Object& bob, Amino::Array<Amino::String> keys) {
    PyObject* dict = PyDict_New();
    for (int i = 0; i < keys.size(); i++) {
        const char* key = keys.at(i).c_str();
        if (bob.hasProperty(key)) {
            PyObject* converted = anyToPy(bob.getProperty(key));
            if (converted) PyDict_SetItemString(dict, key, converted);
        }
    }
    return dict;
}

PyObject* bobToDict(Bifrost::Object& bob, bool properties) {
    PyObject* dict = PyDict_New();
    if (!properties) return dict;

    auto keys = *bob.keys();
    for (int i = 0; i < keys.size(); i++) {
        const char* key = keys.at(i).c_str();
        PyObject* converted = anyToPy(bob.getProperty(key));
        if (converted) PyDict_SetItemString(dict, key, converted);
    }
    return dict;
}
//===============================================================================================================================================



// PYTHON TO BIFROST ============================================================================================================================
//===============================================================================================================================================

// Generic scalar/vector templates
template <typename T>
T valueFromPy(PyObject& py_value);

template <typename VT, typename T>
VT vector2FromPy(PyObject& py_value) {
    VT vector;
    vector.x = valueFromPy<T>(*PyTuple_GetItem(&py_value, 0));
    vector.y = valueFromPy<T>(*PyTuple_GetItem(&py_value, 1));
    return vector;
}
template <typename VT, typename T>
VT vector3FromPy(PyObject& py_value) {
    VT vector;
    vector.x = valueFromPy<T>(*PyTuple_GetItem(&py_value, 0));
    vector.y = valueFromPy<T>(*PyTuple_GetItem(&py_value, 1));
    vector.z = valueFromPy<T>(*PyTuple_GetItem(&py_value, 2));
    return vector;
}
template <typename VT, typename T>
VT vector4FromPy(PyObject& py_value) {
    VT vector;
    vector.x = valueFromPy<T>(*PyTuple_GetItem(&py_value, 0));
    vector.y = valueFromPy<T>(*PyTuple_GetItem(&py_value, 1));
    vector.z = valueFromPy<T>(*PyTuple_GetItem(&py_value, 2));
    vector.w = valueFromPy<T>(*PyTuple_GetItem(&py_value, 3));
    return vector;
}


// FLOAT
template <>
float valueFromPy(PyObject& py_value) {
    return PyFloat_AS_DOUBLE(&py_value);
}

template <>
Bifrost::Math::float2 valueFromPy(PyObject& py_value) {
    return vector2FromPy<Bifrost::Math::float2, float>(py_value);
}

template <>
Bifrost::Math::float3 valueFromPy(PyObject& py_value) {
    return vector3FromPy<Bifrost::Math::float3, float>(py_value);
}

template <>
Bifrost::Math::float4 valueFromPy(PyObject& py_value) {
    return vector4FromPy<Bifrost::Math::float4, float>(py_value);
}

// LONG
template <>
long long valueFromPy(PyObject& py_value) {
    return PyLong_AsLongLong(&py_value);
}

template <>
Bifrost::Math::long2 valueFromPy(PyObject& py_value) {
    return vector2FromPy<Bifrost::Math::long2, long long>(py_value);
}

template <>
Bifrost::Math::long3 valueFromPy(PyObject& py_value) {
    return vector3FromPy<Bifrost::Math::long3, long long>(py_value);
}

template <>
Bifrost::Math::long4 valueFromPy(PyObject& py_value) {
    return vector4FromPy<Bifrost::Math::long4, long long>(py_value);
}

// STRING
template <>
Amino::String valueFromPy(PyObject& py_value) {
    return PyUnicode_AsUTF8(&py_value);
}

// BOOL
template <>
bool valueFromPy(PyObject& py_value) {
    return PyObject_IsTrue(&py_value);
}

// ARRAY
template <typename T>
Amino::Ptr<Amino::Array<T>> listToArray(PyObject& py_value, int list_len) {
    Amino::Array<T> amino_array(list_len);
    for (int i = 0; i < list_len; i++) {
        auto tuple = PyList_GetItem(&py_value, i);
        amino_array[i] = valueFromPy<T>(*tuple);
    }
    auto out_array = Amino::newClassPtr<Amino::Array<T>>(std::move(amino_array));
    return out_array;
}

// Get value from python dict and add to existing BOB -------------------------------------------------------------------------------------------
void propertyFromDictItem(Bifrost::Object& bob, PyObject& py_key, PyObject& py_value) {
    auto key = PyUnicode_AsUTF8(&py_key);


    // Singles ________________________________

    if (Py_IS_TYPE(&py_value, &PyFloat_Type)) {
        bob.setProperty(key, valueFromPy<float>(py_value));
    }
    else if (Py_IS_TYPE(&py_value, &PyLong_Type)) {
        bob.setProperty(key, PyLong_AsLongLong(&py_value));
    }
    else if (Py_IS_TYPE(&py_value, &PyTuple_Type)) {
        int tuple_len = PyTuple_GET_SIZE(&py_value);
        if (tuple_len) {
            PyObject* x_comp = PyTuple_GetItem(&py_value, 0);
            if (Py_IS_TYPE(x_comp, &PyFloat_Type)) {
                if (tuple_len == 2) bob.setProperty(key, valueFromPy<Bifrost::Math::float2>(py_value));
                if (tuple_len == 3) bob.setProperty(key, valueFromPy<Bifrost::Math::float3>(py_value));
                if (tuple_len == 4) bob.setProperty(key, valueFromPy<Bifrost::Math::float4>(py_value));
            }
            else if (Py_IS_TYPE(x_comp, &PyLong_Type)) {
                if (tuple_len == 2) bob.setProperty(key, valueFromPy<Bifrost::Math::long2>(py_value));
                if (tuple_len == 3) bob.setProperty(key, valueFromPy<Bifrost::Math::long3>(py_value));
                if (tuple_len == 4) bob.setProperty(key, valueFromPy<Bifrost::Math::long4>(py_value));
            }
        }
    }
    else if (Py_IS_TYPE(&py_value, &PyUnicode_Type)) {
        bob.setProperty(key, PyUnicode_AsUTF8(&py_value));
    }
    else if (Py_IS_TYPE(&py_value, &PyBool_Type)) {
        bob.setProperty(key, (bool)PyObject_IsTrue(&py_value));
    }


    // Arrays _____________________________________

    else if (Py_IS_TYPE(&py_value, &PyList_Type)) {
        // length check
        const int list_len = PyList_GET_SIZE(&py_value);
        if (list_len == 0) return;
        PyObject* first = PyList_GetItem(&py_value, 0);


        // Type checks
        if (Py_IS_TYPE(first, &PyFloat_Type)) {
            bob.setProperty(key, listToArray<float>(py_value, list_len));
        }
        else if (Py_IS_TYPE(first, &PyLong_Type)) {
            bob.setProperty(key, listToArray<long long>(py_value, list_len));
        }
        else if (Py_IS_TYPE(first, &PyTuple_Type)) {
            int tuple_len = PyTuple_GET_SIZE(first);
            if (tuple_len) {
                PyObject* x_comp = PyTuple_GetItem(first, 0);
                if (Py_IS_TYPE(x_comp, &PyFloat_Type)) {
                    if (tuple_len == 2) bob.setProperty(key, listToArray<Bifrost::Math::float2>(py_value, list_len));
                    if (tuple_len == 3) bob.setProperty(key, listToArray<Bifrost::Math::float3>(py_value, list_len));
                    if (tuple_len == 4) bob.setProperty(key, listToArray<Bifrost::Math::float4>(py_value, list_len));
                }
                else if (Py_IS_TYPE(x_comp, &PyLong_Type)) {
                    if (tuple_len == 2) bob.setProperty(key, listToArray<Bifrost::Math::long2>(py_value, list_len));
                    if (tuple_len == 3) bob.setProperty(key, listToArray<Bifrost::Math::long3>(py_value, list_len));
                    if (tuple_len == 4) bob.setProperty(key, listToArray<Bifrost::Math::long4>(py_value, list_len));
                }
            }
            
        }
        else if (Py_IS_TYPE(first, &PyUnicode_Type)) {
            bob.setProperty(key, listToArray<Amino::String>(py_value, list_len));
        }
        else if (Py_IS_TYPE(first, &PyBool_Type)) {
            bob.setProperty(key, listToArray<bool>(py_value, list_len));
        }
    }

    else if (Py_IS_TYPE(&py_value, &PyDict_Type)) {
        auto sub_bob = Bifrost::createObject();
        addDictToBob(*sub_bob, py_value);
        bob.setProperty(key, std::move(sub_bob));
    }
}

void addDictToBob(Bifrost::Object& bob, PyObject& dict) {
    PyObject* keys = PyDict_Keys(&dict);
    PyObject* values = PyDict_Values(&dict);
    for (int i = 0; i < PyList_GET_SIZE(keys); i++) {
        propertyFromDictItem(bob, *PyList_GetItem(keys, i), *PyList_GetItem(values, i));
    }
}

//===============================================================================================================================================



//===============================================================================================================================================
// Pyfrost Op -----------------------------------------------------------------------------------------------------------------------------------

//template <typename T> //  Both overload bodies are identical but trying to template the Ops directly caused JIT compile error
template <typename T>
bool pyfrostBody(Bifrost::Object& input, const T properties, const long long execution, const Amino::String& script) {
    if (!PY_INITED) initPython();
    PyObject* py_result;

    // pack args and call
    if (execution == 0) {
        PyObject* args = PyTuple_Pack(1, PyUnicode_FromString(script.c_str()));
        py_result = PyObject_CallObject(PY_GLOBAL_EXEC_FUNC, args);
    }
    else {
        PyObject* py_in;
        if (input.hasProperty("_")) py_in = anyToPy(input.getProperty("_"));
        else py_in = bobToDict(input, properties);

        PyObject* args = PyTuple_Pack(2, PyUnicode_FromString(script.c_str()), py_in);
        py_result = PyObject_CallObject(PY_EXEC_FUNC, args);
    }

    if (execution == 2) input.eraseAllProperties();  // if consuming input

    //add py dict to input object, type will not be dict in situations such as a python threw exception
    if (Py_IS_TYPE(py_result, &PyDict_Type)) {
        addDictToBob(input, *py_result);
        return true;
    }

    else if (Py_IsNone(py_result)) {
        return true;
    }

    else {
        if (Py_IS_TYPE(py_result, &PyUnicode_Type)) {
            input.setProperty("python_exception", PyUnicode_AsUTF8(py_result));
        }
        return false;
    }
}

void Pyfrost::Internal::pyfrost(Bifrost::Object& input, const bool properties, const long long execution, const Amino::String& script, bool& success) {
    success = pyfrostBody(input, properties, execution, script);
}

void Pyfrost::Internal::pyfrost(Bifrost::Object& input, const Amino::Array<Amino::String>& properties, const long long execution, const Amino::String& script, bool& success) {
    success = pyfrostBody(input, properties, execution, script);
}

void Pyfrost::Internal::finalize_pyfrost(const bool finalize, bool& pass) {
    if (PY_INITED) {
        Py_FinalizeEx();  // Causes crash if numpy has been imported
        PY_INITED = 0;
    }
    pass = finalize;
}
