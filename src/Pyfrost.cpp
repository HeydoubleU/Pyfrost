#include "python.h"
#include "Pyfrost.h"
#include <Windows.h>
#include <string>
#include <numpy/arrayobject.h>
#include "toPython.h"
#include "fromPython.h"


PyObject* PY_EXEC_FUNC;
PyObject* PY_GLOBAL_EXEC_FUNC;
int PY_INITED = 0;


int initPython() {
    PY_INITED = 1;
    Py_Initialize();
    std::string sys_cmd = "import sys; sys.path.append('')";
    sys_cmd.replace(29, 0, std::getenv("PYFROST_MODULE_PATH"));
    PyRun_SimpleString(sys_cmd.c_str());
    PyObject* py_module = PyImport_Import(PyUnicode_FromString("PyfrostIO"));
    PY_EXEC_FUNC = PyObject_GetAttrString(py_module, (char*)"_execNode");
    PY_GLOBAL_EXEC_FUNC = PyObject_GetAttrString(py_module, (char*)"_execNodeGlobal");
    import_array();
    initToPython();
    initFromPython();
    return 0;
}

// Pyfrost Op

//template <typename T> //  Both overload bodies are identical but trying to template the Ops directly caused JIT compile error
template <typename T>
bool pyfrostBody(Bifrost::Object& input, const T properties, const long long execution, const Amino::String& script) {
    if (!PY_INITED) {
        initPython();
    }
    
    PyObject* py_result;

    // pack args and call
    if (execution == 0) {  // Exec in global space (no input object)
        PyObject* args = PyTuple_Pack(1, PyUnicode_FromString(script.c_str()));
        py_result = PyObject_Call(PY_GLOBAL_EXEC_FUNC, args, NULL);
    }
    else {  // Exec in local space
        PyObject* py_in;
        if (input.hasProperty("_"))
            py_in = anyToPy(input.getProperty("_"));  // user input is not bob
        else
            py_in = bobToDict(input, properties);  // user input is bob with properties

        PyObject* args = PyTuple_Pack(2, PyUnicode_FromString(script.c_str()), py_in);
        py_result = PyObject_Call(PY_EXEC_FUNC, args, NULL);
        Py_CLEAR(args);
        Py_CLEAR(py_in);

    }

    if (execution == 2) input.eraseAllProperties();  // if consuming input

    //add py dict to input object, type will not be dict in situations such as a python threw exception
    if (Py_IS_TYPE(py_result, &PyDict_Type)) {
        addDictToBob(input, *py_result);
        Py_CLEAR(py_result);
        return true;
    }

    else if (py_result == Py_None) {
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