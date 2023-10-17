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
void Pyfrost::Internal::pyfrost(Amino::Ptr<Bifrost::Object>& input, const Amino::String& script, const bool global, const bool consume_input, bool& success) {
    if (!PY_INITED) {
        initPython();
    }

    // Prepare inputs
    PyObject* py_in, *py_result;
    if (input->hasProperty("python_I")) {
        py_in = anyToPy(input->getProperty("python_I"));
        if (py_in == nullptr)
            py_in = Py_None;
    }  
    else
        py_in = ToPython::fromSimple(input);

    auto py_script = PyUnicode_FromString(script.c_str());
    auto args = PyTuple_Pack(2, py_script, py_in);

    if (global)
        py_result = PyObject_Call(PY_GLOBAL_EXEC_FUNC, args, NULL);
    else
        py_result = PyObject_Call(PY_EXEC_FUNC, args, NULL);

    // Clean up and process result
    Py_CLEAR(py_script); Py_CLEAR(args); Py_CLEAR(py_in);
    Amino::MutablePtr<Bifrost::Object> output;
    if (consume_input)
        output = Bifrost::createObject();
    else
        output = input.toMutable();

    // Check for python exception
    PyObject* type, * value, * traceback;
    PyErr_Fetch(&type, &value, &traceback);
    if (value) {
        PyErr_NormalizeException(&type, &value, &traceback);
        auto py_str = PyObject_Str(value);
        output->setProperty("python_exception", PyUnicode_AsUTF8(py_str));
        Py_CLEAR(py_str); Py_CLEAR(type); Py_CLEAR(value); Py_CLEAR(traceback);
        PyErr_Clear();
        success = false;
	}
    else {
        if (PyDict_Check(py_result)) {
			FromPython::mergeBobWithDict(output, py_result);
		}
        else
            output->setProperty("python_O", anyFromPy(py_result));
		success = true;
	}

    Py_CLEAR(py_result); 
    input = output.toImmutable();
}

// CLEAN UP OLD CODE =================================================================================================

// Removed because numpy bug makes this unusable
//void Pyfrost::Internal::finalize_pyfrost(const bool finalize, bool& pass) {
//    if (PY_INITED) {
//        Py_FinalizeEx();  // Causes crash if numpy has been imported
//        PY_INITED = 0;
//    }
//    pass = finalize;
//}