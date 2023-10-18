# Pyfrost
Python embedded in a Bifrost Operator. Created and tested with `Maya 2024.1`, `Bifrost 2.7.1.0`, `Python 3.11.4`.

Built with Python 3.11, a version of which is required to be installed and in the PATH envar.

## Installation

1. Copy the `Pyfrost` folder to the user's compounds directory or other preferred location.
2. Open `Maya.env` and add the following variables:

        BIFROST_LIB_CONFIG_FILES = <pyfrost location>/PyfrostPackConfig.json
        PYFROST_MODULE_PATH = <pyfrost location>/modules
    (if `BIFROST_LIB_CONFIG_FILES` is defined, append to the existing value using `;`)


3. Install a version of Python 3.11, with "Add Python to PATH" checked.
4. Install Numpy: `pip install numpy`.
5. (Optional) Enable `MayaSM` by adding the following to `userSetup.py`:

        import os, sys
        sys.path.append(os.environ["PYFROST_MODULE_PATH"])
        import MayaSM
   
6. (Optional) Pyfrost deals mostly with numpy objects, so it can be useful to have numpy installed to Maya's environment as well:

        "<maya install dir>/bin/mayapy.exe" -m pip install numpy

___

## Pyfrost Compound Usage

```
script
    python script to execute

global
    exec in local or global scope

consume_input
    add the pyfrost's output to the input object, or construct a new output object
```

#### Predefined objects

```
np
    Pyfrost imports numpy as np when initialized.

I
    This variable contains the Bifrost input converted to python

O
    An empty dict, any values added to this dict will be converted to properties on the output Bifrost object.
    This can also be re-assigned directly to some other value/type.
    If `O` type is not dict then it will be added to output as a property called 'python_O'

struct(array)
    Designate that a np.ndarray represents a single value with struct data type, ie. vector or matrix.

SMD
    dict shared with Maya (if MayaSM is imported on startup)
```

#### Bifrost/Pyfrost Type Conversion

Bifrost -> Python
```
    Object -> dict
    String -> str
    Simple numeric -> np.generic (numpy scalar)
    Vector -> np.ndarray (1D array)
    Matrix -> np.ndarray (2D array)
    Bifrost array -> np.ndarray
    *other -> None
```

Python -> Bifrost
```
    dict -> Object
    str -> String
    np.generic -> Simple numeric
    int -> long
    float -> float
    np.ndarray -> Bifrost array / vector / matrix
    list -> Bifrost array
    tuple -> Bifrost array
    *other -> String (using str(py_object))
```

Some np.ndarray type conversions are ambiguous, eg. should a float[3] array be converted to a Bifrost float array, or float3 vector.
By default the np.ndarray is always converted to an array. `struct(array)` adds metadata to the array's dtype, designating it as a single value of struct type. 

> [!NOTE]
> A _struct_ np.ndarray must have a shape matching some Bifrost::Math type, otherwise metadata is ignored and default array conversion is used.

___

### MayaSM / Communication with Maya
MayaSM manages a shared memory dictionary for sending commands/data between Pyfrost and Maya.

Pyfrost script:

    # In pyfrost SMD can be used without the namespace
    SMD["message"] = "hello"

Maya script:

    print(MayaSM.SMD["message"])
    >>> hello
    
___

The module includes functions for queuing/executing scripts:

```
cmd(script)
    Queues a script to be executed as if pasted to the script editor.

mel(script)
    Similar to `cmd` but for mel commands

print(msg, mel_print=True)
    print from Pyfrost to the script editor

warn(msg)
    print from Pyfrost to the script editor as warning

error(msg)
    print from Pyfrost to the script editor as error
```

Maya still needs to be told when to execute queued commands. This can be done with `MayaSM.execPending()`. There're 2 options for performing this action automatically.

```
MayaSM.startTimer(interval=0.1)
    Starts a timer that checks for and executes pending commands at the given interval. Convenient if the expected commands are not time critical.

MayaSM.dgExecutor(graph="")
    Creates an expression node and dummy graph to pull on it. The target graph can be selected or specified by the graph arg.
    The expression calls `MayaSM.execPending()` whenever the given graph evaluates, meaning commands are queued and executed reliably as one evaluation.
```
___

## Bob Explorer
This is simple UI for viewing Bifrost data. If installation step 6. is met, the UI can be opened with:

    import BobExplorer
    BobExplorer.openControl()
