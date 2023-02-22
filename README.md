# Installation
### Compound Lib
Extract `Pyfrost.zip` to the user's compounds directory or other preferred location. Open `Maya.env` and add the following variables:

    BIFROST_LIB_CONFIG_FILES = <pyfrost location>/PyfrostPackConfig.json
    PYFROST_MODULE_PATH = <pyfrost location>/PyfrostModule
___
### Install Python
A supported version of Python must be installed and in the PATH envar. To verify this, enter `python --version` in a Command Prompt.

Included are builds supporting Python 3.10.x and 3.11.x. The config points to 3.11 by default, to use the 3.10 change `./lib311` to `./lib310`  in `PyfrostPackConfig.json` .
___
### Communication with Maya
Since Pyfrost is separate from Maya, calls to maya.cmds or getting/setting Maya variables cannot be done directly. Pyfrost defines the following functions for sending commands to Maya via commandPort.

*     maya(cmd) -> int     # Add command to queue
      mayaExec() -> None   # Execute command queue

    Several commandPort calls at once may cause an overflow error. `maya()` adds the command string to a queue, returning its index. `mayaExec()` sends the list to Maya for execution in a single commandPort call.

*     mayar() -> list   # Get list of return values
    Maya has to wait for Bifrost to finish evaluating before it can execute the command queue, so it is not possible to receive a command's return value immediately, instead the return values are saved to a tmp file. `mayar()` reads this file and returns the list of values. This means values from `mayar()` will always lag behind by one evaluation.
    

*     print(msg)  # Native print override that prints to Maya's script editor
      warn(msg)   # Display warning message in Maya
      error(msg)  # Display error message in Maya


These commands require `PYFROST_MAYA_PORT` is defined and its respective port is open. Adding the following to `userSetup.py` will perform this task automatically on startup.
    
    from maya import cmds
    for x in range(4440, 4460):  # open available port in the given range
        try:
            mc.commandPort(name=f":{x}")
            import os
            os.environ['PYFROST_MAYA_PORT'] = str(x)
            break
        except:
            pass
___
## Pyfrost Module
`PyfrostModule/PyfrostIO.py` initialized the environment in which Pyfrost interacts with Bifrost. This includes the core exec functions, as well common imports and utility functions. The user can extend this file directly
or add modules to the folder which will be available via import command. 

WARNING: The module is initialized without exception handling. An error would cause Maya to crash on the first pyfrost node evaluation.
___
## Missing/Feature Goals
* 2D/3D array support (current workaround is to use array of objects with array in property)
* Use array module or numpy for faster arrays 
* Option to specify output data type ie: python int -> uint
* Option for overriding python interpreter/environement
