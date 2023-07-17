set(PY_VER 3.10)
string(REPLACE "." "" PY_VER_STR "${PY_VER}")

add_library( PyfrostOps SHARED)
target_compile_definitions( PyfrostOps PRIVATE PYFROST_BUILD_NODEDEF_DLL )
target_sources( PyfrostOps PRIVATE toPython.cpp fromPython.cpp)
target_sources( PyfrostOps PRIVATE Pyfrost.cpp)
target_link_libraries( PyfrostOps PUBLIC Amino::Cpp )
target_link_libraries( PyfrostOps PUBLIC Amino::Core )
target_link_libraries( PyfrostOps PUBLIC Bifrost::Object )
bifrost_set_install_rpath(PyfrostOps)
install( TARGETS PyfrostOps DESTINATION lib${PY_VER_STR} )  # folder where lib/dll files go

include_directories(${PYTHON_INCLUDE_DIRS})
find_package(PythonLibs ${PY_VER} EXACT REQUIRED)
target_link_libraries(PyfrostOps PUBLIC ${PYTHON_LIBRARIES})

include_directories("C:/Users/MAIN/AppData/Local/Programs/Python/Python${PY_VER_STR}/Lib/site-packages/numpy/core/include")
find_package(Python ${PY_VER} EXACT COMPONENTS Interpreter Development.Module NumPy)
target_link_libraries(PyfrostOps PRIVATE Python::NumPy)

# Generate and install the json files
set( headers_to_parse 
    Pyfrost.h
)

bifrost_header_parser( 
    PyfrostJsonFiles  # Name of the target.
    "compounds/Internal"                 # Install dir - relative to install prefix.
    INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR}
    INPUTS      ${headers_to_parse})

# Create config json file to load the bifrost pack
set( CONFIG_PACK_VERSION "${PROJECT_VERSION}")
set( CONFIG_PACK_NAME "${PROJECT_NAME}")
set( CONFIG_SHARED_LIBS "./lib${PY_VER_STR}")
set( CONFIG_JSON_FILES "./compounds")
set( config_json ${CMAKE_BINARY_DIR}/${PROJECT_NAME}PackConfig.json )
configure_file(PackConfig.json.in ${config_json} )
install(FILES ${config_json} DESTINATION . )
