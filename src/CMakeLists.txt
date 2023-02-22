#-
# ===========================================================================
# Copyright 2022 Autodesk, Inc. All rights reserved.
#
# Use of this software is subject to the terms of the Autodesk license
# agreement provided at the time of installation or download, or which
# otherwise accompanies this software in either electronic or hard copy form.
# ===========================================================================
#+

# Build and install the library
add_library( PyfrostOps SHARED)
target_compile_definitions( PyfrostOps PRIVATE PYFROST_BUILD_NODEDEF_DLL )
target_sources( PyfrostOps PRIVATE Pyfrost.cpp)
target_link_libraries( PyfrostOps PUBLIC Amino::Cpp )
target_link_libraries( PyfrostOps PUBLIC Amino::Core )
target_link_libraries( PyfrostOps PUBLIC Bifrost::Object )
bifrost_set_install_rpath(PyfrostOps)
install( TARGETS PyfrostOps DESTINATION lib )

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(PyfrostOps PUBLIC ${PYTHON_LIBRARIES})

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
set( CONFIG_SHARED_LIBS "./lib")
set( CONFIG_JSON_FILES "./compounds")
set( config_json ${CMAKE_BINARY_DIR}/${PROJECT_NAME}PackConfig.json )
configure_file(PackConfig.json.in ${config_json} )
install(FILES ${config_json} DESTINATION . )
