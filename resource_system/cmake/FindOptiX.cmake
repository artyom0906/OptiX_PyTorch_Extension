# Find OptiX
#
# This module defines:
# OptiX_FOUND
# OptiX_INCLUDE_DIR
# OptiX_LIBRARY
#

# Try to find OptiX SDK
find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS
    ${OptiX_INSTALL_DIR}/include
    $ENV{OptiX_INSTALL_DIR}/include
    /usr/include
    /usr/local/include
    DOC "The OptiX include directory"
)

# Define macros for version and component checks
if(OptiX_INCLUDE_DIR)
    # Check the OptiX version
    set(OptiX_VERSION_FILE "${OptiX_INCLUDE_DIR}/optix_function_table.h")
    if(EXISTS ${OptiX_VERSION_FILE})
        file(READ "${OptiX_VERSION_FILE}" OptiX_VERSION_CONTENT)
        if(OptiX_VERSION_CONTENT MATCHES "#define OPTIX_VERSION ([0-9]+)")
            set(OptiX_VERSION ${CMAKE_MATCH_1})
            set(OptiX_VERSION_MAJOR ${OptiX_VERSION})
        endif()
    endif()
endif()

# Handle the REQUIRED argument and set OptiX_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS
        OptiX_INCLUDE_DIR
    VERSION_VAR
        OptiX_VERSION
)

# Mark as advanced
mark_as_advanced(OptiX_INCLUDE_DIR)

# Create an imported target for OptiX
if(OptiX_FOUND AND NOT TARGET OptiX::OptiX)
    add_library(OptiX::OptiX INTERFACE IMPORTED)
    set_target_properties(OptiX::OptiX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE_DIR}"
    )
endif()