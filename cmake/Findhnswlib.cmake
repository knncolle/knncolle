# Find hnswlib

find_path(hnswlib_INCLUDE_DIR "hnswlib/hnswlib.h")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hnswlib
    REQUIRED_VARS hnswlib_INCLUDE_DIR)

if(hnswlib_FOUND)
    set(hnswlib_INCLUDE_DIRS ${hnswlib_INCLUDE_DIR})
    if(NOT TARGET hnswlib)
        add_library(hnswlib INTERFACE IMPORTED)
        set_target_properties(hnswlib PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${hnswlib_INCLUDE_DIRS}")
    endif()
endif()

mark_as_advanced(hnswlib_INCLUDE_DIR)
