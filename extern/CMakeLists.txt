include(ExternalProject)

set(ANNOY_DIR ${CMAKE_CURRENT_BINARY_DIR}/annoy)
ExternalProject_Add(annoy-source
    SOURCE_DIR ${ANNOY_DIR}
    GIT_REPOSITORY https://github.com/spotify/annoy/
    GIT_TAG f20ddef54e775448d621621bc7d2ec386dbf0dc0
    BUILD_COMMAND ""
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
)

add_library(annoy INTERFACE IMPORTED GLOBAL)
target_include_directories(annoy INTERFACE ${ANNOY_DIR}/src)
add_dependencies(annoy annoy-source)
