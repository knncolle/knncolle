cmake_minimum_required(VERSION 3.14)

project(knncolle
    VERSION 2.2.0
    DESCRIPTION "Collection of k-nearest neighbors algorithms in C++"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(knncolle INTERFACE)
add_library(knncolle::knncolle ALIAS knncolle)

target_include_directories(knncolle INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/knncolle>)
target_compile_features(knncolle INTERFACE cxx_std_17)

# Dependencies
option(KNNCOLLE_FETCH_EXTERN "Automatically fetch knncolle's external dependencies." ON)
if(KNNCOLLE_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(ltla_kmeans 3.0.0 CONFIG REQUIRED)
endif()

target_link_libraries(knncolle INTERFACE ltla::kmeans)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(KNNCOLLE_TESTS "Build knncolle's test suite." ON)
else()
    option(KNNCOLLE_TESTS "Build knncolle's test suite." OFF)
endif()

if(KNNCOLLE_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/knncolle)

install(TARGETS knncolle
    EXPORT knncolleTargets)

install(EXPORT knncolleTargets
    FILE knncolle_knncolleTargets.cmake
    NAMESPACE knncolle::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/knncolle_knncolle)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolleConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/knncolle_knncolle)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolleConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolleConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolleConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/knncolle_knncolle)
