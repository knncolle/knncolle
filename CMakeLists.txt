cmake_minimum_required(VERSION 3.14...3.25)

project(knncolle
    VERSION 1.0.0
    DESCRIPTION "Collection of k-nearest neighbors algorithms in C++"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(knncolle INTERFACE)
add_library(ltla::knncolle ALIAS knncolle)

target_include_directories(knncolle INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ltla>)
target_compile_features(knncolle INTERFACE cxx_std_17)

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
option(KNNCOLLE_FETCH_EXTERN "Automatically fetch knncolle's external dependencies." ON)
if(KNNCOLLE_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(ltla_kmeans CONFIG REQUIRED)
    find_package(Annoy CONFIG REQUIRED)
    find_package(hnswlib REQUIRED)
endif()

target_link_libraries(knncolle INTERFACE ltla::kmeans Annoy::Annoy hnswlib)

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
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ltla)

install(TARGETS knncolle
    EXPORT knncolleTargets)

install(EXPORT knncolleTargets
    FILE ltla_knncolleTargets.cmake
    NAMESPACE ltla::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_knncolle)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_knncolleConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_knncolle)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_knncolleConfig.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Findhnswlib.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_knncolle)
