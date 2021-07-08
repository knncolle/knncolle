cmake_minimum_required(VERSION 3.14)

project(knncolle
    VERSION 1.0.0
    DESCRIPTION "Collection of k-nearest neighbors algorithms in C++"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(knncolle INTERFACE)

target_include_directories(knncolle INTERFACE include/)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()
