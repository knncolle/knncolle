include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest
    src/Bruteforce.cpp
    src/Vptree.cpp
    src/Kmknn.cpp
    src/L2Normalized.cpp
    src/find_nearest_neighbors.cpp
    src/misc.cpp
)

include(GoogleTest)
option(CODE_COVERAGE "Enable coverage testing" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

macro(decorate_test name)
    target_link_libraries(
        ${name}
        gtest_main
        knncolle
    )

    target_compile_options(${name} PRIVATE -Wall -Werror -Wextra -Wpedantic)

    if(DO_CODE_COVERAGE)
        target_compile_options(${name} PRIVATE -O0 -g --coverage)
        target_link_options(${name} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${name})
endmacro()

decorate_test(libtest)

# Adding checks for parallelization.
find_package(OpenMP)
if (OpenMP_FOUND)
    add_executable(
        omp_test
        src/find_nearest_neighbors.cpp
    )
    target_link_libraries(omp_test OpenMP::OpenMP_CXX)
    decorate_test(omp_test)
endif()

add_executable(
    custom_parallel 
    src/find_nearest_neighbors.cpp
) 
decorate_test(custom_parallel)
target_compile_definitions(custom_parallel PRIVATE TEST_KNNCOLLE_CUSTOM_PARALLEL=1)
