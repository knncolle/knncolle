include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest
    src/BruteForce.cpp
    src/VpTree.cpp
    src/Kmknn.cpp
    src/Annoy.cpp
    src/Hnsw.cpp
    src/find_nearest_neighbors.cpp
)

target_link_libraries(
    libtest
    gtest_main
    knncolle
)

set(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

# Adding checks for parallelization.
add_executable(
    omp_test
    src/find_nearest_neighbors.cpp
) 

target_link_libraries(
    omp_test
    gtest_main
    knncolle
)

set(USE_OPENMP OFF CACHE BOOL "Compile with OpenMP support")
if (USE_OPENMP)
    find_package(OpenMP)
    target_link_libraries(omp_test OpenMP::OpenMP_CXX)
endif()

add_executable(
    custom_parallel 
    src/find_nearest_neighbors.cpp
) 

target_link_libraries(
    custom_parallel 
    gtest_main
    knncolle
)

set_target_properties(custom_parallel PROPERTIES COMPILE_DEFINITIONS "TEST_KNNCOLLE_CUSTOM_PARALLEL")

# Discovering all tests.
include(GoogleTest)
gtest_discover_tests(libtest)
gtest_discover_tests(omp_test)
gtest_discover_tests(custom_parallel)
