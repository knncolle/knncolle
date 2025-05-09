include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/90a41521142c978131f38c6da07b4eb96a9f1ff6.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

option(CODE_COVERAGE "Enable coverage testing" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

include(GoogleTest)

macro(decorate_test target)
    target_link_libraries(
        ${target}
        gtest_main
        knncolle
    )

    target_compile_options(${target} PRIVATE -Wall -Werror -Wextra -Wpedantic)

    if(DO_CODE_COVERAGE)
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

add_executable(
    libtest
    src/Bruteforce.cpp
    src/Vptree.cpp
    src/L2Normalized.cpp
    src/find_nearest_neighbors.cpp
    src/misc.cpp
)
decorate_test(libtest)

add_executable(
    cuspartest
    src/find_nearest_neighbors.cpp
)
decorate_test(cuspartest)
target_compile_definitions(cuspartest PRIVATE TEST_CUSTOM_PARALLEL=1)
