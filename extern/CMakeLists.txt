include(FetchContent)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG master # ^3.0.0
)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^0.2.1
)

FetchContent_MakeAvailable(kmeans)
FetchContent_MakeAvailable(subpar)
