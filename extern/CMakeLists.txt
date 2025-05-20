include(FetchContent)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^0.4.0
)

FetchContent_MakeAvailable(subpar)
