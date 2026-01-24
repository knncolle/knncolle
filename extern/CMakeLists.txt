include(FetchContent)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^0.4.0
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^0.1.0
)

FetchContent_MakeAvailable(subpar)
FetchContent_MakeAvailable(sanisizer)
