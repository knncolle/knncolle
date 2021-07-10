include(FetchContent)

FetchContent_Declare(
  Annoy 
  GIT_REPOSITORY https://github.com/LTLA/Annoy
  GIT_TAG cmake 
)

FetchContent_MakeAvailable(Annoy)

