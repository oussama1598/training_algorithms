set(SOURCES Math.h Math.cpp)

add_library(math ${SOURCES})
target_include_directories(math INTERFACE ..)
