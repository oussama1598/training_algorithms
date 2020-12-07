set(SOURCES ../vendor/csv.hpp DataSet.h DataSet.cpp)

add_library(dataset ${SOURCES})
target_include_directories(dataset INTERFACE ..)
