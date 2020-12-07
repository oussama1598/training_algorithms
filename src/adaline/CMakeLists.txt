set(SOURCES Adaline.h Adaline.cpp)

add_library(adaline ${SOURCES})
target_include_directories(adaline INTERFACE ..)
target_link_libraries(adaline PRIVATE math neuron)

