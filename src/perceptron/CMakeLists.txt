set(SOURCES perceptron.h perceptron.cpp)

add_library(perceptron ${SOURCES})
target_include_directories(perceptron INTERFACE ..)
