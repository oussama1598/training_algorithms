set(SOURCES Perceptron.h Perceptron.cpp)

add_library(perceptron ${SOURCES})
target_include_directories(perceptron INTERFACE ..)
target_link_libraries(perceptron PRIVATE math)

