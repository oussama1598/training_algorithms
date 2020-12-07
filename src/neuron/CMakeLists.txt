set(SOURCES Neuron.h Neuron.cpp)

add_library(neuron ${SOURCES})
target_include_directories(neuron INTERFACE ..)
target_link_libraries(neuron PRIVATE math dataset)

