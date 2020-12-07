set(SOURCES PLA.h PLA.cpp)

add_library(pla ${SOURCES})
target_include_directories(pla INTERFACE ..)
target_link_libraries(pla PRIVATE math neuron)

