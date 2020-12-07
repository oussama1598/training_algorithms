set(SOURCES Pocket.h Pocket.cpp)

add_library(pocket ${SOURCES})
target_include_directories(pocket INTERFACE ..)
target_link_libraries(pocket PRIVATE math neuron)

