set(SOURCES Metrics.h Metrics.cpp)

add_library(metrics ${SOURCES})
target_include_directories(metrics INTERFACE ..)
target_link_libraries(metrics PUBLIC neuron CONAN_PKG::nlohmann_json)

