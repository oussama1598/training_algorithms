set(SOURCES LinearRegression.h LinearRegression.cpp)

add_library(linear_regression ${SOURCES})
target_include_directories(linear_regression INTERFACE ..)
target_link_libraries(linear_regression PRIVATE math neuron)
