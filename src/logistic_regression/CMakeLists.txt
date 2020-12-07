set(SOURCES LogisticRegression.h LogisticRegression.cpp)

add_library(logistic_regression ${SOURCES})
target_include_directories(logistic_regression INTERFACE ..)
target_link_libraries(logistic_regression PRIVATE math neuron)
