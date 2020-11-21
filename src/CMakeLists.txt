add_subdirectory(math)
add_subdirectory(perceptron)
add_subdirectory(adaline)

add_executable(perceptron_bin main.cpp)
target_link_libraries(perceptron_bin PRIVATE perceptron)
