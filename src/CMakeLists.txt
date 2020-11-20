add_subdirectory(cplotplot)

add_executable(cplotplot_bin main.cpp)
target_link_libraries(cplotplot_bin PRIVATE perceptron)
