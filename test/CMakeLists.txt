add_executable(learning_test learning_test.cpp pla.h pla.cpp adaline.h adaline.cpp pocket.h pocket.cpp linear_regression.h linear_regression.cpp logistic_regression.h logistic_regression.cpp)
target_link_libraries(learning_test dataset pla adaline pocket linear_regression logistic_regression metrics CONAN_PKG::Catch2)
add_test(NAME learning_test COMMAND $<TARGET_FILE:learning_test>)
