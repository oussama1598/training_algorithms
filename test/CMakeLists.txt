add_executable(learning_test learning_test.cpp perceptron.h perceptron.cpp adaline.h adaline.cpp helpers.h helpers.cpp pocket.h pocket.cpp regression.h regression.cpp)
target_link_libraries(learning_test perceptron adaline pocket regression CONAN_PKG::Catch2 CONAN_PKG::nlohmann_json CONAN_PKG::fast-cpp-csv-parser)
add_test(NAME learning_test COMMAND $<TARGET_FILE:learning_test>)
