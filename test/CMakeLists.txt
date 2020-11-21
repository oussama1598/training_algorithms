add_executable(learning_test learning_test.cpp perceptron.h perceptron.cpp adaline.h adaline.cpp helpers.h helpers.cpp pocket.h pocket.cpp)
target_link_libraries(learning_test perceptron adaline pocket CONAN_PKG::Catch2 CONAN_PKG::nlohmann_json)
add_test(NAME learning_test COMMAND $<TARGET_FILE:learning_test>)
