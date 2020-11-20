#include <iostream>
#include <perceptron/Perceptron.h>

int main([[maybe_unused]]int argc, [[maybe_unused]]char **argv) {
    std::vector<std::vector<double>> inputs{
            {1, 2},
            {2, 1}
    };

    std::vector<double> labels{
            1, -1
    };


    Perceptron perceptron{inputs, labels};

    perceptron.train();

    std::cout << perceptron.predict({1, 2}) << std::endl;
}
