#include <iostream>
#include <perceptron/Perceptron.h>

int main() {
    std::vector<std::vector<double>> inputs{
            {1.0, 1.0},
            {2.0, 1.0},
            {2.0, 2.0},
            {1.0, 3.0},
            {3.0, 3.0},
            {2.0, 4.0},
            {5.0, 4.0},
            {4.0, 5.0}
    };

    std::vector<double> labels{
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0
    };


    Perceptron perceptron{inputs, labels};

    perceptron.train();

    std::cout << perceptron.predict({2.0, 1.0, 1.0}) << std::endl;
}
