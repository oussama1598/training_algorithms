#pragma once


#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>

class Adaline {
private:
    std::default_random_engine _generator;
    std::uniform_real_distribution<double> _distribution{-1.0, 1.0};

    double _bias = 1;

    std::vector<std::vector<double>> _inputs;
    std::vector<double> _labels;
    std::vector<double> _weights;

private:
    double _calculate_loss();

public:
    Adaline(std::vector<std::vector<double>> &inputs, std::vector<double> &labels);

    void train();

    double predict(std::vector<double> x);
};

