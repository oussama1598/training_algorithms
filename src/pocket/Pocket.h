#pragma once


#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>

class Pocket {
private:
    std::default_random_engine _generator;
    std::uniform_real_distribution<double> _distribution{-1.0, 1.0};

    double _bias = 1;
    int _max_iterations = 1000;

    std::vector<std::vector<double>> _weights_history;
    std::vector<double> _losses_history;

    std::vector<std::vector<double>> _inputs;
    std::vector<double> _labels;
    std::vector<double> _weights;

private:
    double _calculate_loss(std::vector<double> &weights);

public:
    Pocket(std::vector<std::vector<double>> &inputs, std::vector<double> &labels);

    void train();

    double predict(std::vector<double> x);

    double predict(std::vector<double> x, std::vector<double> &weights);

    inline std::vector<std::vector<double>> &get_weight_history() { return _weights_history; }

    inline std::vector<double> &get_losses_history() { return _losses_history; }
};

