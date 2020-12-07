#pragma once


#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>
#include <functional>
#include <dataset/DataSet.h>

typedef double (*activation)(double number);

class Neuron {
protected:
    std::default_random_engine _generator;
    std::uniform_real_distribution<double> _distribution{-1.0, 1.0};

    double _bias = 1;

    std::vector<std::vector<double>> _weights_history;
    std::vector<double> _losses_history;

    std::vector<std::vector<double>> _inputs;
    std::vector<double> _labels;
    std::vector<double> _weights;

    activation _activation_function;

protected:
    virtual double _calculate_loss() = 0;

public:

    Neuron(DataSet &dataset, activation activation_function);

    inline std::vector<std::vector<double>> &get_weight_history() { return _weights_history; }

    inline std::vector<double> &get_losses_history() { return _losses_history; }

    virtual double predict(std::vector<double> x);

    virtual void train() = 0;
};

