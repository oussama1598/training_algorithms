#pragma once

#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>
#include <neuron/Neuron.h>

class LinearRegression : public Neuron {
private:
    int _max_iterations = 100;
    double _learning_rate = 0.0000001;

private:
    double _calculate_loss() override;

    std::vector<double> _compute_grad_MSE();

public:

    LinearRegression(DataSet &dataset, int epochs, double learning_rate);

    void train() override;
};

