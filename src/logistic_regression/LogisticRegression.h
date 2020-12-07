#pragma once

#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>
#include <neuron/Neuron.h>

class LogisticRegression : public Neuron {
private:
    int _max_iterations = 1000;
    double _learning_rate = 0.01;

private:
    double _calculate_loss() override;

    std::vector<double> _compute_grad();

public:

    LogisticRegression(DataSet &dataset, int epochs, double learning_rate);

    void train() override;
};

