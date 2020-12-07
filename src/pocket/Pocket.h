#pragma once


#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>
#include <neuron/Neuron.h>

class Pocket : public Neuron {
private:
    int _max_iterations = 1000;

private:
    double _calculate_loss() override;

    double _calculate_loss(std::vector<double> &weights);

public:

    Pocket(DataSet &dataset, int epochs);

    double predict(std::vector<double> x) override;

    double predict(std::vector<double> x, std::vector<double> &weights);

    void train() override;
};

