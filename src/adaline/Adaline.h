#pragma once


#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>
#include <neuron/Neuron.h>

class Adaline : public Neuron {
private:
    int _max_iterations = 1000;

private:
    double _calculate_loss() override;

public:
    Adaline(DataSet &dataset, int epochs);

    void train() override;
};

