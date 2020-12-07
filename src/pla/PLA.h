#pragma once


#include <vector>
#include <random>
#include <iostream>
#include <math/Math.h>
#include <neuron/Neuron.h>

class PLA : public Neuron {
private:
    double _calculate_loss() override;

public:
    explicit PLA(DataSet &dataset);

    void train() override;
};

