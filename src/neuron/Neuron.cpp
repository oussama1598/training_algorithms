#include "Neuron.h"

Neuron::Neuron(DataSet &dataset, activation activation_function)
        : _inputs(dataset.get_inputs()),
          _labels(dataset.get_labels()),
          _activation_function(activation_function) {

    for (size_t i = 0; i < _inputs[0].size() + 1; i++)
        _weights.push_back(
                _distribution(_generator)
        );

    for (auto &_input : _inputs)
        _input.push_back(_bias);

    _weights_history.emplace_back(_weights);
}

double Neuron::predict(std::vector<double> x) {
    return _activation_function(Math::dot_product(x, _weights));
}
