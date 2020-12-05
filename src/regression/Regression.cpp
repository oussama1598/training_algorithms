#include "Regression.h"

Regression::Regression(std::vector<std::vector<double>> &inputs, std::vector<double> &labels)
        : _inputs(inputs),
          _labels(labels) {

    for (size_t i = 0; i < _inputs[0].size() + 1; i++)
        _weights.push_back(
                _distribution(_generator)
        );

    for (auto &_input : _inputs)
        _input.push_back(_bias);


    _weights_history.emplace_back(_weights);
    _losses_history.push_back(_calculate_loss());
}

double Regression::_calculate_loss() {
    double loss = 0;

    for (size_t i = 0; i < _inputs.size(); i++) {
        std::vector<double> &x = _inputs.at(i);

        double label = _labels.at(i);
        double dot_product = predict(x);

        loss += std::pow((label - dot_product), 2);
    }

    return loss / _inputs.size();
}

std::vector<double> Regression::_compute_grad_MSE() {
    std::vector<double> grad;

    for (size_t j = 0; j < _inputs.at(0).size(); j++) {
        double g = 0;

        for (size_t i = 0; i < _inputs.size(); i++) {
            std::vector<double> &x = _inputs.at(i);

            g += x[j] *  (predict(x) - _labels.at(i));
        }

        g *= (2.0 / _inputs.size());

        grad.push_back(g);
    }

    return grad;
}

void Regression::train() {
    for (int i = 0; i < _max_iterations; i++) {
        std::vector<double> grad = _compute_grad_MSE();

        _weights = Math::sum_vectors(
                _weights,
                Math::scalar_product(grad, -1 * _learning_rate)
        );

        _weights_history.emplace_back(_weights);
        _losses_history.push_back(_calculate_loss());
    }
}

double Regression::predict(std::vector<double> x) {
    return Math::dot_product(x, _weights);
}
