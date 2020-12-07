#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(DataSet &dataset, int epochs, double learning_rate) : Neuron(
        dataset,
        Math::sigmoid
), _max_iterations(epochs), _learning_rate(learning_rate) {}

double LogisticRegression::_calculate_loss() {
    double loss = 0;

    for (size_t i = 0; i < _inputs.size(); i++) {
        std::vector<double> &x = _inputs.at(i);

        double label = _labels.at(i);
        double prediction = predict(x);

        loss += (label * std::log(prediction)) + ((1 - label) * std::log(1 - prediction));
    }

    return -(loss / _inputs.size());
}

std::vector<double> LogisticRegression::_compute_grad() {
    std::vector<double> grad;

    for (size_t j = 0; j < _inputs.at(0).size(); j++) {
        double g = 0;

        for (size_t i = 0; i < _inputs.size(); i++) {
            std::vector<double> &x = _inputs.at(i);

            g += x[j] * (predict(x) - _labels.at(i));
        }

        grad.push_back(g);
    }

    return grad;
}

void LogisticRegression::train() {
    for (int i = 0; i < _max_iterations; i++) {
        std::vector<double> grad = _compute_grad();

        _weights = Math::sum_vectors(
                _weights,
                Math::scalar_product(grad, -1 * _learning_rate)
        );

        _weights_history.emplace_back(_weights);
        _losses_history.push_back(_calculate_loss());
    }
}
