#include "Perceptron.h"

Perceptron::Perceptron(std::vector<std::vector<double>> &inputs, std::vector<double> &labels)
        : _inputs(inputs),
          _labels(labels) {

    for (size_t i = 0; i < _inputs[0].size() + 1; i++)
        _weights.push_back(
                _distribution(_generator)
        );

    for (auto &_input : _inputs)
        _input.push_back(_bias);

}

double Perceptron::_calculate_loss() {
    double loss = 0;

    for (size_t i = 0; i < _inputs.size(); i++) {
        std::vector<double> &x = _inputs.at(i);

        double label = _labels.at(i);
        double predicted_label = predict(x);

        if (predicted_label != label)
            loss += 1;
    }

    return loss / _inputs.size();
}

void Perceptron::train() {
    while (_calculate_loss() != 0) {
        for (size_t i = 0; i < _inputs.size(); i++) {
            std::vector<double> &x = _inputs.at(i);

            double label = _labels.at(i);

            if (predict(x) * label <= 0)
                _weights = Math::sum_vectors(
                        _weights,
                        Math::scalar_product(x, label)
                );
        }
    }
}

double Perceptron::predict(std::vector<double> x) {
    return Math::sign(Math::dot_product(x, _weights));
}
