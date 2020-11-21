#include "Adaline.h"

Adaline::Adaline(std::vector<std::vector<double>> &inputs, std::vector<double> &labels)
        : _inputs(inputs),
          _labels(labels) {

    for (size_t i = 0; i < _inputs[0].size() + 1; i++)
        _weights.push_back(
                _distribution(_generator)
        );

    for (auto &_input : _inputs)
        _input.push_back(_bias);

    _weights_history.emplace_back(_weights);

}

double Adaline::_calculate_loss() {
    double loss = 0;

    for (size_t i = 0; i < _inputs.size(); i++) {
        std::vector<double> &x = _inputs.at(i);

        double label = _labels.at(i);
        double dot_product = predict(x);

        loss += std::pow((label - dot_product), 2);
    }

    return loss / _inputs.size();
}

void Adaline::train() {
    for (int i = 0; i < _max_iterations; i++) {
        for (size_t j = 0; j < _inputs.size(); j++) {
            std::vector<double> &x = _inputs.at(j);

            double label = _labels.at(j);
            double dot_product = predict(x);

            double error = label - dot_product;

            if (error != 0) {
                _weights = Math::sum_vectors(
                        _weights,
                        Math::scalar_product(x, 2 * error)
                );

                _weights_history.emplace_back(_weights);
            }
        }
    }
}

double Adaline::predict(std::vector<double> x) {
    return Math::sign(Math::dot_product(x, _weights));
}
