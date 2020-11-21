#include "Pocket.h"

Pocket::Pocket(std::vector<std::vector<double>> &inputs, std::vector<double> &labels)
        : _inputs(inputs),
          _labels(labels) {

    for (size_t i = 0; i < _inputs[0].size() + 1; i++)
        _weights.push_back(
                _distribution(_generator)
        );

    for (auto &_input : _inputs)
        _input.push_back(_bias);

    _weights_history.emplace_back(_weights);
    _losses_history.push_back(_calculate_loss(_weights));
}

double Pocket::_calculate_loss(std::vector<double> &weights) {
    double loss = 0;

    for (size_t i = 0; i < _inputs.size(); i++) {
        std::vector<double> &x = _inputs.at(i);

        double label = _labels.at(i);

        if (predict(x, weights) != label)
            loss += 1;
    }

    return loss / _inputs.size();
}

void Pocket::train() {
    for (int j = 0; j < _max_iterations; j++) {
        std::vector<double> weights(_weights);

        for (int k = 0; k < 200; k++) {
            for (size_t i = 0; i < _inputs.size(); i++) {
                std::vector<double> &x = _inputs.at(i);

                double label = _labels.at(i);

                if (predict(x, weights) != label) {
                    weights = Math::sum_vectors(
                            weights,
                            Math::scalar_product(x, label)
                    );
                }
            }
        }

        if (_calculate_loss(weights) < _calculate_loss(_weights)) {
            _weights = weights;

            _weights_history.emplace_back(_weights);
        }

        _losses_history.push_back(_calculate_loss(_weights));
    }
}

double Pocket::predict(std::vector<double> x) {
    return Math::sign(Math::dot_product(x, _weights));
}

double Pocket::predict(std::vector<double> x, std::vector<double> &weights) {
    return Math::sign(Math::dot_product(x, weights));
}
