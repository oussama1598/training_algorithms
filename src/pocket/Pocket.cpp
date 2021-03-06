#include "Pocket.h"

Pocket::Pocket(DataSet &dataset, int epochs) : Neuron(
        dataset,
        Math::sign
), _max_iterations(epochs) {}

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
    std::vector<double> weights(_weights);

    for (int j = 0; j < _max_iterations; j++) {
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

        if (_calculate_loss(weights) < _calculate_loss(_weights)) {
            _weights = weights;

            _weights_history.emplace_back(_weights);
        }

        _losses_history.push_back(_calculate_loss(_weights));
    }
}

double Pocket::predict(std::vector<double> x, std::vector<double> &weights) {
    return Math::sign(Math::dot_product(x, weights));
}

double Pocket::_calculate_loss() {
    return 0;
}

double Pocket::predict(std::vector<double> x) {
    return Neuron::predict(std::move(x));
}
