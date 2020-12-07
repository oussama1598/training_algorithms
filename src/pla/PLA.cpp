#include "PLA.h"

PLA::PLA(DataSet &dataset) : Neuron(
        dataset,
        Math::sign
) {}

double PLA::_calculate_loss() {
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

void PLA::train() {
    while (_calculate_loss() != 0) {
        for (size_t i = 0; i < _inputs.size(); i++) {
            std::vector<double> &x = _inputs.at(i);

            double label = _labels.at(i);

            if (predict(x) * label <= 0) {
                _weights = Math::sum_vectors(
                        _weights,
                        Math::scalar_product(x, label)
                );

                _weights_history.emplace_back(_weights);
            }
        }

        _losses_history.push_back(_calculate_loss());
    }
}
