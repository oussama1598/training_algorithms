#include "Metrics.h"

Metrics::Metrics(DataSet &dataset, Neuron &neuron) : _dataset(dataset), _neuron(neuron) {}

Metrics Metrics::load(DataSet &dataset, Neuron &neuron) {
    return Metrics(dataset, neuron);
}

void Metrics::_save_to_file(const std::string &file_name, const std::string &content) {
    std::ofstream fileStream;

    fileStream.open(file_name);

    if (fileStream.is_open()) {
        fileStream << content;
    }

    fileStream.close();
}

void Metrics::save(const std::string &file_name) {
    json j;

    j["data"] = _dataset.get_inputs();
    j["labels"] = _dataset.get_labels();
    j["weights"] = _neuron.get_weight_history();
    j["losses"] = _neuron.get_losses_history();

    _save_to_file(file_name, j.dump(3));
}
