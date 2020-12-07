#pragma once


#include <neuron/Neuron.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Metrics {
    DataSet &_dataset;
    Neuron &_neuron;

private:
    void _save_to_file(const std::string &file_name, const std::string &content);

public:

    explicit Metrics(DataSet &dataset, Neuron &neuron);

    static Metrics load(DataSet &dataset, Neuron &neuron);

    void save(const std::string &file_name);
};

