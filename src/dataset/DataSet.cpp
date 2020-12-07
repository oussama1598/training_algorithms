#include "DataSet.h"

DataSet::DataSet(std::vector<std::vector<double>> &inputs, std::vector<double> &labels) : _inputs(
        inputs), _labels(labels) {}

DataSet DataSet::from_raw(std::vector<std::vector<double>> &inputs, std::vector<double> &labels) {
    return DataSet(inputs, labels);
}

DataSet
DataSet::from_csv(const std::string &file_name, const std::vector<std::string> &inputs_headers,
                  const std::string &output_header) {
    std::vector<std::vector<double>> inputs;
    std::vector<double> labels;

    csv::CSVReader reader(file_name);

    for (auto &row: reader) {
        std::vector<double> x;

        auto label = row[output_header].get<double>();

        x.reserve(inputs_headers.size());

        for (auto &input_name: inputs_headers) {
            x.push_back(row[input_name].get<double>());
        }

        inputs.push_back(x);
        labels.push_back(label);
    }

    return DataSet(inputs, labels);
}
