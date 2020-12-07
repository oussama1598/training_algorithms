#pragma once


#include <string>
#include <vector>
#include "../vendor/csv.hpp"

struct DataSet {
private:
    std::vector<std::vector<double>> _inputs;
    std::vector<double> _labels;

public:
    DataSet(std::vector<std::vector<double>> &inputs, std::vector<double> &labels);

    static DataSet from_raw(std::vector<std::vector<double>> &inputs, std::vector<double> &labels);

    static DataSet
    from_csv(const std::string &file_name, const std::vector<std::string> &inputs_headers,
             const std::string &output_header);

    std::vector<std::vector<double>> get_inputs() { return _inputs; }

    std::vector<double> get_labels() { return _labels; }
};

